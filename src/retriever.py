"""Retrievers: BaseRetriever + HybridRetriever + FactRetriever"""

import json
import re
import numpy as np
from typing import List, Dict
import networkx as nx
from src.utils.prompts import prompt_extract_entities_str
from database.milvus import MilvusDB


class BaseRetriever:
    """Base retriever class: shared utilities + hybrid_retrieve template method."""

    def __init__(self, vector_store=None, memory_graph=None, llm=None,
                 reranker=None, chunk_registry=None, fusion=None,
                 mode="hybrid"):
        self.vector_store = vector_store
        self.memory_graph = memory_graph
        self.persistent_graph = memory_graph.persistent_graph if memory_graph else None
        self.llm = llm
        self.reranker = reranker
        self.chunk_registry = chunk_registry or {}
        self.fusion = fusion
        self.mode = mode  # "hybrid" or "graph_only"

    # ── Shared Utilities ──────────────────────────────────────────

    def _embed_text(self, text: str) -> np.ndarray:
        if not self.llm:
            raise ValueError("LLM env is required for embeddings.")
        return np.array(self.llm.embed_model.get_embedding(text), dtype=float)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _get_chunk_text(self, chunk_ids: List[str]) -> List[str]:
        texts = []
        for cid in chunk_ids:
            info = self.vector_store.get_chunk_text(cid)
            texts.append(info.get("text", "") if info else "")
        return texts

    def _rerank_score(self, query: str, passage: str) -> float:
        if not self.reranker or not passage:
            return 0.0
        if hasattr(self.reranker, "score"):
            return float(self.reranker.score(query, passage))
        return 0.0

    def _rehydrate_vector_hits(self, hits: List[Dict]):
        if not self.memory_graph:
            return
        for hit in hits:
            chunk_id = hit.get("id")
            if not chunk_id or not str(chunk_id).startswith("chunk_"):
                continue
            if self.memory_graph.is_chunk_in_l1(chunk_id) or self.memory_graph.is_chunk_in_l2(chunk_id):
                self.memory_graph.touch_chunk(chunk_id)
                continue
            self.memory_graph.rehydrate_chunk_from_milvus(chunk_id)

    # ── DPR ─────────────────────────────────────────────

    def retrieve_from_embedding(self, query_emb: np.ndarray, topk: int = 5):
        try:
            search_params = {"metric_type": self.vector_store.metric, "params": {"nprobe": 10}}
            output_fields = ["chunk_id"] if self.vector_store._has_field("chunk_id") else []
            result = self.vector_store.search(
                query_emb.tolist(), search_params, topk, output_fields=output_fields)
        except Exception:
            result = []
        hits = []
        for hits_group in result:
            for hit in hits_group:
                chunk_id = None
                if output_fields:
                    chunk_id = hit.entity.get("chunk_id")
                hits.append({"id": str(chunk_id) if chunk_id else str(hit.pk),
                             "score": float(hit.distance)})
        return {"embedding_hits": hits}

    # ── Template Method ─────────────────────────────────────────

    def hybrid_retrieve(self, query: str, topk: int = 10, top_entities: int = 5,
                        top_chunks: int = 15, top_rerank: int = 15, answer_topk: int = 6,
                        track_promotion: bool = True):
        """Template method: subclasses override _extract_entities / _handle_promotion."""
        query_emb = self._embed_text(query)

        # 1. DPR (skipped in graph_only mode)
        vector_hits = []
        if self.mode != "graph_only":
            embedding_res = self.retrieve_from_embedding(query_emb, topk=topk)
            vector_hits = embedding_res.get("embedding_hits", [])
            self._rehydrate_vector_hits(vector_hits)

        # 2. Entity extraction + weights (implemented by subclasses)
        extracted_entities, query_relations, entity_weights, entity_chunks = \
            self._extract_entities(query_emb, query, top_entities)

        # 3. Graph retrieval
        memory_res, persistent_res, chunk_entity_coverage = \
            self._graph_retrieve(query_emb, extracted_entities, query_relations,
                                 entity_weights, top_entities, top_chunks, track_promotion)

        # 4. Score aggregation
        graph_ranked_chunks = self._aggregate_graph_scores(
            memory_res, persistent_res, top_chunks, query)

        # 5. Fusion / graph-only mode
        if self.mode == "graph_only":
            # Graph-only retrieval: directly use graph-ranked chunks, skip DPR fusion
            final_chunks = [cid for cid, _ in graph_ranked_chunks[:answer_topk]]
        elif self.fusion:
            self.fusion._entity_weights = entity_weights
            final_chunks = self.fusion.fuse(query, vector_hits, graph_ranked_chunks,
                                            chunk_entity_coverage, top_rerank, answer_topk)
            self.fusion.handle_promotion(chunk_entity_coverage, entity_chunks, track_promotion)
        else:
            final_chunks = []

        # Track sources
        dpr_cids = {h["id"] for h in vector_hits}
        graph_cids = set()
        for m in memory_res + persistent_res:
            graph_cids.update(m.get("chunk_scores", {}).keys())
        final_cids = set(final_chunks)
        graph_in_final = final_cids & graph_cids
        dpr_in_final = final_cids & dpr_cids

        return {
            "chunks": final_chunks,
            "embedding": {"embedding_hits": vector_hits},
            "memory": memory_res,
            "persistent": persistent_res,
            "chunk_entity_coverage": chunk_entity_coverage,
            "stats": {
                "dpr_total": len(dpr_cids),
                "graph_total": len(graph_cids),
                "graph_in_final": len(graph_in_final),
                "dpr_in_final": len(dpr_in_final),
                "overlap": len(graph_cids & dpr_cids),
                "n_entities": len(extracted_entities),
                "n_chunks_covered": len(chunk_entity_coverage),
            },
        }

    # ── Subclasses Must Override ──────────────────────────────────────

    def _extract_entities(self, query_emb: np.ndarray, query: str, top_entities: int):
        """Returns (entities, relations, entity_weights, entity_chunks)"""
        raise NotImplementedError

    def _handle_promotion(self, chunk_entity_coverage, entity_chunks, track_promotion):
        """Subclasses may optionally override"""
        pass

    # ── Graph Retrieval + Score Aggregation (Shared) ─────────────────────────

    def _graph_retrieve(self, query_emb, entities, relations, entity_weights,
                        top_entities, top_chunks, track_promotion):
        memory_res, persistent_res = [], []
        chunk_entity_coverage = {}

        for ent in entities:
            ew = entity_weights.get(ent["id"], 0)
            m_res = self._retrieve_memory(query_emb, ent, top_entities, top_chunks,
                                          relations, ew)
            memory_res.append(m_res)
            p_res = self._retrieve_persistent(query_emb, ent, top_entities, top_chunks,
                                              relations, ew)
            persistent_res.append(p_res)
            for cid in set(list(m_res.get("chunk_scores", {}).keys()) +
                           list(p_res.get("chunk_scores", {}).keys())):
                chunk_entity_coverage.setdefault(cid, set()).add(ent["id"])
        return memory_res, persistent_res, chunk_entity_coverage

    def _retrieve_memory(self, query_emb, ent, top_entities, top_chunks,
                         relations, entity_weight):
        """Subclasses override this method to provide L1 retrieval. Empty by default."""
        return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                "matched_entities": []}

    def _retrieve_persistent(self, query_emb, ent, top_entities, top_chunks,
                             relations, entity_weight):
        """Subclasses override this method to provide L2 retrieval. Empty by default."""
        return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                "matched_entities": []}

    def _aggregate_graph_scores(self, memory_res, persistent_res, top_chunks, query=""):
        graph_chunk_scores_aggregated = {}
        all_node_scores = {}
        all_node_chunks = {}
        for res in memory_res + persistent_res:
            for cid, score in res.get("chunk_scores", {}).items():
                graph_chunk_scores_aggregated[cid] = graph_chunk_scores_aggregated.get(cid, 0.0) + score
            for nid, score in res.get("node_scores", {}).items():
                all_node_scores[nid] = all_node_scores.get(nid, 0) + score
            for nid, chunks in res.get("node_chunks", {}).items():
                all_node_chunks.setdefault(nid, set()).update(chunks)
        top_nodes = sorted(all_node_scores.items(), key=lambda x: x[1], reverse=True)[:top_chunks]

        # Entity rerank: reorder top-N entities by semantic similarity between description text and query
        if top_nodes and self.reranker:
            entity_texts = []
            for nid, _ in top_nodes:
                # Lookup in L1
                d = self.memory_graph.graph.nodes.get(nid, {})
                name = d.get('name', '')
                etype = d.get('type', '')
                desc = d.get('desc', '')
                # Not in L1 → fetch from L2 (NebulaGraph)
                if not name and self.persistent_graph:
                    try:
                        vid = self.persistent_graph._format_vid(nid)
                        rows = self.persistent_graph.query(
                            f"FETCH PROP ON `entity` {vid} "
                            "YIELD properties(vertex).name AS name, "
                            "properties(vertex).type AS type;")
                        name = str(rows.get('name', [''])[0] or '') if rows else ''
                        etype = str(rows.get('type', [''])[0] or '') if rows else ''
                    except Exception:
                        pass
                entity_texts.append(f"{name} {etype}")
            if entity_texts:
                reranked = self.reranker.rerank(query, entity_texts, top_n=len(top_nodes))
                top_nodes = [(top_nodes[item["index"]][0], item["index"]) for item in reranked]

        # Chunks ordered by entity rerank: entities with higher rank have their associated chunks listed first
        entity_rank = {nid: rank for rank, (nid, _) in enumerate(top_nodes)}
        entity_chunk_set = set()
        for nid, _ in top_nodes:
            entity_chunk_set.update(all_node_chunks.get(nid, set()))
        graph_ranked = []
        for cid, score in graph_chunk_scores_aggregated.items():
            if cid not in entity_chunk_set:
                continue
            # Take the best rerank among entities associated with this chunk as the main sorting criterion
            best_entity_rank = min(
                (entity_rank.get(nid, 999) for nid in all_node_chunks if cid in all_node_chunks[nid]),
                default=999)
            graph_ranked.append((cid, -best_entity_rank, score))  # -rank makes smaller rank come first
        graph_ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [(cid, score) for cid, _, score in graph_ranked]


# ================================================================
#  HybridRetriever — Traditional Entity Extraction + Graph Traversal
# ================================================================

class HybridRetriever(BaseRetriever):
    """Traditional retrieval: extract entities via milvus or LLM → graph traversal → equal weighting.

    Configurable hyperparameters (from config/config.yaml → retrieval):
        gamma: hop-decay weight (gamma^l). Default 0.5 → [1.0, 0.5, 0.25, ...]
        B: beam width (top-B nodes per hop). Default 5.
        max_hops: maximum BFS hops. Default 10.
    """

    def __init__(self, vector_store=None, entity_index_name="entity_index_example",
                 memory_graph=None, llm=None, reranker=None, chunk_registry=None,
                 entity_extraction: str = "milvus", mode="hybrid",
                 gamma: float = 0.5, B: int = 5, max_hops: int = 10):
        super().__init__(vector_store=vector_store, memory_graph=memory_graph,
                         llm=llm, reranker=reranker, chunk_registry=chunk_registry,
                         mode=mode)
        self.entity_index_name = entity_index_name
        self.entity_extraction = entity_extraction
        self.gamma = gamma          # hop-decay weight
        self.B = B                  # beam width
        self.max_hops = max_hops    # max traversal hops
        self._entity_store = MilvusDB(
            db_name=entity_index_name, overwrite=False,
            embed_model=llm.embed_model if llm else None)

    # ── Entity Extraction ─────────────────────────────────────────

    def _extract_entities_heuristic(self, query, max_entities=6):
        """Heuristically extract entities from query text (uppercase words, quoted phrases)."""
        entities = []
        seen = set()
        # Quoted phrases first
        for token in query.split('"'):
            token = token.strip()
            if token and token not in seen and len(entities) < max_entities:
                seen.add(token)
                entities.append({"id": token, "type": "", "desc": ""})
        # Uppercase words
        words = [w.strip(".,:;!?()[]{}\"") for w in query.split()]
        for word in words:
            if len(entities) >= max_entities:
                break
            if not word:
                continue
            if word[0].isupper() or word.isupper():
                if word not in seen:
                    seen.add(word)
                    entities.append({"id": word, "type": "", "desc": ""})
        # Fallback: use all lowercase words
        if not entities:
            for word in words[:max_entities]:
                if word not in seen:
                    seen.add(word)
                    entities.append({"id": word, "type": "", "desc": ""})
        return entities

    def _extract_entities(self, query_emb, query, top_entities):
        if self.mode == "graph_only":
            entities = self._extract_entities_heuristic(query, top_entities)
            weights = {e["id"]: 1.0 for e in entities}
            return entities, [], weights, {}
        if self.entity_extraction == "milvus":
            entities = self._extract_entities_from_milvus(query_emb, top_entities)
            relations = []
        else:
            raw = self.llm.complete(prompt=prompt_extract_entities_str.format(context=query))
            parsed = self._parse_entities_from_llm(raw)
            entities = parsed.get("entities", [])
            relations = parsed.get("relations", [])
        weights = {e["id"]: 1.0 for e in entities}
        return entities, relations, weights, {}

    def _extract_entities_from_milvus(self, query_emb, topk=5):
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self._search_milvus(self._entity_store, query_emb, search_params, limit=topk)
        entities = []
        if results and len(results) > 0:
            for hit in results[0]:
                ent = hit.entity
                name = ent.get("name") or ""
                etype = ent.get("type") or ""
                edesc = ent.get("desc") or ""
                if name:
                    entities.append({"id": str(name), "type": str(etype), "desc": str(edesc)})
        return entities

    def _parse_entities_from_llm(self, raw_text: str) -> Dict[str, List]:
        if not raw_text:
            return {"entities": [], "relations": []}
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:]
        start, end = raw_text.find("{"), raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"entities": [], "relations": []}
        try:
            payload = json.loads(raw_text[start:end + 1])
        except json.JSONDecodeError:
            return {"entities": [], "relations": []}
        entities = payload.get("entities", []) or []
        relations_raw = (payload.get("relations") or payload.get("relationship")
                         or payload.get("relationships") or [])
        relations = []
        if isinstance(relations_raw, str):
            relations = [r.strip() for r in re.split(r"[;,]", relations_raw) if r.strip()]
        elif isinstance(relations_raw, list):
            for rel in relations_raw:
                if isinstance(rel, str) and rel.strip():
                    relations.append(rel.strip())
                elif isinstance(rel, dict):
                    v = rel.get("rel") or rel.get("relation") or rel.get("text")
                    if v: relations.append(str(v).strip())
        return {"entities": entities, "relations": relations}

    # ── Promotion ────────────────────────────────────────────

    def _handle_promotion(self, chunk_entity_coverage, entity_chunks, track_promotion):
        if not track_promotion:
            return
        for cid in chunk_entity_coverage:
            triggered, data = self.memory_graph.access_chunk(cid)
            if triggered:
                self.memory_graph.write_to_persistent_graph([data])

    # ── Graph Traversal ──────────────────────────────────────────

    def _search_milvus(self, milvus_db, vector, search_params, limit):
        return milvus_db.search(vector, search_params, limit,
                                       output_fields=["uid", "name", "type", "desc"])

    def _retrieve_memory(self, query_emb, ent, top_entities, top_chunks,
                         relations, entity_weight):
        return self.retrieve_from_memory_graph(
            query_emb, ent["id"], ent["type"], ent["desc"],
            top_entities, top_chunks, relations, entity_weight)

    def _retrieve_persistent(self, query_emb, ent, top_entities, top_chunks,
                             relations, entity_weight):
        return self.retrieve_from_persistent_graph(
            query_emb, ent["id"], ent["type"], ent["desc"],
            top_entities, top_chunks, relations, entity_weight)

    def retrieve_from_memory_graph(self, query_emb, entity_name, entity_type,
                                   entity_desc, top_entities=5, top_chunks=5,
                                   query_relations=None, entity_weight=1.0):
        if not self.memory_graph or (not entity_name and not entity_desc):
            return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                    "matched_entities": []}

        graph = self.memory_graph.graph
        vs = self._entity_store
        text = f"Entity: {entity_name}. Type: {entity_type}. Description: {entity_desc}"
        entity_emb = np.array(vs.embed_model.get_embedding(text), dtype=float)
        similarity_emb = query_emb if query_emb is not None else entity_emb
        sp = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self._search_milvus(vs, entity_emb, sp, limit=top_entities)

        matched, matched_ids = [], []
        if results and len(results) > 0:
            for hit in results[0]:
                uid = hit.entity.get("uid")
                name = hit.entity.get("name")
                matched.append({"uid": uid, "name": name, "score": float(hit.distance)})
                matched_ids.append(uid)
        matched_id_set = {str(uid) for uid in matched_ids if uid is not None}

        node_set = set()
        self.memory_graph._rebuild_id_index()
        for mid in matched_id_set:
            node_set.update(self.memory_graph.get_nodes_by_id(mid))

        # BFS decay propagation with configurable gamma and beam width B
        # hop 0=1.0, hop 1=gamma, hop 2=gamma^2, ... (clamped to 3 levels)
        gamma = self.gamma
        HOP_WEIGHTS = [1.0, gamma, gamma * gamma]
        HOP_TOPK = [0, self.B, self.B]  # hop 0 unlimited, hop 1/2 take top-B
        chunk_scores, node_scores, node_chunks = {}, {}, {}

        def add_score(chunks_data, weight):
            if isinstance(chunks_data, str):
                for c in [x.strip() for x in chunks_data.split(",") if x.strip()]:
                    chunk_scores[c] = chunk_scores.get(c, 0.0) + weight
            elif isinstance(chunks_data, (set, list, tuple)):
                for c in chunks_data:
                    c = str(c).strip()
                    if c: chunk_scores[c] = chunk_scores.get(c, 0.0) + weight

        def _topk_frontier(nodes, k):
            """Take top-k by semantic similarity between node name and query."""
            if k <= 0 or len(nodes) <= k:
                return nodes
            texts, valid = [], []
            for n in nodes:
                d = graph.nodes.get(n, {})
                name = d.get('name', '') or ''
                if name.strip():
                    texts.append(f"Entity: {name}. Type: {d.get('type', '') or ''}.")
                    valid.append(n)
            if not valid:
                return list(nodes)[:k]
            embs = vs.embed_model.get_embeddings(texts)
            scored = [(n, self._cosine(similarity_emb, np.array(e, dtype=float)))
                      for n, e in zip(valid, embs)]
            scored.sort(key=lambda x: x[1], reverse=True)
            top_set = {n for n, _ in scored[:k]}
            rest = [n for n in nodes if n not in top_set]
            return list(top_set) + rest[:max(0, k - len(top_set))]

        # hop 0: seed entity
        frontier = set(node_set)
        visited = set()
        for hop in range(self.max_hops):  # Configurable max hops to prevent infinite loops
            if not frontier:
                break
            hop_weight = entity_weight * HOP_WEIGHTS[min(hop, len(HOP_WEIGHTS) - 1)]
            next_frontier = set()
            for uid in frontier:
                if uid in visited:
                    continue
                visited.add(uid)
                d = graph.nodes.get(uid, {})
                # Node chunk weighting
                add_score(d.get("source_chunks"), hop_weight)
                node_scores[uid] = node_scores.get(uid, 0) + hop_weight
                node_chunks.setdefault(uid, set()).update(d.get("source_chunks") or [])
                # Traverse out-edges, collect next-hop nodes + current edge's chunk
                for _, neighbor, data in graph.out_edges(uid, data=True):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                    chunk = data.get("source_chunk")
                    if chunk:
                        add_score([chunk], hop_weight)
                for neighbor, _, data in graph.in_edges(uid, data=True):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                    chunk = data.get("source_chunk")
                    if chunk:
                        add_score([chunk], hop_weight)
            # Next hop take top-k (hop 1 take 5, hop 2 take 5, then unlimited)
            topk = HOP_TOPK[min(hop + 1, len(HOP_TOPK) - 1)]
            if topk > 0:
                frontier = set(_topk_frontier(list(next_frontier), topk))
            else:
                frontier = next_frontier

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_chunks]
        return {"chunks": [c for c, _ in sorted_chunks], "chunk_scores": dict(sorted_chunks),
                "node_scores": node_scores, "node_chunks": {k: list(v) for k, v in node_chunks.items()},
                "matched_entities": matched}

    def retrieve_from_persistent_graph(self, query_emb, entity_name, entity_type,
                                       entity_desc, top_entities=5, top_chunks=5,
                                       query_relations=None, entity_weight=1.0):
        if not self.persistent_graph or (not entity_name and not entity_desc):
            return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                    "matched_entities": []}

        g = self.persistent_graph
        vs = self._entity_store
        text = f"Entity: {entity_name}. Type: {entity_type}. Description: {entity_desc}"
        entity_emb = np.array(vs.embed_model.get_embedding(text), dtype=float)
        similarity_emb = query_emb if query_emb is not None else entity_emb
        sp = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self._search_milvus(vs, entity_emb, sp, limit=top_entities)

        matched, matched_ids = [], []
        if results and len(results) > 0:
            for hit in results[0]:
                matched.append({"uid": hit.entity.get("uid"),
                                "name": hit.entity.get("name"),
                                "score": float(hit.distance)})
                matched_ids.append(hit.entity.get("uid"))
        if not matched_ids:
            return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                    "matched_entities": []}

        vid_list = [g._format_vid(uid) for uid in matched_ids if uid is not None]
        if not vid_list:
            return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                    "matched_entities": []}

        vids_str = ", ".join(vid_list)
        try:
            node_rows = g.query(
                f"FETCH PROP ON `entity` {vids_str} "
                "YIELD id(vertex) AS vid, properties(vertex).name AS name, "
                "properties(vertex).source_chunk AS source_chunk;")
        except Exception:
            return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                    "matched_entities": []}

        existing_vids = set(str(v) for v in node_rows.get("vid", []) if v is not None)
        if not existing_vids:
            return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                    "matched_entities": []}
        matched = [m for m in matched if str(m.get("uid")) in existing_vids]
        if not matched:
            return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                    "matched_entities": []}

        def _fetch_node_props(vids):
            if not vids: return {}
            vs_str = ", ".join([g._format_vid(v) for v in vids])
            try:
                rows = g.query(
                    f"FETCH PROP ON `entity` {vs_str} "
                    "YIELD id(vertex) AS vid, properties(vertex).name AS name, "
                    "properties(vertex).type AS type, "
                    "properties(vertex).source_chunk AS source_chunk;")
            except Exception:
                return {}
            props = {}
            for idx, vid in enumerate(rows.get("vid", [])):
                if vid is not None:
                    props[str(vid)] = {
                        "name": rows["name"][idx] if idx < len(rows.get("name", [])) else "",
                        "type": rows["type"][idx] if idx < len(rows.get("type", [])) else "",
                        "source_chunk": rows["source_chunk"][idx] if idx < len(rows.get("source_chunk", [])) else "",
                    }
            return props

        def _run_go(vids):
            if not vids: return {}
            vs_str = ", ".join([g._format_vid(v) for v in vids])
            try:
                return g.query(
                    f"GO 1 STEPS FROM {vs_str} OVER `relationship` BIDIRECT "
                    "YIELD src(edge) AS src, dst(edge) AS dst, "
                    "properties(edge).relationship AS relation, "
                    "properties(edge).source_chunk AS source_chunk;")
            except Exception:
                return {}

        def _node_text(row):
            return f"Entity: {row.get('name', '')}. Type: {row.get('type', '')}."

        def _topk_sim(vids, k=5):
            props = _fetch_node_props(vids)
            if not props: return []
            texts = [(_node_text(r), vid) for vid, r in props.items()]
            embs = vs.embed_model.get_embeddings([t for t, _ in texts])
            scored = [(vid, self._cosine(similarity_emb, np.array(e, dtype=float)))
                      for (_, vid), e in zip(texts, embs)]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [vid for vid, _ in scored[:k]]

        base_vids = set(str(v) for v in existing_vids)
        edge_rows = _run_go(base_vids)
        _collect = lambda rows: {str(v) for k in ("src", "dst") for v in rows.get(k, []) if v is not None}
        first_hop = _collect(edge_rows) - base_vids
        top_first = _topk_sim(first_hop, k=5)
        edge_rows_2 = _run_go(set(top_first))
        second_hop = _collect(edge_rows_2) - base_vids - set(top_first)
        top_second = _topk_sim(second_hop, k=5)

        first_props = _fetch_node_props(set(top_first))
        second_props = _fetch_node_props(set(top_second))

        chunk_scores, node_scores, node_chunks = {}, {}, {}
        def add_score(chunks_data, weight):
            if isinstance(chunks_data, str):
                for c in [x.strip() for x in chunks_data.split(",") if x.strip()]:
                    chunk_scores[c] = chunk_scores.get(c, 0.0) + weight
            elif isinstance(chunks_data, list):
                for c in chunks_data:
                    c = str(c).strip()
                    if c: chunk_scores[c] = chunk_scores.get(c, 0.0) + weight

        node_vids = node_rows.get("vid", [])
        node_chunk_list = node_rows.get("source_chunk", [])

        for idx, chunks in enumerate(node_chunk_list):
            add_score(chunks, entity_weight)
            if idx < len(node_vids) and node_vids[idx] is not None:
                vid = str(node_vids[idx])
                node_scores[vid] = node_scores.get(vid, 0) + entity_weight
                node_chunks.setdefault(vid, set()).update(
                    [c.strip() for c in str(chunks).split(",") if c.strip()] if chunks else [])
        for vid, row in first_props.items():
            add_score(row.get("source_chunk", ""), 0.5 * entity_weight)
            node_scores[vid] = node_scores.get(vid, 0) + 0.5 * entity_weight
            node_chunks.setdefault(vid, set()).update(
                [c.strip() for c in str(row.get("source_chunk", "")).split(",") if c.strip()]
                if row.get("source_chunk") else [])
        for vid, row in second_props.items():
            add_score(row.get("source_chunk", ""), 0.3 * entity_weight)
            node_scores[vid] = node_scores.get(vid, 0) + 0.3 * entity_weight
            node_chunks.setdefault(vid, set()).update(
                [c.strip() for c in str(row.get("source_chunk", "")).split(",") if c.strip()]
                if row.get("source_chunk") else [])
        for chunk in edge_rows.get("source_chunk", []):
            if chunk: add_score([chunk], 0.5 * entity_weight)
        for chunk in edge_rows_2.get("source_chunk", []):
            if chunk: add_score([chunk], 0.3 * entity_weight)

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return {"chunks": [c for c, _ in sorted_chunks], "chunk_scores": chunk_scores,
                "node_scores": node_scores, "node_chunks": {k: list(v) for k, v in node_chunks.items()},
                "matched_entities": matched}


# ================================================================
#  FactRetriever — Fact Embedding Retrieval + Convergence Promotion
# ================================================================

class FactRetriever(BaseRetriever):
    """Fact retrieval: Query → fact embedding → entities+weights → graph traversal → convergence promotion."""

    def __init__(self, vector_store=None, entity_index_name="entity_index_example",
                 memory_graph=None, llm=None, reranker=None, chunk_registry=None,
                 fact_store=None, fact_config: dict = None):
        super().__init__(vector_store=vector_store, memory_graph=memory_graph,
                         llm=llm, reranker=reranker, chunk_registry=chunk_registry)
        self.fact_store = fact_store
        self.fact_config = fact_config or {}
        # Reuse HybridRetriever's graph traversal, create internal instance
        self._graph = HybridRetriever(
            vector_store=vector_store, entity_index_name=entity_index_name,
            memory_graph=memory_graph, llm=llm, chunk_registry=chunk_registry,
            entity_extraction="milvus")

    # ── Entity Extraction ─────────────────────────────────────────

    def _extract_entities(self, query_emb, query, top_entities):
        if not self.fact_store or not self.llm:
            return [], [], {}, {}
        topk = self.fact_config.get("topk_facts", 100)
        max_entities = self.fact_config.get("topk_entities", 10)
        facts = self.fact_store.search_facts(query_emb.tolist(), topk=topk)
        if not facts:
            return [], [], {}, {}

        entity_scores, entity_names, entity_chunks = {}, {}, {}
        for f in facts:
            score = float(f.get("score", 0))
            chunk_id = f.get("chunk_id", "")
            for uk, nk in [("subj_uid", "subj_name"), ("obj_uid", "obj_name")]:
                uid = str(f.get(uk, ""))
                name = f.get(nk, "")
                if not uid or uid == "0": continue
                entity_scores[uid] = entity_scores.get(uid, 0.0) + score
                entity_names[uid] = name or entity_names.get(uid, "")
                entity_chunks.setdefault(uid, set()).add(chunk_id)

        entity_weights = {}
        for uid, total_score in entity_scores.items():
            if not self.memory_graph or not self.memory_graph.graph.has_node(uid):
                continue  # Skip entities not in the graph
            ref_count = max(1, int(self.memory_graph.graph.nodes[uid].get("ref_count", 1)))
            entity_weights[uid] = total_score / ref_count

        if not entity_weights:
            return [], [], {}, {}

        sorted_ents = sorted(entity_weights.items(), key=lambda x: x[1], reverse=True)[:max_entities]
        entities = [{"id": uid, "type": "", "desc": entity_names.get(uid, "")}
                    for uid, _ in sorted_ents]
        entity_weights = {uid: w for uid, w in sorted_ents}
        total = sum(entity_weights.values()) or 1.0
        entity_weights = {k: v / total for k, v in entity_weights.items()}

        return entities, [], entity_weights, entity_chunks

    # ── Graph Retrieval (directly locate nodes by UID, no re-embedding search) ──

    def _retrieve_memory(self, query_emb, ent, top_entities, top_chunks,
                         relations, entity_weight):
        return self._retrieve_by_uid(query_emb, ent, top_chunks, entity_weight,
                                     use_persistent=False)

    def _retrieve_persistent(self, query_emb, ent, top_entities, top_chunks,
                             relations, entity_weight):
        uid = str(ent["id"])
        if self.persistent_graph:
            return self._retrieve_from_l2_by_uid(uid, top_chunks, entity_weight)
        return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                "matched_entities": []}

    def _retrieve_by_uid(self, query_emb, ent, top_chunks, entity_weight, use_persistent):
        """Given entity UID, directly start BFS traversal from graph node, skip entity_index search."""
        uid = str(ent["id"])

        if use_persistent and self.persistent_graph:
            return self._retrieve_from_l2_by_uid(uid, top_chunks, entity_weight)

        if not self.memory_graph or not self.memory_graph.graph.has_node(uid):
            return {"chunks": [], "chunk_scores": {}, "node_scores": {}, "node_chunks": {},
                    "matched_entities": []}

        return self._bfs_from_uid(uid, entity_weight, top_chunks)

    def _bfs_from_uid(self, uid, entity_weight, top_chunks):
        graph = self.memory_graph.graph
        chunk_scores, node_scores, node_chunks = {}, {}, {}

        def add_score(chunks_data, weight):
            if isinstance(chunks_data, str):
                for c in [x.strip() for x in chunks_data.split(",") if x.strip()]:
                    chunk_scores[c] = chunk_scores.get(c, 0.0) + weight
            elif isinstance(chunks_data, (set, list, tuple)):
                for c in chunks_data:
                    c = str(c).strip()
                    if c: chunk_scores[c] = chunk_scores.get(c, 0.0) + weight

        # BFS starting from UID
        DECAY = 0.5
        visited = set()
        frontier = {uid}
        hop_weight = entity_weight
        for _ in range(10):
            if not frontier: break
            next_frontier = set()
            for n in frontier:
                if n in visited: continue
                visited.add(n)
                d = graph.nodes.get(n, {})
                add_score(d.get("source_chunks"), hop_weight)
                node_scores[n] = node_scores.get(n, 0) + hop_weight
                node_chunks.setdefault(n, set()).update(d.get("source_chunks") or [])
                for _, nb, data in graph.out_edges(n, data=True):
                    if nb not in visited:
                        next_frontier.add(nb)
                    c = data.get("source_chunk")
                    if c: add_score([c], hop_weight * DECAY)
                for nb, _, data in graph.in_edges(n, data=True):
                    if nb not in visited:
                        next_frontier.add(nb)
                    c = data.get("source_chunk")
                    if c: add_score([c], hop_weight * DECAY)
            frontier = next_frontier
            hop_weight *= DECAY

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_chunks]
        return {"chunks": [c for c, _ in sorted_chunks], "chunk_scores": dict(sorted_chunks),
                "node_scores": node_scores, "node_chunks": {k: list(v) for k, v in node_chunks.items()},
                "matched_entities": [{"uid": uid, "name": "", "score": 1.0}]}

    def _retrieve_from_l2_by_uid(self, uid, top_chunks, entity_weight):
        """Retrieve chunk of entity and its neighbors from L2 (NebulaGraph) by UID."""
        g = self.persistent_graph
        vid = g._format_vid(uid)
        chunk_scores = {}

        def add_score(chunks_data, weight):
            if isinstance(chunks_data, str):
                for c in [x.strip() for x in chunks_data.split(",") if x.strip()]:
                    chunk_scores[c] = chunk_scores.get(c, 0.0) + weight
            elif isinstance(chunks_data, list):
                for c in chunks_data:
                    c = str(c).strip()
                    if c: chunk_scores[c] = chunk_scores.get(c, 0.0) + weight

        try:
            rows = g.query(
                f"FETCH PROP ON `entity` {vid} "
                "YIELD properties(vertex).source_chunk AS source_chunk;")
            if rows:
                for sc in rows.get("source_chunk", []):
                    add_score(sc, entity_weight)
        except Exception:
            pass

        try:
            rows = g.query(
                f"GO 1 STEPS FROM {vid} OVER `relationship` BIDIRECT "
                "YIELD properties(edge).source_chunk AS source_chunk;")
            if rows:
                for sc in rows.get("source_chunk", []):
                    add_score(sc, 0.5 * entity_weight)
        except Exception:
            pass

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_chunks]
        return {"chunks": [c for c, _ in sorted_chunks], "chunk_scores": dict(sorted_chunks),
                "node_scores": {}, "node_chunks": {},
                "matched_entities": [{"uid": uid, "name": "", "score": 1.0}]}

    # ── Quality Promotion (called by PurifyGraphRAG after answer generation) ──

    def _handle_promotion(self, chunk_entity_coverage, entity_chunks, track_promotion):
        pass  # Promotion deferred to after answer generation, triggered by promote_after_qa

    def promote_after_qa(self, chunk_entity_coverage, quality_ok):
        """Called after answer generation, skips promotion when quality_ok=False."""
        if not quality_ok:
            return
        threshold = self.fact_config.get("entity_convergence_threshold", 3)
        self.memory_graph.promote_by_convergence(chunk_entity_coverage, threshold=threshold)
