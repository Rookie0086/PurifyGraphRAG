#!/usr/bin/env python3
"""QA locality (retrieval pattern) analysis for PurifyGraphRAG.

Adapts the retrieval-pattern methodology of DepCache (SIGMOD'25, Fig. 4) to
PurifyGraphRAG's own datasets and retrieval pipeline: it measures how unevenly
QA requests reference graph entities / chunks, i.e. whether the top-p% most
frequent entities (chunks) absorb most of the retrieval requests. This serves
as motivation evidence for the L1/L2 two-tier graph cache and the
access-frequency-driven promotion mechanism (tau_hit).

Methodology (following DepCache Fig. 4):
  1. For each question, obtain the referenced entities (or final chunks).
  2. Count, per entity/chunk, the number of questions that reference it.
  3. Sort by frequency (descending) and plot the cumulative distribution:
     x = ratio of entities (%), y = share of requests covered (%).
  4. Compare against the uniform diagonal; report coverage at top-10/20/50%.

Two granularities:
  entity - entity-level locality (directly comparable to DepCache Fig. 4);
           supports the frequency-driven hot/cold promotion (Knowledge
           Purification) design.
  chunk  - chunk-level locality over the FINAL retrieved chunks, computed
           offline from existing QA result files (output/qa/qa_results_*.json
           which record the "chunk" list per question); directly supports the
           L1 chunk cache design (C_max, LRU+TTL eviction). No DB/LLM needed.

Entity sources (--entity-mode):
  llm       - same prompt + parser as the system's retrieval path
              (config: retrieval.entity_extraction=llm); needs the LLM API.
  embedding - top-k matches from the Milvus entity index, identical to
              (config: retrieval.entity_extraction=milvus); needs a running
              Milvus + built entity_index_<dataset> collection.
  heuristic - zero-dependency preview (quoted phrases / capitalized tokens).

Usage:
  # entity-level locality (LLM mode) on a question range
  python scripts/qa_locality_analysis.py --dataset hotpotqa --start 0 --end 120

  # multiple datasets -> 1xN panel figure in DepCache Fig.4 style
  python scripts/qa_locality_analysis.py --dataset rgb hotpotqa wikimultihopqa

  # embedding mode (needs Milvus + entity_index_<dataset>)
  python scripts/qa_locality_analysis.py --dataset hotpotqa --entity-mode embedding

  # chunk-level locality from existing QA results (no DB / LLM needed)
  python scripts/qa_locality_analysis.py --qa-results output/qa/qa_results_hotpotqa_0_120.json --label HotpotQA

Outputs (default under output/locality/):
  <name>_<granularity>_frequency.json   frequency data + coverage summary
  <name>_<granularity>_locality.pdf/png single-dataset CDF figure
  locality_panel.pdf/png                 multi-dataset panel figure
"""

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from tqdm import tqdm

from src.utils.base import read_json, save_to_json

# Dataset display names -> figure subplot labels
DATASET_LABELS = {
    "rgb": "RGB",
    "hotpotqa": "HotpotQA",
    "wikimultihopqa": "2WikiMultihopQA",
    "2wikimultihopqa": "2WikiMultihopQA",
    "specificqa": "SpecificQA",
}

COVERAGE_POINTS = (10, 20, 30, 50)  # top-p% coverage reported in the summary


# ── Config ────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load the project config (config/config.yaml), honoring CACHEGRAPH_CONFIG."""
    from src.utils import get_config

    try:
        cfg = get_config()
        if isinstance(cfg, dict) and cfg.get("model"):
            return cfg
    except Exception:
        pass
    import yaml

    local = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    with open(local, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Question loading ──────────────────────────────────────────────

def load_questions(dataset: str, start: int, end: int) -> list:
    """Load questions of a dataset using the same loaders as the main entry."""
    if dataset.startswith("rgb"):
        from data.rgb import get_rgb_info

        sub = dataset[4:] or "en_refine"  # e.g. "rgb_en_refine" -> "en_refine"
        info = get_rgb_info(file=sub)
    elif dataset == "hotpotqa":
        from data.hotpotqa import get_hotpotqa_info

        info = get_hotpotqa_info(file="hotpot_dev_distractor_v1", num=10 ** 9)
    elif dataset in ("wikimultihopqa", "2wikimultihopqa"):
        from data.wikimultihopqa import get_2wikimultihopqa_info

        info = get_2wikimultihopqa_info()
    elif dataset == "specificqa":
        from data.specificqa import get_specificqa_info

        info = get_specificqa_info(limit=10 ** 9, update=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    questions = [q for q in info["questions"] if q]
    questions = questions[start:end] if end else questions[start:]
    print(f"[{dataset}] loaded {len(questions)} questions (range {start}:{end or 'end'})")
    return questions


# ── Entity sources ────────────────────────────────────────────────

def _normalize(name: str) -> str:
    return " ".join(str(name).split()).lower().strip()


def extract_entities_llm(questions: list, cfg: dict, concurrency: int,
                         cache_file: str) -> list:
    """Extract per-question entities with the SAME prompt + parser as the
    system's retrieval path (HybridRetriever, entity_extraction=llm)."""
    from src.llm.env import LLMEnv
    from src.retriever import HybridRetriever
    from src.utils.prompts import prompt_extract_entities_str

    mc, ec = cfg.get("model", {}), cfg.get("embedding", {})
    llm = LLMEnv(
        backend=mc.get("backend", "openai"),
        model=mc.get("model_name", "gpt-4o-mini"),
        api_key=mc.get("api_key"),
        base_url=mc.get("base_url"),
        embed_model_name=ec.get("model_name", "BAAI/bge-m3"),
        embed_backend=ec.get("backend", "local"),
        embed_api_key=ec.get("api_key") if ec.get("backend") == "api" else None,
        embed_base_url=ec.get("base_url") if ec.get("backend") == "api" else None,
    )

    # Per-question cache for resume support
    done = {}
    if cache_file and os.path.exists(cache_file):
        try:
            for item in read_json(cache_file):
                done[item["question"]] = item["entities"]
            print(f"  resume: {len(done)} questions already extracted "
                  f"({len(questions) - sum(1 for q in questions if q in done)} remaining)")
        except Exception as e:
            print(f"  cache load failed, restarting: {e}")

    todo = [q for q in questions if q not in done]
    if todo:
        def _one(question):
            try:
                raw = llm.complete(prompt=prompt_extract_entities_str.format(context=question))
                parsed = HybridRetriever._parse_entities_from_llm(None, raw)
            except Exception as e:
                print(f"  extraction failed ({type(e).__name__}: {e})")
                return question, []
            ents = []
            for e in parsed.get("entities", []):
                name = e.get("id") if isinstance(e, dict) else e
                if name and str(name).strip():
                    ents.append(str(name).strip())
            return question, ents

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {pool.submit(_one, q): q for q in todo}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="LLM entity extraction"):
                q, ents = fut.result()
                done[q] = ents
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            save_to_json(cache_file,
                         [{"question": q, "entities": ents} for q, ents in done.items()],
                         indent=2, info=False)

    return [done.get(q, []) for q in questions]


def extract_entities_embedding(questions: list, cfg: dict, top_k: int) -> list:
    """Match per-question entities from the Milvus entity index — identical to
    the system's retrieval path (HybridRetriever, entity_extraction=milvus)."""
    from database.milvus import MilvusDB
    from src.llm.env import LLMEnv

    mc, ec = cfg.get("model", {}), cfg.get("embedding", {})
    llm = LLMEnv(
        backend=mc.get("backend", "openai"),
        model=mc.get("model_name", "gpt-4o-mini"),
        api_key=mc.get("api_key"),
        base_url=mc.get("base_url"),
        embed_model_name=ec.get("model_name", "BAAI/bge-m3"),
        embed_backend=ec.get("backend", "local"),
        embed_api_key=ec.get("api_key") if ec.get("backend") == "api" else None,
        embed_base_url=ec.get("base_url") if ec.get("backend") == "api" else None,
    )

    dataset = os.environ.get("CG_LOCALITY_DATASET", "")
    index_name = os.environ.get("CG_LOCALITY_ENTITY_INDEX", f"entity_index_{dataset}")
    print(f"  entity index: {index_name}")
    store = MilvusDB(db_name=index_name, overwrite=False, embed_model=llm.embed_model)
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

    out = []
    for q in tqdm(questions, desc="Embedding entity matching"):
        emb = llm.embed_model.get_embedding(q)
        try:
            results = store.search(emb, search_params, top_k,
                                   output_fields=["uid", "name", "type", "desc"])
            ents = []
            if results and len(results) > 0:
                for hit in results[0]:
                    name = hit.entity.get("name")
                    if name:
                        ents.append(str(name))
        except Exception as e:
            print(f"  search failed ({type(e).__name__}: {e})")
            ents = []
        out.append(ents)
    return out


def extract_entities_heuristic(questions: list, max_entities: int = 6) -> list:
    """Zero-dependency preview (mirrors HybridRetriever._extract_entities_heuristic)."""
    out = []
    for query in questions:
        entities, seen = [], set()

        def _add(token):
            token = token.strip(".,:;!?()[]{}\"' ")
            if token and token.lower() not in seen and len(entities) < max_entities:
                seen.add(token.lower())
                entities.append(token)

        for token in query.split('"'):
            _add(token)
        for word in query.split():
            if len(entities) >= max_entities:
                break
            if word[:1].isupper():
                _add(word)
        if not entities:
            for word in query.split()[:max_entities]:
                _add(word)
        out.append(entities)
    return out


# ── Chunk source (offline, from existing QA results) ─────────────

def load_chunks_from_qa_results(qa_result_files: list) -> dict:
    """Read final retrieved chunk ids per question from QA result files
    (items recorded by PurifyGraphRAG.query() carry a "chunk" field)."""
    chunks_per_question = []
    for path in qa_result_files:
        items = read_json(path)
        n = 0
        for item in items:
            chunk_ids = item.get("chunk") or []
            if chunk_ids:
                chunks_per_question.append([str(c) for c in chunk_ids])
                n += 1
        print(f"  {path}: {n} questions with retrieved chunks")
    return chunks_per_question


# ── Frequency / CDF statistics ────────────────────────────────────

def entity_frequency(entity_lists: list) -> Counter:
    """Count per entity the number of referencing questions (DepCache style:
    each question contributes at most once per entity)."""
    counter = Counter()
    for ents in entity_lists:
        uniq = {_normalize(e) for e in ents if _normalize(e)}
        counter.update(uniq)
    return counter


def coverage_at(counter: Counter, pct: float) -> float:
    """Share of requests (%) covered by the top-p% most frequent items."""
    counts = sorted(counter.values(), reverse=True)
    total = sum(counts)
    if not total:
        return 0.0
    k = max(1, int(round(len(counts) * pct / 100.0)))
    return sum(counts[:k]) / total * 100.0


def cdf_arrays(counter: Counter):
    counts = np.array(sorted(counter.values(), reverse=True), dtype=float)
    cdf = np.cumsum(counts) / counts.sum() * 100.0
    n = len(cdf)
    x = np.concatenate([[0.0], np.arange(1, n + 1) / n * 100.0])
    y = np.concatenate([[0.0], cdf])
    return x, y


def summarize(name: str, counter: Counter, granularity: str, mode: str,
              n_questions: int) -> dict:
    counts = sorted(counter.values(), reverse=True)
    x, y = cdf_arrays(counter)
    top_entities = [{"entity": e, "questions": c} for e, c in counter.most_common(20)]
    return {
        "name": name,
        "granularity": granularity,
        "mode": mode,
        "n_questions": n_questions,
        "n_unique": len(counter),
        "n_references": sum(counts),
        "coverage": {f"top{p}%": round(coverage_at(counter, p), 1) for p in COVERAGE_POINTS},
        "top_items": top_entities,
        "frequency": counts,
        "cdf": {"x": [round(v, 3) for v in x.tolist()],
                "y": [round(v, 3) for v in y.tolist()]},
    }


# ── Plotting (DepCache Fig.4 style) ───────────────────────────────

def _setup_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "axes.labelsize": "10",
        "xtick.labelsize": "9",
        "ytick.labelsize": "9",
        "lines.linewidth": 1.5,
        "legend.fontsize": "9",
        "legend.frameon": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    })
    plt.rcParams["pdf.fonttype"] = 42  # editable text in the paper PDF
    return plt


def _plot_cdf_on_ax(ax, plt, counter: Counter):
    x, y = cdf_arrays(counter)
    ax.plot(x, y, color="blue", zorder=1, label="Graph Retrieve")
    ax.plot([0, 100], [0, 100], color="orange", alpha=0.7, zorder=0,
            label="Uniform Distribution")

    # Mark the (20%, cov%) point as in DepCache Fig. 4
    cov20 = coverage_at(counter, 20)
    ax.scatter([20], [cov20], c="red", s=15, zorder=2)
    ax.text(min(98, 20 + 38), max(2, cov20 - 20), f"(20%,{cov20:.0f}%)",
            fontsize=9, color="red", ha="right")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.linspace(0, 100, 5))
    ax.set_yticks(np.linspace(0, 100, 5))
    ax.grid(True, alpha=0.5)


def plot_single(summary: dict, out_dir: str):
    plt = _setup_style()
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    _plot_cdf_on_ax(ax, plt, Counter({i: c for i, c in enumerate(summary["frequency"])}))
    ax.set_xlabel("Ratio of entities (%)" if summary["granularity"] == "entity"
                  else "Ratio of chunks (%)")
    ax.set_ylabel("CDF (%)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{summary['name']}_{summary['granularity']}_locality.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(fig)


def plot_panel(summaries: list, out_dir: str, figname="locality_panel"):
    plt = _setup_style()
    n = len(summaries)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.2))
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.32, bottom=0.34)
    for idx, (s, ax) in enumerate(zip(summaries, axes)):
        _plot_cdf_on_ax(ax, plt, Counter({i: c for i, c in enumerate(s["frequency"])}))
        ax.set_xlabel("Ratio of entities (%)" if s["granularity"] == "entity"
                      else "Ratio of chunks (%)")
        if idx == 0:
            ax.set_ylabel("CDF (%)")
        ax.text(0.5, -0.42, f"({chr(97 + idx)}) {s['name']}",
                transform=ax.transAxes, ha="center", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.12), columnspacing=3)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{figname}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="QA locality (retrieval pattern) analysis for PurifyGraphRAG "
                    "(methodology adapted from DepCache Fig. 4)")
    parser.add_argument("--dataset", nargs="+", default=[],
                        help="datasets: rgb | hotpotqa | wikimultihopqa | specificqa "
                             "(rgb accepts rgb_en_refine style subtype)")
    parser.add_argument("--start", type=int, default=0, help="question range start")
    parser.add_argument("--end", type=int, default=0, help="question range end (0 = all)")
    parser.add_argument("--entity-mode", choices=["llm", "embedding", "heuristic"],
                        default="llm", help="entity source for entity-level analysis")
    parser.add_argument("--top-entities", type=int, default=5,
                        help="top-k entity index matches (embedding mode)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="parallel LLM extraction workers (llm mode)")
    parser.add_argument("--qa-results", nargs="+", default=[],
                        help="QA result JSON files for chunk-level locality "
                             "(records the final retrieved chunks per question)")
    parser.add_argument("--label", nargs="+", default=[],
                        help="display labels for --qa-results files")
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "output", "locality"))
    args = parser.parse_args()

    if not args.dataset and not args.qa_results:
        parser.error("provide --dataset and/or --qa-results")

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = load_config()
    summaries = []

    # ── Entity-level locality ──
    if args.dataset:
        mode = args.entity_mode
        for dataset in args.dataset:
            print(f"\n=== Entity-level locality: {dataset} (mode={mode}) ===")
            questions = load_questions(dataset, args.start, args.end)
            if not questions:
                print(f"  no questions loaded for {dataset}, skipped")
                continue

            if mode == "llm":
                cache_file = os.path.join(
                    args.out_dir, f"entity_cache_{dataset}_{args.start}_{args.end or 'all'}.json")
                entity_lists = extract_entities_llm(questions, cfg, args.concurrency, cache_file)
            elif mode == "embedding":
                os.environ["CG_LOCALITY_DATASET"] = dataset
                entity_lists = extract_entities_embedding(questions, cfg, args.top_entities)
            else:
                entity_lists = extract_entities_heuristic(questions)

            counter = entity_frequency(entity_lists)
            if not counter:
                print(f"  no entities extracted for {dataset}, skipped")
                continue
            label = DATASET_LABELS.get(dataset.split("_")[0], dataset)
            summary = summarize(label, counter, "entity", mode, len(questions))

            save_to_json(os.path.join(args.out_dir,
                                      f"{label}_{mode}_entity_frequency.json"),
                         summary, indent=2, info=False)
            print(f"  questions={summary['n_questions']}, unique entities={summary['n_unique']}")
            print(f"  coverage: {summary['coverage']}")
            plot_single(summary, args.out_dir)
            summaries.append(summary)

    # ── Chunk-level locality (offline from QA results) ──
    if args.qa_results:
        print(f"\n=== Chunk-level locality from {len(args.qa_results)} QA result file(s) ===")
        chunk_lists = load_chunks_from_qa_results(args.qa_results)
        counter = Counter()
        for chunks in chunk_lists:
            counter.update(set(chunks))
        if counter:
            labels = args.label or [Path(p).stem for p in args.qa_results]
            label = labels[0]
            summary = summarize(label, counter, "chunk", "qa_results", len(chunk_lists))
            save_to_json(os.path.join(args.out_dir, f"{label}_chunk_frequency.json"),
                         summary, indent=2, info=False)
            print(f"  questions={summary['n_questions']}, unique chunks={summary['n_unique']}")
            print(f"  coverage: {summary['coverage']}")
            plot_single(summary, args.out_dir)
            summaries.append(summary)
        else:
            print("  no chunks found in the given QA result files")

    # ── Panel figure across datasets ──
    if len(summaries) > 1:
        print(f"\n=== Panel figure ({len(summaries)} subplots) ===")
        plot_panel(summaries, args.out_dir)

    print("\nDone. Summary:")
    for s in summaries:
        print(f"  [{s['granularity']}] {s['name']}: "
              f"top20% covers {s['coverage']['top20%']}% of "
              f"{s['n_references']} requests over {s['n_unique']} unique items")


if __name__ == "__main__":
    main()
