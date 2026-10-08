"""Structured pipeline logger, tracking document processing, entity extraction, triplet extraction, embedding progress and timing."""

import datetime as dt
import json
import os
import sys
import time
from collections import defaultdict
from typing import Optional

from src.utils.base import create_dir


class PipelineLogger:
    """Logs key metrics for each stage of the PurifyGraphRAG pipeline.

    Features:
    - Per-stage timing (ingestion / extraction / embedding / retrieval)
    - Per-document entity and triplet extraction tracking
    - Auto-flush, suitable for long-running processes
    - Summary report at the end
    """

    def __init__(self, log_dir: str = "logs", dataset: str = ""):
        create_dir(log_dir)
        timestamp = dt.datetime.now().strftime("%m%d_%H%M%S")
        suffix = f"_{dataset}" if dataset else ""
        self._log_file = os.path.join(log_dir, f"pipeline{suffix}_{timestamp}.log")
        self._json_file = os.path.join(log_dir, f"pipeline{suffix}_{timestamp}.json")

        # Write log file using print-style formatting
        self._handle = open(self._log_file, "w", encoding="utf-8")
        self._start_time = time.time()
        self._buffer = []
        self._pbar = None
        self._qa_mode = False

        # Structured metrics
        self.metrics = defaultdict(list)
        self._doc_count = 0
        self._chunk_count = 0
        self._entity_count = 0
        self._triplet_count = 0
        self._embedding_count = 0

        self._prompt_tokens = 0
        self._completion_tokens = 0

        self._print(f"Pipeline started | dataset={dataset} | log={self._log_file}")

    def log_config(self, config: dict):
        """Write runtime config to the log."""
        self._print("=" * 50)
        self._print("  Runtime Config")
        self._print("=" * 50)
        for key, value in config.items():
            self._print(f"  {key}: {value}")
        self._print("=" * 50)

    def _print(self, *args, oneline=False):
        """Output to both terminal and log file."""
        head = dt.datetime.now().strftime("%H:%M:%S")
        line = head + " " + " ".join(map(str, args))
        if self._buffer is not None:
            self._buffer.append((line, oneline))
        else:
            end = "\r" if oneline else "\n"
            print(line, end=end, flush=True)
            self._handle.write(line + ("\n" if not oneline else ""))
            self._handle.flush()

    def buffer_on(self):
        self._buffer = []

    def buffer_off(self):
        if self._buffer is not None:
            for line, oneline in self._buffer:
                self._handle.write(line + ("\n" if not oneline else ""))
                if not self._qa_mode:
                    end = "\r" if oneline else "\n"
                    print(line, end=end, flush=True)
            self._handle.flush()
            self._buffer = None

    def warn(self, *args):
        """Log a warning (with [WARN] tag for easy log search)."""
        self._print("[WARN]", *args)
        self.metrics["warnings"].append(" ".join(map(str, args)))

    # ── Document Progress ─────────────────────────────────

    def doc_start(self, doc_idx: int, source: str, num_chunks: int):
        """Start processing a document."""
        self._doc_count += 1
        self._chunk_count += num_chunks
        self._print(f"[Doc {doc_idx}] {source}  → {num_chunks} chunks")

    def doc_done(self, doc_idx: int, success: int, total: int, elapsed: float):
        """Finished processing a document."""
        self._print(f"[Doc {doc_idx}] completed {success}/{total} chunks | elapsed {elapsed:.1f}s")

    # ── Chunk Extraction Results ──────────────────────────

    def chunk_extracted(self, chunk_id: str, n_entities: int, n_relations: int, elapsed: float = 0, cached: bool = False):
        """Entities and triplets extracted by LLM from a single chunk."""
        self._entity_count += n_entities
        self._triplet_count += n_relations
        self.metrics["entities_per_chunk"].append(n_entities)
        self.metrics["triplets_per_chunk"].append(n_relations)
        self.metrics["extract_time"].append(elapsed)
        if n_entities > 0:
            status = " |cache = true" if cached else ""
            self._print(
                f"  extract {chunk_id[:16]}... | entities {n_entities}, relations {n_relations}"
                f"{f' | LLM {elapsed:.1f}s' if elapsed else ''}{status}"
            )

    def chunk_empty(self, chunk_id: str):
        """No entities extracted from this chunk."""
        self._print(f"  extract {chunk_id[:16]}... | empty result (skipped)")

    # ── Embedding Progress ────────────────────────────────

    def embedding_call(self, model: str, text_len: int, elapsed: float):
        """Log a single embedding API/model call."""
        self._embedding_count += 1
        self.metrics["embedding_time"].append(elapsed)
        self.metrics["embedding_text_len"].append(text_len)
        # Too noisy to print every call, batch output at intervals
        if self._embedding_count % 50 == 0:
            self._print(f"  Embedding total {self._embedding_count} calls | last {elapsed:.2f}s")

    # ── Retrieval ─────────────────────────────────────────

    def set_qa_mode(self, total: int):
        """Switch to QA progress bar mode; terminal shows only progress, details go to log file."""
        from tqdm import tqdm
        self._qa_mode = True
        self._pbar = tqdm(total=total, desc="QA", unit="q", ncols=80)

    def qa_progress(self, n: int = 1):
        """Update the progress bar."""
        if self._pbar:
            self._pbar.update(n)

    def qa_done(self):
        """Close the progress bar and restore terminal output."""
        if self._pbar:
            self._pbar.close()
            self._pbar = None
        self._qa_mode = False

    def query_start(self, q_idx: int, query: str):
        """Retrieval query started (log file only)."""
        pass

    def query_done(self, q_idx: int, n_chunks: int, elapsed: float):
        """Retrieval query done (log file only)."""
        self.metrics["retrieval_time"].append(elapsed)
        self.metrics["retrieved_chunks"].append(n_chunks)

    def set_token_usage(self, prompt: int, completion: int):
        """Set LLM token usage (called by PurifyGraphRAG before shutdown)."""
        self._prompt_tokens = prompt
        self._completion_tokens = completion

    # ── Summary ───────────────────────────────────────────

    def summary(self):
        """Print and save the summary report."""
        elapsed = time.time() - self._start_time
        avg_extract = sum(self.metrics.get("extract_time", [])) / max(len(self.metrics.get("extract_time", [])), 1)
        avg_embed = sum(self.metrics.get("embedding_time", [])) / max(len(self.metrics.get("embedding_time", [])), 1)
        avg_retrieval = sum(self.metrics.get("retrieval_time", [])) / max(len(self.metrics.get("retrieval_time", [])), 1)

        report = {
            "total_time_s": round(elapsed, 1),
            "documents": self._doc_count,
            "chunks": self._chunk_count,
            "entities": self._entity_count,
            "triplets": self._triplet_count,
            "embedding_calls": self._embedding_count,
            "avg_extract_time_s": round(avg_extract, 2),
            "avg_embedding_time_s": round(avg_embed, 2),
            "avg_retrieval_time_s": round(avg_retrieval, 2),
            "queries": len(self.metrics.get("retrieval_time", [])),
            "retrieved_chunks_total": sum(self.metrics.get("retrieved_chunks", [])),
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
        }

        total_tokens = self._prompt_tokens + self._completion_tokens
        lines = [
            "",
            "=" * 50,
            "  Pipeline Summary",
            "=" * 50,
            f"  Total time      {report['total_time_s']:.1f} s",
            f"  Documents       {report['documents']}",
            f"  Chunks          {report['chunks']}",
            f"  Entities        {report['entities']}",
            f"  Triplets        {report['triplets']}",
            f"  Embedding calls {report['embedding_calls']}",
            f"  Avg extract     {report['avg_extract_time_s']:.2f} s",
            f"  Avg embedding   {report['avg_embedding_time_s']:.2f} s",
            f"  Avg retrieval   {report['avg_retrieval_time_s']:.2f} s",
            f"  Queries         {report['queries']}",
            f"  Retrieved total {report['retrieved_chunks_total']}",
        ]
        if total_tokens:
            lines.append(f"  LLM prompt tokens  {self._prompt_tokens}")
            lines.append(f"  LLM output tokens  {self._completion_tokens}")
            lines.append(f"  Total tokens       {total_tokens}")
        warnings = self.metrics.get("warnings", [])
        if warnings:
            lines.append(f"  Warnings           {len(warnings)}")
        lines.extend(["=" * 50, ""])
        for line in lines:
            self._print(line)

        # Save JSON
        with open(self._json_file, "w", encoding="utf-8") as f:
            json.dump({**report, **dict(self.metrics)}, f, indent=2, ensure_ascii=False)
        self._print(f"JSON report saved → {self._json_file}")

    def close(self):
        """Close the log file."""
        self._handle.close()


# Global singleton (shared across pipeline modules)
_global_logger: Optional[PipelineLogger] = None


def set_logger(logger: PipelineLogger):
    global _global_logger
    _global_logger = logger


def get_logger() -> Optional[PipelineLogger]:
    return _global_logger
