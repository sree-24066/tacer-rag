"""
Inspect RAGTruth data files and report statistics.

Usage:
    python scripts/inspect_data.py

Requires Phase 1 (cloning RAGTruth) to be complete.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from tacer_rag.data.inspect import inspect_ragtruth
from tacer_rag.utils import get_logger, project_root

logger = get_logger("inspect_script")


def main() -> None:
    root = project_root()
    response_file = root / "external" / "RAGTruth" / "data" / "response.jsonl"
    source_file = root / "external" / "RAGTruth" / "data" / "source_info.jsonl"
    output_dir = root / "data" / "processed"

    if not response_file.exists():
        logger.error(
            "RAGTruth response.jsonl not found.\n"
            "Please complete Phase 1 (clone RAGTruth) first:\n"
            "  cd external\n"
            "  git clone https://github.com/ParticleMedia/RAGTruth.git"
        )
        sys.exit(1)

    stats = inspect_ragtruth(response_file, source_file, output_dir)

    print("\n" + "=" * 60)
    print("INSPECTION SUMMARY")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:<45} {v}")
    print("=" * 60)
    print(f"\nFull stats saved to: {output_dir / 'inspection_stats.json'}")


if __name__ == "__main__":
    main()
