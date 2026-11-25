"""
Data download helper (Phase 1).

This script verifies that the RAGTruth repository has been cloned
and that the expected data files exist. It does NOT modify the
cloned repository.

Usage:
    python scripts/download_data.py [--check-only]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from tacer_rag.utils import get_logger, project_root

logger = get_logger("download")

RAGTRUTH_REPO_URL = "https://github.com/ParticleMedia/RAGTruth.git"
LETTUCEDETECT_REPO_URL = "https://github.com/KRLabsOrg/LettuceDetect.git"

EXPECTED_RAGTRUTH_FILES = [
    "data/response.jsonl",
    "data/source_info.jsonl",
]


def check_ragtruth(external_dir: Path) -> bool:
    """Check whether RAGTruth is already cloned and data files exist."""
    repo_dir = external_dir / "RAGTruth"
    if not repo_dir.exists():
        logger.warning("RAGTruth not cloned at: %s", repo_dir)
        return False

    all_present = True
    for rel_path in EXPECTED_RAGTRUTH_FILES:
        f = repo_dir / rel_path
        if f.exists():
            size_kb = f.stat().st_size // 1024
            logger.info("  FOUND: %s  (%d KB)", rel_path, size_kb)
        else:
            logger.warning("  MISSING: %s", rel_path)
            all_present = False

    return all_present


def print_clone_instructions(external_dir: Path) -> None:
    logger.info("\n" + "="*60)
    logger.info("Clone instructions:")
    logger.info("  cd \"%s\"", external_dir)
    logger.info("  git clone %s", RAGTRUTH_REPO_URL)
    logger.info("  git clone %s", LETTUCEDETECT_REPO_URL)
    logger.info("="*60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check RAGTruth data availability")
    parser.add_argument("--check-only", action="store_true", default=True)
    args = parser.parse_args()

    root = project_root()
    external_dir = root / "external"

    logger.info("Checking RAGTruth repository at: %s", external_dir / "RAGTruth")
    ok = check_ragtruth(external_dir)

    if ok:
        logger.info("RAGTruth data files found. Ready for Phase 1 inspection.")
    else:
        logger.warning("RAGTruth data files not found.")
        print_clone_instructions(external_dir)
        logger.info("After cloning, re-run this script to verify.")
        sys.exit(1)


if __name__ == "__main__":
    main()
