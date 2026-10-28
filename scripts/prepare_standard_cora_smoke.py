"""Export the deterministic standard-Cora smoke split used by the runner."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tt_greb_reproduce.data import standard_cora_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("TT_GREB_Reproduce/data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    splits = standard_cora_smoke(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train": splits.train, "valid": splits.valid, "test": splits.test}, args.output)
    print(f"wrote smoke split: {args.output}")


if __name__ == "__main__":
    main()
