from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.dataset.runner import run_batch
from src.pipeline import run_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P.L.U.M.E. Advanced cave generation prototype")
    parser.add_argument("command", choices=["run", "batch"])
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--samples", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    output_root = Path(args.output or config.dataset.output_dir)

    if args.command == "run":
        run_dir = output_root / f"{config.name}-{config.seed}"
        run_simulation(config, run_dir)
    else:
        run_batch(config, output_root, sample_count=args.samples)


if __name__ == "__main__":
    main()

