"""Command line helpers for local reproduction."""

from __future__ import annotations

import argparse

from .facade import evaluate_experiment, prepare_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tool-hallu")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--quick", action="store_true")
    prepare_parser.add_argument("--seed", type=int, default=42)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--quick", action="store_true")
    evaluate_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        dataset = prepare_dataset(quick=args.quick, seed=args.seed)
        for split, records in dataset.items():
            print(f"{split}: {len(records)} records")
        return

    if args.command == "evaluate":
        result = evaluate_experiment(quick=args.quick, seed=args.seed)
        print("Sentence metrics")
        print(result["sentence_metrics"])
        print("\nSpan metrics")
        print(result["span_metrics"])
        return


if __name__ == "__main__":
    main()
