"""Command-line entry point: `python -m app.cli --help`."""
import argparse
import os

from app import core

CORPUS_DIR = os.environ.get("CORPUS_DIR", "corpus")


def cmd_prepare(args) -> None:
    stats = core.prepare_corpus(args.raw, args.out)
    print(f"Prepared {stats['interactions']} interactions -> {args.out}/{core.AGENT_FILE}")
    print(f"  agent responses: {stats['responses']}, tokens: {stats['tokens']}, vocabulary: {stats['vocabulary']}")
    print("  placeholders: " + ", ".join(f"{p} x{n}" for p, n in stats["placeholders"].items()))


def cmd_evaluate(args) -> None:
    sequences = core.load_sequences(args.corpus)
    rows = core.evaluate_models(sequences, orders=args.orders, smoothings=args.smoothings, seed=args.seed)
    print(f"Held-out evaluation on {len(sequences)} responses (80/20 split, seed {args.seed}). Lower perplexity is better.")
    print(f"{'n':>2}  {'smoothing':<12} {'perplexity':>10} {'top-1':>7} {'top-3':>7} {'top-5':>7}")
    for r in rows:
        print(f"{r['n']:>2}  {r['smoothing']:<12} {r['perplexity']:>10.2f} {r['top1']:>7.1%} {r['top3']:>7.1%} {r['top5']:>7.1%}")
    best = min(rows, key=lambda r: r["perplexity"])
    print(f"Best: n={best['n']} with {best['smoothing']} smoothing (perplexity {best['perplexity']:.2f}).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="support-response-generator", description="Customer support agents use this CLI tool to quickly draft responses to common queries by generating next-word suggestions based on a corpus of past support interactions.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    p = sub.add_parser("prepare", help="anonymize and clean the raw tickets into the training corpus")
    p.add_argument("--raw", default=f"{CORPUS_DIR}/raw_interactions.txt", help="raw tickets file")
    p.add_argument("--out", default=CORPUS_DIR, help="folder for the prepared corpus")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("evaluate", help="compare n-gram orders and smoothing techniques on held-out responses")
    p.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4], help="n-gram orders to try")
    p.add_argument("--smoothings", nargs="+", choices=core.SMOOTHINGS, default=list(core.SMOOTHINGS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corpus", default=CORPUS_DIR, help="folder with the prepared corpus")
    p.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "hello":
        print("Support Response Generator is ready. Open README.md for your milestones.")
        return
    args.func(args)


if __name__ == "__main__":
    main()
