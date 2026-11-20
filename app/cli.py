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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="support-response-generator", description="Customer support agents use this CLI tool to quickly draft responses to common queries by generating next-word suggestions based on a corpus of past support interactions.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    p = sub.add_parser("prepare", help="anonymize and clean the raw tickets into the training corpus")
    p.add_argument("--raw", default=f"{CORPUS_DIR}/raw_interactions.txt", help="raw tickets file")
    p.add_argument("--out", default=CORPUS_DIR, help="folder for the prepared corpus")
    p.set_defaults(func=cmd_prepare)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "hello":
        print("Support Response Generator is ready. Open README.md for your milestones.")
        return
    args.func(args)


if __name__ == "__main__":
    main()
