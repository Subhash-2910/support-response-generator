"""Command-line entry point: `python -m app.cli --help`.

    python -m app.cli prepare                         anonymize and clean the raw tickets
    python -m app.cli next "Thank you for"            top-N next words
    python -m app.cli suggest "I have issued a full"  full continuations
    python -m app.cli interactive                     draft a response word by word
    python -m app.cli add "Your parcel is on its way."   teach the model a new response
    python -m app.cli evaluate                        compare n-gram orders and smoothing techniques
"""
import argparse
import os
import sys

from app import core

CORPUS_DIR = os.environ.get("CORPUS_DIR", "corpus")


def _model(args) -> core.NGramModel:
    try:
        return core.build_model(args.corpus, n=args.n, smoothing=args.smoothing)
    except FileNotFoundError as err:
        sys.exit(str(err))


def _text(args) -> str:
    return " ".join(args.text)


def _fill_in_note(text: str) -> str:
    holes = core.placeholders_in(text)
    return f"  (fill in: {', '.join(holes)})" if holes else ""


# --- commands

def cmd_prepare(args) -> None:
    stats = core.prepare_corpus(args.raw, args.out)
    print(f"Prepared {stats['interactions']} interactions -> {args.out}/{core.AGENT_FILE}")
    print(f"  agent responses: {stats['responses']}, tokens: {stats['tokens']}, vocabulary: {stats['vocabulary']}")
    print("  placeholders: " + ", ".join(f"{p} x{n}" for p, n in stats["placeholders"].items()))


def cmd_next(args) -> None:
    model, text = _model(args), _text(args)
    print(f'Next word after "{text}":' if text else "First word of a response:")
    for i, (word, p) in enumerate(core.suggest_next_words(model, text, top_n=args.top), 1):
        print(f"  {i}. {word:<16} {p:6.1%}")


def cmd_suggest(args) -> None:
    model, text = _model(args), _text(args)
    alternatives = core.suggest_continuations(model, text, n=args.top, max_words=args.max_words,
                                              stop_at_sentence_end=not args.keep_going)
    if not alternatives:
        print("No suggestion: the model has nothing likely to add here.")
        return
    for i, (tokens, p) in enumerate(alternatives, 1):
        full = core.complete(text, tokens) if text else core.detokenize(tokens)
        print(f"  {i}. {full}{_fill_in_note(full)}")


def cmd_add(args) -> None:
    try:
        clean = core.add_response(_text(args), args.corpus)
    except ValueError as err:
        sys.exit(str(err))
    print(f"Added to {args.corpus}/{core.CONTRIBUTED_FILE} (anonymized):\n  {clean}")


def cmd_evaluate(args) -> None:
    sequences = core.load_sequences(args.corpus)
    rows = core.evaluate_models(sequences, orders=args.orders, smoothings=args.smoothings, seed=args.seed)
    print(f"Held-out evaluation on {len(sequences)} responses (80/20 split, seed {args.seed}). Lower perplexity is better.")
    print(f"{'n':>2}  {'smoothing':<12} {'perplexity':>10} {'top-1':>7} {'top-3':>7} {'top-5':>7}")
    for r in rows:
        print(f"{r['n']:>2}  {r['smoothing']:<12} {r['perplexity']:>10.2f} {r['top1']:>7.1%} {r['top3']:>7.1%} {r['top5']:>7.1%}")
    best = min(rows, key=lambda r: r["perplexity"])
    print(f"Best: n={best['n']} with {best['smoothing']} smoothing (perplexity {best['perplexity']:.2f}).")


INTERACTIVE_HELP = """How to draft:
  1-9        add that next word            c1-c3   add that whole continuation
  <Enter>    add continuation c1           any other text is added as you typed it
  /undo      take back the last step       /clear  start again
  /save      add the draft to the corpus   /done   finish and print the draft
  /help      show this help"""


def run_interactive(model: core.NGramModel, top: int = 5, corpus_dir: str = CORPUS_DIR,
                    read=input, write=print, rebuild=None) -> str:
    """The drafting loop. `read`, `write` and `rebuild` are parameters so tests can drive it without a terminal."""
    write("Support Response Generator: interactive drafting. Type /help for help.")
    write(INTERACTIVE_HELP)
    draft, history = "", []
    while True:
        words = core.suggest_next_words(model, draft, top_n=top)
        conts = core.suggest_continuations(model, draft, n=3)
        write("")
        write(f"Draft: {draft or '(empty)'}")
        write("  Next word:  " + "   ".join(f"[{i}] {w} ({p:.0%})" for i, (w, p) in enumerate(words, 1)))
        for i, (tokens, _) in enumerate(conts, 1):
            write(f"  [c{i}] ...{core.detokenize(tokens, draft)}")
        try:
            choice = read("> ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = "/done"

        if choice in ("/done", "/quit", "/q"):
            write("")
            write("Final response:")
            write(draft + _fill_in_note(draft))
            return draft
        if choice == "/help":
            write(INTERACTIVE_HELP)
            continue
        if choice == "/undo":
            draft = history.pop() if history else draft
            continue
        if choice == "/clear":
            history.append(draft)
            draft = ""
            continue
        if choice == "/save":
            if draft.strip():
                write(f"Saved to the corpus: {core.add_response(draft, corpus_dir)}")
                if rebuild:
                    model = rebuild()
            else:
                write("Nothing to save yet.")
            continue
        if choice.startswith("/"):
            write(f"Unknown command {choice}. Type /help for help.")
            continue

        new = None
        if choice == "" and conts:
            new = core.complete(draft, conts[0][0]) if draft else core.detokenize(conts[0][0])
        elif choice.isdigit() and 1 <= int(choice) <= len(words):
            new = draft.rstrip() + core.detokenize([words[int(choice) - 1][0]], draft)
        elif len(choice) == 2 and choice[0] == "c" and choice[1].isdigit() and 1 <= int(choice[1]) <= len(conts):
            new = core.complete(draft, conts[int(choice[1]) - 1][0]) if draft else core.detokenize(conts[int(choice[1]) - 1][0])
        elif choice:
            new = f"{draft} {choice}".strip() if draft else choice
        if new is None:
            write("Nothing to add here. Type some words, or /done.")
            continue
        history.append(draft)
        draft = new.strip()


def cmd_interactive(args) -> None:
    model = _model(args)
    run_interactive(model, top=args.top, corpus_dir=args.corpus, rebuild=lambda: _model(args))


# --- argument parsing

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="support-response-generator", description="Customer support agents use this CLI tool to quickly draft responses to common queries by generating next-word suggestions based on a corpus of past support interactions.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    def model_options(p):
        p.add_argument("--n", type=int, default=4, help="n-gram order (default 4)")
        p.add_argument("--smoothing", choices=core.SMOOTHINGS, default="kneser_ney", help="default kneser_ney")
        p.add_argument("--corpus", default=CORPUS_DIR, help="folder with the prepared corpus")

    p = sub.add_parser("prepare", help="anonymize and clean the raw tickets into the training corpus")
    p.add_argument("--raw", default=f"{CORPUS_DIR}/raw_interactions.txt", help="raw tickets file")
    p.add_argument("--out", default=CORPUS_DIR, help="folder for the prepared corpus")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("next", help="show the top-N next words for a partial response")
    p.add_argument("text", nargs="*", help="the response so far (empty = the first word)")
    p.add_argument("--top", type=int, default=5, help="how many words to show")
    model_options(p)
    p.set_defaults(func=cmd_next)

    p = sub.add_parser("suggest", help="suggest complete continuations of a partial response")
    p.add_argument("text", nargs="*", help="the response so far (empty = suggest an opening)")
    p.add_argument("--top", type=int, default=3, help="how many alternatives to show")
    p.add_argument("--max-words", type=int, default=20, help="longest continuation, in tokens")
    p.add_argument("--keep-going", action="store_true", help="do not stop at the end of a sentence")
    model_options(p)
    p.set_defaults(func=cmd_suggest)

    p = sub.add_parser("interactive", help="draft a response by picking suggestions as you go")
    p.add_argument("--top", type=int, default=5, help="how many next words to show")
    model_options(p)
    p.set_defaults(func=cmd_interactive)

    p = sub.add_parser("add", help="add a good response to the corpus (it is anonymized first)")
    p.add_argument("text", nargs="+", help="the response to add")
    p.add_argument("--corpus", default=CORPUS_DIR, help="folder with the prepared corpus")
    p.set_defaults(func=cmd_add)

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
