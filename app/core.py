"""The next-word predictor, built from scratch with the standard library only.

Milestone 1  corpus:      parse raw tickets, anonymize personal details, clean and tokenize the text.
Milestone 2  model:       an n-gram model with add-k, interpolation or Kneser-Ney smoothing.
Milestone 3  generation:  next-word suggestions, greedy continuations and beam-search alternatives.
Milestone 4  CLI support: build a model from the corpus folder, contribute new responses (the CLI is app/cli.py).
"""
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# =====================================================================================================================
# --- Milestone 1: Prepare an anonymized corpus of customer support interactions.
# Raw tickets -> anonymized (personal details replaced by placeholders such as <EMAIL>) -> normalized -> tokens.
# =====================================================================================================================

PLACEHOLDERS = ("<NAME>", "<EMAIL>", "<PHONE>", "<ORDER_ID>", "<CARD>", "<TRACKING>", "<ADDRESS>", "<URL>",
                "<NUMBER>")

_STREET_WORDS = r"(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Court|Ct|Place|Pl)"
_NAME = r"[A-Z][a-zà-ÿ]+(?:[-'][A-Z][a-zà-ÿ]+)?"

# Order matters: the most specific patterns run first, so a card number is not mistaken for a phone number.
_PII_PATTERNS = [
    ("<EMAIL>", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("<URL>", re.compile(r"\b(?:https?://|www\.)\S+", re.I)),
    ("<ADDRESS>", re.compile(rf"\b\d{{1,5}}\s+(?:[A-Z][a-z]+\s+){{1,3}}{_STREET_WORDS}\b\.?"
                             rf"(?:,?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)?(?:,?\s+[A-Z]{{2}})?(?:\s+\d{{5}}(?:-\d{{4}})?)?")),
    ("<CARD>", re.compile(r"\b(?:\d{4}[ -]?){3}\d{1,7}\b")),
    ("<CARD>", re.compile(r"(?<=\bending in )\d{4}\b", re.I)),
    ("<TRACKING>", re.compile(r"\b1Z[0-9A-Z]{16}\b")),
    ("<PHONE>", re.compile(r"(?<![\w#-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b")),
    ("<ORDER_ID>", re.compile(r"(?:#|\bORD-)\d{4,}\b|(?<=\border )\d{5,}\b|(?<=\border number )\d{5,}\b", re.I)),
    ("<NUMBER>", re.compile(r"\b\d{5,}\b")),
]

# A capitalised word right after one of these cues is taken to be a person's name.
_NAME_CUES = re.compile(
    rf"(?:\b(?:Hi|Hello|Hey|Dear|Thanks|Thank you|Cheers|[Rr]egards|Best|Sincerely)\b,?|\b[Mm]y name is|\bI'm|"
    rf"\bI am|\b[Tt]his is)\s+({_NAME}(?:\s+{_NAME})?)")
_TRAILING_NAME = re.compile(rf"(?<=[.!?,])\s+({_NAME})\s*$")  # a sign-off such as "Have a great day! Alex"
_NOT_NAMES = {"There", "Team", "All", "Everyone", "Customer", "Support", "Again", "So", "For", "Very", "The", "Sorry",
              "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Please", "Yes", "No",
              "Ok", "Okay", "Sir", "Madam", "We", "You", "It", "This", "That", "Your", "My", "Our", "In", "If"}


@dataclass
class Interaction:
    ticket: str
    intent: str
    customer: str
    agent: str


def parse_raw(text: str) -> list[Interaction]:
    """Read tickets written as '### Ticket <id> | <intent>' then 'Customer: ...' and 'Agent: ...' lines."""
    out = []
    for block in re.split(r"^###\s*", text, flags=re.M):
        if not block.strip():
            continue
        header, _, body = block.partition("\n")
        ticket, _, intent = (part.strip() for part in header.partition("|"))
        customer = re.search(r"^Customer:\s*(.+)$", body, re.M)
        agent = re.search(r"^Agent:\s*(.+)$", body, re.M)
        if agent:
            out.append(Interaction(ticket=ticket.replace("Ticket", "").strip(), intent=intent or "unknown",
                                   customer=customer.group(1).strip() if customer else "",
                                   agent=agent.group(1).strip()))
    return out


def find_names(text: str) -> set[str]:
    """Names found next to greeting/sign-off cues ('Hi Sarah', 'my name is Sarah Lee', 'Thanks, Sam')."""
    names = set()
    for match in list(_NAME_CUES.finditer(text)) + list(_TRAILING_NAME.finditer(text)):
        for word in match.group(1).split():
            if word not in _NOT_NAMES:
                names.add(word)
    return names


def anonymize(text: str, known_names: set[str] | frozenset = frozenset()) -> str:
    """Replace personal details with placeholders. `known_names` adds names found elsewhere in the corpus."""
    for placeholder, pattern in _PII_PATTERNS:
        text = pattern.sub(placeholder, text)
    names = find_names(text) | set(known_names)
    if names:
        alternatives = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
        text = re.sub(rf"\b(?:{alternatives})\b", "<NAME>", text)
    # a capitalised word straight after a name is a surname ("<NAME> Johnson"), unless it is a common word
    text = re.sub(rf"<NAME>\s+({_NAME})\b", lambda m: m.group(0) if m.group(1) in _NOT_NAMES else "<NAME>", text)
    text = re.sub(r"<NAME>(?:\s+<NAME>)+", "<NAME>", text)  # "Sarah Johnson" -> one <NAME>
    return text


def normalize(text: str) -> str:
    """Unicode clean-up: curly quotes and dashes become plain ones, whitespace is collapsed."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                                         "–": "-", "—": "-"}))
    return re.sub(r"\s+", " ", text).strip()


_TOKEN = re.compile(r"<[A-Z_]+>|[^\W\d_]+(?:'[^\W\d_]+)*|\d+(?:\.\d+)?|[.,!?;:%$&/]")


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens; placeholders stay as they are and punctuation marks are their own tokens."""
    return [t if t in PLACEHOLDERS else t.lower() for t in _TOKEN.findall(normalize(text))]


def anonymize_interactions(interactions: list[Interaction]) -> list[Interaction]:
    """Two passes: collect every name the cues reveal anywhere, then remove those names everywhere.
    So a name seen once as 'Hi Sarah' is also caught where it appears without a cue."""
    names = set()
    for it in interactions:
        names |= find_names(it.customer) | find_names(it.agent)
    return [Interaction(it.ticket, it.intent, normalize(anonymize(it.customer, names)),
                        normalize(anonymize(it.agent, names))) for it in interactions]


AGENT_FILE, CUSTOMER_FILE, CONTRIBUTED_FILE = "agent_responses.txt", "customer_messages.txt", "contributed.txt"


def prepare_corpus(raw_path: str | Path = "corpus/raw_interactions.txt", out_dir: str | Path = "corpus") -> dict:
    """Raw tickets -> anonymized agent responses and customer messages, one per line. Returns corpus statistics."""
    interactions = anonymize_interactions(parse_raw(Path(raw_path).read_text(encoding="utf-8")))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / AGENT_FILE).write_text("\n".join(it.agent for it in interactions) + "\n", encoding="utf-8")
    (out_dir / CUSTOMER_FILE).write_text("\n".join(it.customer for it in interactions) + "\n", encoding="utf-8")
    return corpus_stats([tokenize(it.agent) for it in interactions]) | {"interactions": len(interactions)}


def corpus_stats(sequences: list[list[str]]) -> dict:
    tokens = [t for seq in sequences for t in seq]
    placeholders = {p: tokens.count(p) for p in PLACEHOLDERS if p in tokens}
    return {"responses": len(sequences), "tokens": len(tokens), "vocabulary": len(set(tokens)),
            "placeholders": placeholders}


def load_sequences(corpus_dir: str | Path = "corpus", include_customers: bool = False) -> list[list[str]]:
    """Tokenized training sequences: the prepared agent responses plus anything agents contributed."""
    corpus_dir = Path(corpus_dir)
    names = [AGENT_FILE, CONTRIBUTED_FILE] + ([CUSTOMER_FILE] if include_customers else [])
    if not (corpus_dir / AGENT_FILE).exists():
        raise FileNotFoundError(f"no prepared corpus in {corpus_dir}/ - run `python -m app.cli prepare` first")
    sequences = []
    for name in names:
        path = corpus_dir / name
        if path.exists():
            sequences += [tokenize(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [s for s in sequences if s]
