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


# =====================================================================================================================
# --- Milestone 2: Implement the n-gram model with smoothing.
# Count which word follows each context of n-1 words. Smoothing gives unseen sequences a small, non-zero probability:
#   add_k        (c(ctx, w) + k) / (c(ctx) + k|V|)                       k = 1 is Laplace smoothing
#   interpolate  lambda * P_ML(w | ctx) + (1 - lambda) * P(w | shorter ctx), down to an add-k unigram
#   kneser_ney   interpolated Kneser-Ney: absolute discounting, with continuation counts for the lower orders
# =====================================================================================================================

BOS, EOS, UNK = "<s>", "</s>", "<unk>"
SMOOTHINGS = ("add_k", "interpolate", "kneser_ney")


class NGramModel:
    def __init__(self, n: int = 4, smoothing: str = "kneser_ney", k: float = 0.1, lam: float = 0.7,
                 discount: float = 0.75, min_count: int = 1):
        if n < 1:
            raise ValueError("n must be at least 1")
        if smoothing not in SMOOTHINGS:
            raise ValueError(f"smoothing must be one of {SMOOTHINGS}")
        self.n, self.smoothing, self.k, self.lam, self.discount, self.min_count = n, smoothing, k, lam, discount, min_count

    # --- training
    def fit(self, sequences: list[list[str]]) -> "NGramModel":
        word_counts: dict[str, int] = {}
        for seq in sequences:
            for w in seq:
                word_counts[w] = word_counts.get(w, 0) + 1
        self.vocab = sorted(w for w, c in word_counts.items() if c >= self.min_count) + [EOS, UNK]
        self._known = set(self.vocab)
        # counts[j][context of j words] = {next word: count}
        self.counts: list[dict[tuple, dict[str, int]]] = [{} for _ in range(self.n)]
        for seq in sequences:
            padded = [BOS] * (self.n - 1) + [self._map(w) for w in seq] + [EOS]
            for i in range(self.n - 1, len(padded)):
                word = padded[i]
                for j in range(self.n):
                    table = self.counts[j].setdefault(tuple(padded[i - j:i]), {})
                    table[word] = table.get(word, 0) + 1
        # Kneser-Ney continuation counts: in how many distinct contexts does `w` follow this shorter context?
        self.continuation: list[dict[tuple, dict[str, int]]] = [{} for _ in range(self.n)]
        for j in range(self.n - 1):
            for ctx, followers in self.counts[j + 1].items():
                table = self.continuation[j].setdefault(ctx[1:], {})
                for w in followers:
                    table[w] = table.get(w, 0) + 1
        self.totals = [{ctx: sum(c.values()) for ctx, c in level.items()} for level in self.counts]
        self.cont_totals = [{ctx: sum(c.values()) for ctx, c in level.items()} for level in self.continuation]
        return self

    def _map(self, word: str) -> str:
        return word if word in self._known else UNK

    def context_of(self, history: list[str]) -> tuple:
        """The last n-1 tokens, mapped into the vocabulary and padded on the left with <s>."""
        if self.n == 1:
            return ()
        padded = [BOS] * (self.n - 1) + [self._map(w) for w in history]
        return tuple(padded[-(self.n - 1):])

    # --- probabilities
    def prob(self, word: str, history: list[str] | tuple = ()) -> float:
        """P(word | the last n-1 words of history), smoothed. Never 0 for a word in the vocabulary."""
        return self._prob(self._map(word), self.context_of(list(history)))

    def _prob(self, w: str, ctx: tuple) -> float:
        V = len(self.vocab)
        if self.smoothing == "add_k":
            c = self.counts[self.n - 1].get(ctx, {})
            return (c.get(w, 0) + self.k) / (self.totals[self.n - 1].get(ctx, 0) + self.k * V)
        if self.smoothing == "interpolate":
            p = (self.counts[0][()].get(w, 0) + self.k) / (self.totals[0][()] + self.k * V)
            for j in range(1, self.n):
                sub = ctx[len(ctx) - j:]
                total = self.totals[j].get(sub, 0)
                if total:  # an unseen context gives no evidence, so all weight stays on the shorter context
                    p = self.lam * self.counts[j][sub].get(w, 0) / total + (1 - self.lam) * p
            return p
        # Kneser-Ney: the lowest order spreads the discounted mass uniformly, so unseen words keep some probability.
        d = self.discount
        table0, total0 = (self.continuation[0], self.cont_totals[0]) if self.n > 1 else (self.counts[0], self.totals[0])
        c0, t0 = table0.get((), {}), total0.get((), 0)
        p = (max(c0.get(w, 0) - d, 0) + d * len(c0) / V) / t0 if t0 else 1 / V
        for j in range(1, self.n):
            sub = ctx[len(ctx) - j:]
            table, totals = (self.counts, self.totals) if j == self.n - 1 else (self.continuation, self.cont_totals)
            c, total = table[j].get(sub), totals[j].get(sub, 0)
            if total:
                p = (max(c.get(w, 0) - d, 0) + d * len(c) * p) / total
        return p

    def distribution(self, history: list[str] | tuple = ()) -> dict[str, float]:
        """P(w | history) for every word in the vocabulary (sums to 1)."""
        ctx = self.context_of(list(history))
        return {w: self._prob(w, ctx) for w in self.vocab}

    def top_next(self, history: list[str] | tuple = (), k: int = 5, exclude=(UNK,)) -> list[tuple[str, float]]:
        dist = self.distribution(history)
        ranked = sorted(((w, p) for w, p in dist.items() if w not in exclude), key=lambda wp: (-wp[1], wp[0]))
        return ranked[:k]

    # --- evaluation
    def perplexity(self, sequences: list[list[str]]) -> float:
        """2 ** (average negative log2 probability per token, counting the end-of-response token). Lower is better."""
        import math
        log_sum, count = 0.0, 0
        for seq in sequences:
            history: list[str] = []
            for w in list(seq) + [EOS]:
                log_sum -= math.log2(self.prob(w, history))
                count += 1
                history.append(w)
        return 2 ** (log_sum / count) if count else float("inf")


def train_test_split(sequences: list, test_fraction: float = 0.2, seed: int = 0) -> tuple[list, list]:
    """The same shuffled split for the same seed."""
    import random
    order = list(range(len(sequences)))
    random.Random(seed).shuffle(order)
    n_test = max(1, int(round(test_fraction * len(sequences))))
    test = set(order[:n_test])
    return [s for i, s in enumerate(sequences) if i not in test], [s for i, s in enumerate(sequences) if i in test]


def next_word_accuracy(model: NGramModel, sequences: list[list[str]], ks=(1, 3, 5)) -> dict:
    """Share of positions where the true next word is among the model's top-k guesses."""
    hits, total = {k: 0 for k in ks}, 0
    for seq in sequences:
        for i, word in enumerate(list(seq) + [EOS]):
            guesses = [w for w, _ in model.top_next(seq[:i], k=max(ks))]
            total += 1
            for k in ks:
                hits[k] += word in guesses[:k]
    return {f"top{k}": hits[k] / total for k in ks} | {"positions": total}


def evaluate_models(sequences: list[list[str]], orders=(1, 2, 3, 4), smoothings=SMOOTHINGS,
                    test_fraction: float = 0.2, seed: int = 0) -> list[dict]:
    """Train every (order, smoothing) pair on the same split; report held-out perplexity and next-word accuracy."""
    train, test = train_test_split(sequences, test_fraction, seed)
    rows = []
    for n in orders:
        for smoothing in smoothings:
            model = NGramModel(n=n, smoothing=smoothing).fit(train)
            rows.append({"n": n, "smoothing": smoothing, "perplexity": model.perplexity(test),
                         **next_word_accuracy(model, test)})
    return rows
