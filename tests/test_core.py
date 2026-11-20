"""Tests for each milestone in README.md."""
import pytest

from app import core

RAW = """### Ticket 1 | refund
Customer: Hi, my name is Sarah Johnson. I was charged twice for #482913 on my card 4111 1111 1111 1111. Email sarah.j@gmail.com
Agent: Hi Sarah, thank you for contacting us. I have refunded your card ending in 1111. Call me at (555) 123-4567. Thanks, Alex

### Ticket 2 | address
Customer: Please send it to 12 Maple Street, Springfield 12345 instead. Tracking 1Z999AA10123456784.
Agent: Hello Sarah, I have updated the address for order 4829130. Best regards, Alex
"""


# --- Milestone 1: Prepare an anonymized corpus of customer support interactions.

def test_anonymize_replaces_personal_details():
    text = ("My name is Sarah Johnson, email sarah.j@gmail.com, phone (555) 123-4567, card 4111 1111 1111 1111, "
            "order #482913, tracking 1Z999AA10123456784, address 12 Maple Street, Springfield 12345, "
            "see https://shop.example.com/x")
    out = core.anonymize(text)
    for secret in ["Sarah", "Johnson", "gmail", "555", "4111", "482913", "1Z999", "Maple", "https"]:
        assert secret not in out
    for placeholder in ["<NAME>", "<EMAIL>", "<PHONE>", "<CARD>", "<ORDER_ID>", "<TRACKING>", "<ADDRESS>", "<URL>"]:
        assert placeholder in out


def test_anonymize_keeps_ordinary_text():
    text = "I'm sorry for the inconvenience. Your refund will arrive within 5 business days."
    assert core.anonymize(text) == text


def test_names_found_once_are_removed_everywhere():
    interactions = core.anonymize_interactions(core.parse_raw(RAW))
    assert [it.intent for it in interactions] == ["refund", "address"]
    everything = " ".join(it.customer + " " + it.agent for it in interactions)
    assert "Sarah" not in everything and "Alex" not in everything and "Johnson" not in everything
    assert interactions[0].agent.startswith("Hi <NAME>, thank you for contacting us.")


def test_tokenize_lowercases_and_keeps_placeholders_and_punctuation():
    assert core.tokenize("Hi <NAME>, I’m  SORRY!") == ["hi", "<NAME>", ",", "i'm", "sorry", "!"]


def test_prepare_corpus_writes_clean_files(tmp_path):
    raw = tmp_path / "raw.txt"
    raw.write_text(RAW)
    stats = core.prepare_corpus(raw, tmp_path)
    assert stats["interactions"] == 2 and stats["placeholders"]["<NAME>"] >= 2
    agent_lines = (tmp_path / core.AGENT_FILE).read_text().splitlines()
    assert len(agent_lines) == 2 and "@" not in "".join(agent_lines)
    sequences = core.load_sequences(tmp_path)
    assert sequences[0][:3] == ["hi", "<NAME>", ","]


def test_the_shipped_corpus_has_no_personal_details():
    text = (open("corpus/agent_responses.txt").read() + open("corpus/customer_messages.txt").read())
    assert "@" not in text and "1Z" not in text
    assert not any(ch.isdigit() and len(word) >= 5 for word in text.split() for ch in word[:1])


# --- Milestone 2: Implement the n-gram model with smoothing.

TOY = [core.tokenize(t) for t in [
    "Thank you for contacting us.",
    "Thank you for your patience.",
    "Thank you for contacting our support team.",
    "We will send you a replacement.",
]]


@pytest.mark.parametrize("smoothing", core.SMOOTHINGS)
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_probabilities_sum_to_one_even_for_unseen_contexts(n, smoothing):
    model = core.NGramModel(n=n, smoothing=smoothing).fit(TOY)
    for history in ([], ["thank", "you", "for"], ["never", "seen", "words"]):
        assert sum(model.distribution(history).values()) == pytest.approx(1.0)


@pytest.mark.parametrize("smoothing", core.SMOOTHINGS)
def test_smoothing_gives_unseen_sequences_a_small_probability(smoothing):
    model = core.NGramModel(n=3, smoothing=smoothing).fit(TOY)
    seen = model.prob("contacting", ["you", "for"])
    unseen = model.prob("replacement", ["you", "for"])  # a known word that never followed "you for"
    assert 0 < unseen < seen
    assert model.prob("banana", ["you", "for"]) > 0  # an unknown word maps to <unk>


def test_counts_drive_the_most_likely_next_word():
    model = core.NGramModel(n=3).fit(TOY)
    assert model.top_next(["thank", "you", "for"], k=1)[0][0] == "contacting"  # seen 2 times out of 3
    assert model.top_next(["thank"], k=1)[0][0] == "you"


def test_add_k_matches_the_formula():
    model = core.NGramModel(n=2, smoothing="add_k", k=1).fit(TOY)
    V = len(model.vocab)
    # "for" is followed by "contacting" twice and "your" once
    assert model.prob("contacting", ["for"]) == pytest.approx((2 + 1) / (3 + V))
    assert model.prob("team", ["for"]) == pytest.approx(1 / (3 + V))


def test_rare_words_become_unk():
    model = core.NGramModel(n=2, min_count=2).fit(TOY)
    assert "replacement" not in model.vocab and core.UNK in model.vocab
    assert model.prob("replacement", ["a"]) == model.prob(core.UNK, ["a"])


def test_higher_order_models_fit_the_support_corpus_better():
    train, test = core.train_test_split(core.load_sequences(), test_fraction=0.2, seed=0)
    unigram = core.NGramModel(n=1).fit(train).perplexity(test)
    trigram = core.NGramModel(n=3).fit(train).perplexity(test)
    assert trigram < unigram / 5


# Milestone 3: Create a text generation function to suggest response continuations.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Build a command-line interface for agents to get suggestions.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")
