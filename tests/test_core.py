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


# Milestone 2: Implement the n-gram model with smoothing.
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Create a text generation function to suggest response continuations.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Build a command-line interface for agents to get suggestions.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")
