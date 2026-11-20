# Support Response Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/3p6ptew6ad/badge.svg)](https://ask.thili.ai/b/3p6ptew6ad)

This CLI tool, designed for retail support agents, streamlines response drafting. By leveraging a next-word predictor trained on past customer interactions, it provides quick, relevant suggestions. The goal is to enhance efficiency and consistency in customer service.

## What you are building

**Goal.** You will build a command-line interface (CLI) tool that helps customer support agents draft responses by generating next-word suggestions based on a corpus of past support interactions.

**Done when.** The tool successfully processes an anonymized corpus, implements an n-gram model with smoothing, and generates relevant next-word suggestions through a functional command-line interface.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Prepare an anonymized corpus of customer support interactions.**  
      Hint: Use text preprocessing and tokenization techniques to clean and structure the data.
- [x] **2. Implement the n-gram model with smoothing.**  
      Hint: Implement the n-gram model and apply a smoothing technique to handle unseen sequences.
- [x] **3. Create a text generation function to suggest response continuations.**  
      Hint: Develop a function that uses your n-gram model to iteratively predict and suggest the most probable next words.
- [x] **4. Build a command-line interface for agents to get suggestions.**  
      Hint: Create a user-friendly command-line interface to input partial responses and display generated suggestions.

## Run it

The predictor uses only the Python standard library (no AI service, no API key, no download). The data is in the repo
(`corpus/raw_interactions.txt`). You need Docker, or Python 3.12.

**Reproduce everything in one command.** This prepares the corpus from the raw tickets, evaluates every model against
the baseline and writes [results/evaluation.md](results/evaluation.md) and
[results/sample_suggestions.md](results/sample_suggestions.md):

```bash
docker build -t support-response-generator . && docker run --rm -v "$PWD/results:/app/results" support-response-generator report
```

or, without Docker, `python -m app.cli report`.

**With Docker**

```bash
docker build -t support-response-generator .
docker run --rm support-response-generator suggest "Hi Sarah, thank you for"
docker run --rm -it -v "$PWD/corpus:/app/corpus" support-response-generator interactive   # -v keeps what you /save
docker run --rm --entrypoint pytest support-response-generator -q                          # run the tests
```

**With Python**

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.cli --help
pytest
```

### Commands

| Command | What it does |
|---|---|
| `prepare` | Anonymize and clean `corpus/raw_interactions.txt` into `corpus/agent_responses.txt` and `corpus/customer_messages.txt` |
| `next "thank you for" --top 5` | The top-N next words, with probabilities |
| `suggest "I have issued a full" --top 3` | Complete continuations (beam search), with placeholders to fill in |
| `interactive` | Draft a response step by step: pick a word by number, a continuation with `c1`-`c3` (Enter = `c1`), or type your own words; `/undo`, `/clear`, `/save`, `/done` |
| `add "Your parcel is on its way."` | Add a good response to `corpus/contributed.txt` (anonymized first); the model learns it on the next run |
| `evaluate` | Compare n-gram orders and smoothing techniques on held-out responses |
| `report` | Run the whole pipeline (prepare, evaluate, sample suggestions) and write `results/` |

`next`, `suggest` and `interactive` take `--n` (order, default 4) and `--smoothing` (`add_k`, `interpolate`, `kneser_ney`).

```text
$ python -m app.cli next "Hi Sarah, thank you for"
Next word after "Hi Sarah, thank you for":
  1. your              36.3%
  2. contacting        30.7%
  3. reaching          16.6%
  4. shopping          16.1%

$ python -m app.cli suggest "Your order"
  1. Your order <ORDER_ID> has already been shipped, so we cannot change the address.  (fill in: <ORDER_ID>)
  2. Your order <ORDER_ID>.  (fill in: <ORDER_ID>)
  3. Your order <ORDER_ID> has already been shipped, so we cannot cancel it.  (fill in: <ORDER_ID>)
```

## How it works

1. **Corpus** (`app/core.py`, Milestone 1). `scripts/generate_raw_corpus.py` writes 600 synthetic retail tickets
   (orders, refunds, returns, damaged or wrong items, cancellations, address changes, payments, accounts, discount
   codes, stock). **Every personal detail in it is invented**, and it is there on purpose so the anonymizer has
   something to remove. The anonymizer replaces emails, URLs, addresses, card numbers, tracking numbers, phone
   numbers, order IDs and long numbers with placeholders (`<EMAIL>`, `<ORDER_ID>`, ...). Names are found next to
   cues ("Hi Sarah", "my name is ...", "Thanks, Alex") and then removed everywhere in the corpus, including surnames
   that follow a removed first name. Text is normalized (Unicode, quotes, whitespace) and tokenized into lower-case
   words, numbers and punctuation; placeholders stay as single tokens.
2. **Model** (Milestone 2). `NGramModel` counts every context of up to n-1 words, with `<s>` padding, an end token
   `</s>` and `<unk>` for rare words. Three smoothing techniques give unseen sequences a small probability while keeping
   each distribution summing to 1: **add-k** (Laplace when k=1), **linear interpolation** with lower orders, and
   **interpolated Kneser-Ney** (absolute discounting with continuation counts).
3. **Generation** (Milestone 3). The agent's partial text is anonymized and tokenized like the corpus. The model then
   gives the top-N next words, a greedy continuation (stops at the end of a sentence and blocks repeated phrases), or
   beam search for several distinct, high-probability continuations. Tokens are turned back into readable text with
   proper spacing and capitals.
4. **CLI** (Milestone 4). `app/cli.py`. The model trains from the corpus on each run, in well under a second.

### Results

Held-out evaluation (`python -m app.cli report`, 80/20 split of the 600 agent responses, 7,148 next-word
predictions). Top-k is the share of positions where the real next word is among the model's k best guesses. The
**baseline** is a unigram model (n=1): it ignores the context and always guesses the most frequent words. Full output:
[results/evaluation.md](results/evaluation.md); example suggestions: [results/sample_suggestions.md](results/sample_suggestions.md).

| n | Smoothing | Perplexity | Top-1 | Top-3 | Top-5 |
|---|---|---:|---:|---:|---:|
| 1 | baseline (unigram) | 113.54 | 7.8% | 17.1% | 23.8% |
| 2 | add-k | 4.59 | 51.9% | 77.7% | 86.5% |
| 2 | interpolation | 5.15 | 51.9% | 77.8% | 86.3% |
| 2 | Kneser-Ney | 3.80 | 51.9% | 77.8% | 86.5% |
| 3 | add-k | 2.59 | 79.2% | 95.1% | 98.0% |
| 3 | interpolation | 1.96 | 79.2% | 95.2% | 97.9% |
| 3 | Kneser-Ney | 1.64 | 79.3% | 95.2% | 98.2% |
| 4 | add-k | 2.75 | 84.3% | 95.5% | 97.9% |
| 4 | interpolation | 1.54 | 84.6% | 96.2% | 98.6% |
| 4 | **Kneser-Ney** | **1.44** | **84.9%** | **96.2%** | **98.8%** |

- Every model with context beats the baseline by far: the best one (n=4, Kneser-Ney) has perplexity 1.44 instead of
  113.5, and top-1 accuracy 84.9% instead of 7.8%.
- Kneser-Ney has the lowest perplexity at every order, so it is the default (with n=4).
- Add-k gets *worse* going from n=3 to n=4: with only the full-order context and no backoff, more of its contexts are
  rare, and it spreads their probability over the whole vocabulary. Interpolation and Kneser-Ney fall back on shorter
  contexts and keep improving.

### Using your own tickets

Put them in `corpus/raw_interactions.txt` in the same format (`### Ticket <id> | <intent>`, then `Customer: ...` and
`Agent: ...` lines), and run `python -m app.cli prepare`. Check the prepared files before you commit them: the
anonymizer is rule-based, so a name with no greeting or sign-off cue next to it can slip through.

## Limitations

- **Synthetic data.** The corpus is generated from a limited set of phrasings, so the accuracy above is much higher
  than it would be on real tickets. Expect higher perplexity and lower accuracy on real support conversations.
- **Short memory.** An n-gram model only sees the last n-1 words. It cannot use the customer's message or the topic
  of the ticket, so a continuation can drift to another topic (for example, from a refund to a password reset).
- **Rule-based anonymization.** Emails, phones, cards, order and tracking numbers and addresses are found by pattern.
  Names are found only next to greeting or sign-off cues, so an uncued name can slip through. Check real data by hand.
- **Lower-case output.** Tokens are lower-cased, so proper nouns in suggestions come out lower-case
  ("customer support"). The start of each sentence and "I" are capitalized again.
- **Retrains on every run.** This takes well under a second for this corpus. A much larger corpus would need the
  model to be saved to disk.

## Stretch ideas

- [x] Allow agents to choose from the top N suggested words instead of just the single best one (`next --top N`, and numbers in `interactive`).
- [x] Explore and implement different smoothing techniques to potentially improve prediction accuracy (add-k, interpolation, Kneser-Ney; compare with `evaluate`).
- [x] Add a feature for agents to easily contribute new, effective responses to the corpus, enabling the model to learn and adapt over time (`add`, and `/save` in `interactive`).
