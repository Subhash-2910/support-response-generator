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
- [ ] **4. Build a command-line interface for agents to get suggestions.**  
      Hint: Create a user-friendly command-line interface to input partial responses and display generated suggestions.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow agents to choose from the top N suggested words instead of just the single best one.
- Explore and implement different smoothing techniques to potentially improve prediction accuracy.
- Add a feature for agents to easily contribute new, effective responses to the corpus, enabling the model to learn and adapt over time.
