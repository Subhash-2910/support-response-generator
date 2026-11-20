# Evaluation

Corpus: 600 anonymized agent responses, 35761 tokens, vocabulary 298. Held-out 80/20 split (seed 0): 7148 next-word predictions.

n=1 is the **baseline**: it ignores the context and always guesses the most frequent words.

| n | Smoothing | Perplexity | Top-1 | Top-3 | Top-5 |
|---|---|---:|---:|---:|---:|
| 1 (baseline) | add-k | 113.54 | 7.8% | 17.1% | 23.8% |
| 1 (baseline) | interpolation | 113.54 | 7.8% | 17.1% | 23.8% |
| 1 (baseline) | Kneser-Ney | 113.54 | 7.8% | 17.1% | 23.8% |
| 2 | add-k | 4.59 | 51.9% | 77.7% | 86.5% |
| 2 | interpolation | 5.15 | 51.9% | 77.8% | 86.3% |
| 2 | Kneser-Ney | 3.80 | 51.9% | 77.8% | 86.5% |
| 3 | add-k | 2.59 | 79.2% | 95.1% | 98.0% |
| 3 | interpolation | 1.96 | 79.2% | 95.2% | 97.9% |
| 3 | Kneser-Ney | 1.64 | 79.3% | 95.2% | 98.2% |
| 4 | add-k | 2.75 | 84.3% | 95.5% | 97.9% |
| 4 | interpolation | 1.54 | 84.6% | 96.2% | 98.6% |
| 4 | Kneser-Ney | 1.44 | 84.9% | 96.2% | 98.8% |

Best: n=4 with Kneser-Ney smoothing: perplexity 1.44 vs 113.54 for the baseline, top-1 accuracy 84.9% vs 7.8%.
