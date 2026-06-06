# LLM confidence calibration on tweet irony detection

A small set of notebooks measuring how well an LLM's self-reported confidence tracks
its actual empirical accuracy on a binary classification task. Each notebook runs the
same task end-to-end (classification → ECE / AUROC / reliability diagram / confidence
histogram) with a different model or signal source, so they're comparable in shape.

## The task

[`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval), config
`irony` (the SemEval-2018 Task 3 split). 4,601 tweets total across train/val/test —
pooled, since the model is evaluated zero-shot. Label `1` = ironic (verbal, situational,
or sarcasm), label `0` = literal.

Irony was chosen after a CoLA pilot run showed grammaticality is too easy — when a
model is right ~86% of the time, almost all stated confidences collapse near 1.0 and
the calibration picture degenerates. Irony is hard enough to give a real spread of
correct and wrong predictions, even for strong models.

## Two confidence signals

- **Verbalized** — ask the model for a JSON object with `{"label": 0|1, "confidence":
  0-100}`. The only signal available from a closed API like Anthropic's.
- **Probability** — for the local-model notebook only — ask for a single character
  (`0` or `1`), run one forward pass, and read the softmax probability mass on those
  two tokens directly off the final logits. Confidence = `max(P("0"), P("1")) /
  (P("0") + P("1"))`.

Same calibration treatment runs on both signals downstream.

## Notebooks

| Notebook | Model | Confidence scale | Accuracy | ECE | AUROC |
|---|---|---|---:|---:|---:|
| [`irony_haiku_100.ipynb`](irony_haiku_100.ipynb) | Claude Haiku 4.5 | 0–100 verbal | 0.756 | 0.094 | 0.655 |
| [`irony_haiku_10.ipynb`](irony_haiku_10.ipynb) | Claude Haiku 4.5 | 0–10 verbal | 0.754 | 0.082 | 0.634 |
| [`irony_sonnet_100.ipynb`](irony_sonnet_100.ipynb) | Claude Sonnet 4.6 | 0–100 verbal | 0.791 | 0.038 | 0.694 |
| [`irony_opus_100.ipynb`](irony_opus_100.ipynb) | Claude Opus 4.8 | 0–100 verbal | 0.791 | 0.052 | 0.705 |
| [`irony_llama_100.ipynb`](irony_llama_100.ipynb) | local Llama 3.1 8B via MLX (verbal) | 0–100 verbal | 0.642 | 0.159 | 0.526 |
| [`irony_llama_100.ipynb`](irony_llama_100.ipynb) | local Llama 3.1 8B via MLX (prob) | softmax over `{"0","1"}` | 0.639 | 0.104 | 0.634 |

The local notebook produces both signals on the same examples in one pass through the
data, so the verbal-vs-prob comparison is apples-to-apples.

Result CSVs (`data/irony_<model>_<scale>.csv`) are written by each notebook and cached,
so re-running a notebook with the cache in place skips inference and just renders the
plots. Delete the CSV to force a fresh run.

Shared logic — data loading, prompt templates, the JSON-response parser, the run loop,
ECE / Wilson-interval / AUROC math, and the histogram + reliability plots — lives in
[`calib.py`](calib.py). The notebooks contain the per-model configuration, narrative,
and result rendering.

## Findings

- **Verbalized confidence clusters at a few round numbers, on every model tested.**
  Haiku 4.5 on the 0–100 scale used 9 of the 100 possible values; on the 0–10 scale
  it used 7 of 11. Sonnet 4.6 used 23 of 100; Opus 4.8 used 18. Llama 3.1 8B used 6
  of 100 — and one value (80) accounted for 94% of all responses. Equal-width binning
  over `[0, 1]` wastes most of its bins on regions with no data; the notebooks bin
  equal-width over `[min(conf), max(conf)]` instead, with quantile and uniform
  binning available as alternatives.
- **Coarsening the scale doesn't fix the clustering.** Haiku 0–100 → 0–10 produced
  essentially the same headline numbers (accuracy 0.756 → 0.754, ECE 0.094 → 0.082,
  AUROC 0.655 → 0.634). The model still concentrates its responses on a small subset
  of the available values; reducing the alphabet from 100 to 11 just shrinks the
  alphabet, not the model's habit of picking favorites.
- **Going up the model tier helps non-monotonically.** Sonnet 4.6 vs Haiku 4.5 (both
  0–100) is the clear lift: accuracy 0.791 vs 0.756, ECE 0.038 vs 0.094, AUROC 0.694
  vs 0.655 — Sonnet wins on every axis and its calibration curve sits closer to the
  45° line throughout its populated range. Opus 4.8 vs Sonnet 4.6 doesn't continue
  the trend: accuracy stays at 0.791, AUROC nudges up to 0.705, but ECE *worsens* to
  0.052 (Sonnet is the better-calibrated model here). Opus also produced 158
  unparseable responses out of 4601, vs ~5 for haiku/sonnet — likely a side-effect
  of its higher default verbosity blowing past `max_tokens=40` before closing the
  JSON object.
- **Verbalized confidence is much weaker than token probabilities on the open model.**
  Llama 3.1 8B's verbalized AUROC (0.526) is barely above chance — the 94%-at-80
  collapse means the signal can't rank correct above incorrect predictions. The
  probability signal on the same model, same examples reaches AUROC 0.634 (comparable
  to Haiku's verbalized AUROC) and a notably better ECE (0.104 vs 0.159). The
  verbalized number on this model is a token-emission artifact; what's actually in
  the model's head lives in the next-token softmax, and it's a real signal worth
  using.

## Setup

```sh
pip install datasets anthropic mlx-lm scikit-learn numpy matplotlib
export ANTHROPIC_API_KEY=...   # for the Haiku and Sonnet notebooks
```

The local-model notebook additionally needs you to be authenticated against the Llama
3.1 repo on HuggingFace (it's a gated model — accept the license on the model page,
then `huggingface-cli login` or set `HF_TOKEN`).

## Caveats

- Measures calibration on `tweet_eval/irony` only — short, English-language,
  tweet-style text with a particular annotation guideline. Does not transport to your
  real input distribution; recalibrate on data that looks like your inputs.
- Irony labels are subjective. Some apparent miscalibration is label noise rather than
  model fault — inspect a sample of model-vs-label disagreements before drawing strong
  conclusions.
- Results are model- and version-specific. Re-run when `MODEL` changes.
- Verbalized and probability-based confidence are different signals with different
  failure modes. The local-model notebook compares them directly; the API notebooks
  have access to only the verbalized one.
