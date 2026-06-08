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

## Three confidence signals

- **Verbalized label + confidence** — ask the model for a JSON object with
  `{"label": 0|1, "confidence": 0-N}`. Two free parameters per example. The original
  elicitation, available from any chat-completion API.
- **Verbalized P(ironic)** — ask the model for a single integer in `[0, 100]`
  representing `P(text is ironic)`. The implied label is `1 if P >= 0.5 else 0`,
  confidence-in-chosen-label is `max(P, 1-P)`. One number per example. Also available
  from any chat-completion API, just a different prompt.
- **Token-softmax probability** — for the local-model notebooks only — ask for a
  single character (`0` or `1`), run one forward pass, and read the softmax probability
  mass on those two tokens directly off the final logits. Confidence in chosen label =
  `max(P("0"), P("1")) / (P("0") + P("1"))`. Cannot be obtained from a closed API.

All three signals get the same calibration treatment downstream — same ECE, reliability
diagram, AUROC, histogram code.

## ECE

**Expected Calibration Error** asks a simpler, more literal question than AUROC does:
"when the model says it's 80% confident, is it actually right about 80% of the time?"

To measure this, bucket the examples by their stated confidence (e.g. all the times
the model said "70-80% confident" go in one bucket), then within each bucket compare
two numbers: the average confidence the model claimed, and the fraction of examples in
that bucket it actually got right. A perfectly calibrated model has those two numbers
match in every bucket — its 80%-confidence bucket really is right 80% of the time, its
60%-confidence bucket really is right 60% of the time, and so on. ECE is the average
gap between claimed-confidence and actual-accuracy across all the buckets, weighted by
how many examples fall in each one. So ECE is in the same units as confidence and
accuracy — `0.0` is perfect calibration, and `0.094` (Haiku's JSON-prompt run, say)
means the model's stated confidence is off by about 9.4 percentage points from its
real accuracy, on average across the confidence range it actually used.

Note what ECE does *not* care about: it has no opinion on whether the model gets the
underlying classification right, and — unlike calibration AUROC — no opinion on
whether confidence successfully separates correct from incorrect predictions either.
A model that says "60% confident" on every single example, and is in fact right 60%
of the time overall, scores a perfect ECE of `0.0` despite that confidence number
carrying zero example-by-example information (calibration AUROC would correctly call
this out as useless, at `0.5`). ECE and calibration AUROC are answering related but
distinct questions — "are the numbers honest on average?" vs. "do the numbers
discriminate between right and wrong?" — and a model can do well on one while doing
poorly on the other.

## Two AUROC flavors

A quick refresher on what AUROC measures, since both flavors below build on it: give
AUROC a pile of examples, each with a numeric score and a yes/no label, and it asks
"if I picked one yes-example and one no-example at random, how often does the
yes-example get the higher score?" That fraction is the AUROC. 1.0 means the score
perfectly separates the two groups (every yes outscores every no); 0.5 means the score
is no better than a coin flip; under 0.5 means the score is ranking things backwards.
Crucially, AUROC only cares about *relative ranking*, not the actual score values — so
it's a clean way to ask "does this number contain useful signal about that label?"
without worrying about whether the number is calibrated to mean anything in absolute
terms.

The two flavors below feed AUROC completely different scores and labels — they're
asking two different yes/no questions about the same run:

- **Calibration AUROC** — `AUROC(scores=conf, labels=correct)`. The yes/no label here
  is "did the model get this example right?", and the score is the model's own stated
  confidence. So this AUROC asks: *across all the examples, does the model's confidence
  tend to be higher on the ones it gets right than on the ones it gets wrong?* A model
  that "knows what it doesn't know" — that hedges on hard examples and commits on easy
  ones — scores high here, regardless of how good it is at the underlying task. This is
  the standard quantity for evaluating calibration / selective prediction (i.e. "should
  I trust this prediction enough to act on it, or send it for human review?").
- **Classifier AUROC** — `AUROC(scores=P(class=1), labels=true_label)`. Forget
  confidence and correctness entirely — this one throws away the model's self-assessment
  and re-scores each example as a plain binary classifier would: `P(class=1) = conf if
  pred == 1 else 1 - conf` turns "I said ironic with 80% confidence" into `P(ironic) =
  0.8` and "I said literal with 80% confidence" into `P(ironic) = 0.2`, on the same
  0–1 scale regardless of which label the model picked. The yes/no label is now "is
  this tweet actually ironic?", so the question becomes: *does this reconstructed
  probability rank the truly-ironic tweets above the truly-literal ones?* This is the
  number you'd put next to any other irony classifier's AUROC for an apples-to-apples
  comparison — it has nothing to do with whether the model can introspect on its own
  correctness.

Classifier AUROC is uniformly higher than calibration AUROC across all runs (roughly
+0.15). That gap makes sense once you see the two questions side by side: "is this
tweet ironic?" is the task the model was tuned and prompted for, while "was *I* right
just now?" is a harder, second-order question the model was never directly optimized
to answer. A model can be a strong classifier — good at the first question — while
still being a mediocre judge of its own correctness — weak at the second.

## Notebooks

| Notebook | Model | Elicitation | Accuracy | ECE | Calibration AUROC | Classifier AUROC |
|---|---|---|---:|---:|---:|---:|
| [`irony_haiku_100.ipynb`](irony_haiku_100.ipynb)   | Claude Haiku 4.5  | JSON label + conf (0–100)  | 0.756 | 0.094 | 0.655 | 0.813 |
| [`irony_haiku_10.ipynb`](irony_haiku_10.ipynb)     | Claude Haiku 4.5  | JSON label + conf (0–10)   | 0.755 | 0.080 | 0.632 | 0.802 |
| [`irony_haiku_prob.ipynb`](irony_haiku_prob.ipynb) | Claude Haiku 4.5  | direct P(ironic)           | 0.769 | 0.079 | 0.648 | 0.826 |
| [`irony_sonnet_100.ipynb`](irony_sonnet_100.ipynb) | Claude Sonnet 4.6 | JSON label + conf (0–100)  | 0.791 | 0.038 | 0.694 | 0.851 |
| [`irony_sonnet_prob.ipynb`](irony_sonnet_prob.ipynb) | Claude Sonnet 4.6 | direct P(ironic)         | 0.804 | 0.028 | 0.724 | 0.877 |
| [`irony_opus_100.ipynb`](irony_opus_100.ipynb)     | Claude Opus 4.8   | JSON label + conf (0–100)  | 0.791 | 0.052 | 0.705 | 0.882 |
| [`irony_opus_prob.ipynb`](irony_opus_prob.ipynb)   | Claude Opus 4.8   | direct P(ironic)           | 0.799 | 0.035 | 0.764 | 0.885 |
| [`irony_llama_100.ipynb`](irony_llama_100.ipynb)   | local Llama 3.1 8B via MLX | JSON label + conf (0–100) | 0.642 | 0.159 | 0.526 | 0.663 |
| [`irony_llama_100.ipynb`](irony_llama_100.ipynb)   | local Llama 3.1 8B via MLX | token-softmax over `{"0","1"}` | 0.639 | 0.104 | 0.634 | 0.718 |
| [`irony_llama_prob.ipynb`](irony_llama_prob.ipynb) | local Llama 3.1 8B via MLX | direct P(ironic) | 0.537 | 0.279 | 0.548 | 0.558 |
| [`irony_llama_prob.ipynb`](irony_llama_prob.ipynb) | local Llama 3.1 8B via MLX | token-softmax over `{"0","1"}` | 0.639 | 0.104 | 0.634 | 0.718 |

The two local-model notebooks each produce two signals on the same examples in one pass
through the data, so the elicitation comparisons within each are apples-to-apples. The
token-softmax results match across the two llama notebooks because the underlying prompt
and decoding for that signal are identical — the only thing that differs is which
verbal-prompt sibling it's paired with.

Result CSVs (`data/irony_<model>_<scale>.csv`) are written by each notebook and cached,
so re-running a notebook with the cache in place skips inference and just renders the
plots. Delete the CSV to force a fresh run.

Shared logic — data loading, prompt templates, the JSON/integer response parsers, the
run loop, ECE / Wilson-interval / AUROC math, and the histogram + reliability plots —
lives in [`calib.py`](calib.py). The notebooks contain the per-model configuration,
narrative, and result rendering.

## Findings

- **Asking for `P(ironic)` directly beats asking for a label + confidence, on every
  Anthropic model tested.** Haiku 4.5: accuracy 0.756 → 0.769, ECE 0.094 → 0.079,
  classifier AUROC 0.813 → 0.826. Sonnet 4.6: 0.791 → 0.804, 0.038 → 0.028, 0.851 →
  0.877. Opus 4.8: 0.791 → 0.799, 0.052 → 0.035, 0.882 → 0.885, with calibration AUROC
  jumping 0.705 → 0.764 — the biggest swing of any pairing. The verbal-label-and-conf
  prompt asks the model to commit to an answer and then rate its certainty, which
  conflates the answer with self-assessment; the direct-probability prompt asks the
  raw question a calibrated model should be tracking frequencies of. Opus also went
  from 158 parse failures (out of 4,601) on the JSON prompt down to 1 on the integer
  prompt, because the response no longer has to close a JSON object before hitting
  `max_tokens`.
- **Verbalized confidence clusters at a few round numbers, on every model tested.**
  Haiku 4.5 on the 0–100 scale used 9 of the 100 possible values; on the 0–10 scale
  it used 7 of 11. Sonnet 4.6 used 23 of 100; Opus 4.8 used 18. Llama 3.1 8B used 6
  of 100 on the JSON prompt and 7 of 100 on the direct-P(ironic) prompt — with one
  value accounting for the vast majority of responses in both cases. Equal-width
  binning over `[0, 1]` wastes most of its bins on regions with no data; the
  notebooks bin equal-width over `[min(conf), max(conf)]` instead, with quantile and
  uniform binning available as alternatives.
- **Coarsening the scale doesn't fix the clustering.** Haiku 0–100 → 0–10 produced
  essentially the same headline numbers (accuracy 0.756 → 0.755, ECE 0.094 → 0.080,
  classifier AUROC 0.813 → 0.802). The model still concentrates its responses on a
  small subset of the available values; reducing the alphabet from 100 to 11 just
  shrinks the alphabet, not the model's habit of picking favorites.
- **Going up the model tier helps non-monotonically on the JSON prompt, monotonically
  on the direct-prob prompt.** With the JSON prompt: Sonnet 4.6 vs Haiku 4.5 is a
  clear lift, then Opus 4.8 vs Sonnet 4.6 keeps accuracy flat at 0.791, nudges
  calibration AUROC up but ECE *worsens* (0.038 → 0.052). With the direct-prob prompt
  the picture is cleaner: classifier AUROC 0.826 (Haiku) → 0.877 (Sonnet) → 0.885
  (Opus), and ECE 0.079 → 0.028 → 0.035. The elicitation matters enough to flip the
  Sonnet-vs-Opus ordering on calibration.
- **Prompting wins on Anthropic models don't transfer to Llama — and the direct-
  P(ironic) prompt actively breaks it.** The same elicitation that gave Anthropic
  models their best run produces Llama's *worst* run: accuracy 0.537 (barely above
  the 0.481 base rate), ECE 0.279, classifier AUROC 0.558. Inspecting the response
  distribution makes the failure mode obvious: Llama emitted "80" for **90.3% of
  examples**, "0" for 7.8%, and ~nothing else worth mentioning. Under that response
  pattern the model is implicitly calling 91% of tweets ironic when only 48% are,
  which is exactly where the accuracy hit comes from. The JSON variant of the same
  model also concentrated on 80, but there 80 was a *confidence level* attached to
  a separately-emitted label, so the label distribution stayed sensible (the verbal
  signal was useless, but predictions were still decent). Collapsing label and
  confidence into one number removed the model's escape hatch — it had nowhere to
  put the label information separately, and the label decision got pulled into the
  same favored-number attractor.
- **Verbalized confidence is much weaker than token probabilities on the open
  model.** Across all three of Llama's verbal-style runs, calibration AUROC is at
  most 0.548 — within rounding of chance. The token-softmax signal on the same
  model, same examples reaches calibration AUROC 0.634 (comparable to Haiku's
  verbalized AUROC) and a notably better ECE (0.104 vs 0.159 or 0.279). The
  verbalized numbers on this model are mostly token-emission artifacts; what's
  actually in the model's head lives in the next-token softmax — and on this kind
  of model that's the only signal worth using.

## Setup

```sh
pip install datasets anthropic mlx-lm scikit-learn numpy matplotlib
export ANTHROPIC_API_KEY=...   # for the Haiku, Sonnet, and Opus notebooks
```

The local-model notebooks additionally need you to be authenticated against the Llama
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
- All three elicitations are different signals with different failure modes. The
  local-model notebooks compare verbal vs token-softmax directly; the API notebooks
  have access to verbal-style signals only and compare elicitations across notebook
  pairs.
