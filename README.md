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
- **Logprob** — for the local-model notebook only — ask for a single character (`0` or
  `1`), run one forward pass, and read the softmax probability mass on those two
  tokens directly off the final logits. Confidence = `max(P("0"), P("1")) / (P("0") +
  P("1"))`.

Same calibration treatment runs on both signals downstream.

## Notebooks

| Notebook | Model | Confidence scale | Notes |
|---|---|---|---|
| [`llm_conf_haiku_100.ipynb`](llm_conf_haiku_100.ipynb) | Claude Haiku 4.5 | 0–100 verbal | original baseline; introduces the equal-width-over-`[min,max]` binning and Wilson error bars |
| [`llm_conf_haiku_10.ipynb`](llm_conf_haiku_10.ipynb) | Claude Haiku 4.5 | 0–10 verbal | does coarsening the scale change how the model distributes its self-assessment? |
| [`llm_conf_sonnet_100.ipynb`](llm_conf_sonnet_100.ipynb) | Claude Sonnet 4.6 | 0–100 verbal | stronger-model comparison on the same task |
| [`llm_conf_qwen_100.ipynb`](llm_conf_qwen_100.ipynb) | local (Llama 3.1 8B via MLX) | 0–100 verbal **and** logprobs | dual-signal — same model, same examples, both signals side-by-side. file is named after the original local-model attempt (Qwen3-30B-A3B); see "switching back" below |

Result CSVs (`irony_<model>_<scale>.csv`) are written by each notebook and cached, so
you can re-render plots without re-running inference.

## Findings so far

- **Verbalized confidence clusters at a few round numbers.** Haiku 4.5 emitted 9
  distinct values across 4,596 parsed responses; Sonnet 4.6 emitted 23. Equal-width
  binning over `[0, 1]` wastes most of its bins on regions with no data — the
  notebooks bin equal-width over `[min(conf), max(conf)]` instead, with quantile and
  uniform binning available as alternatives.
- **Stronger models are better calibrated and more discriminative, but not by much.**
  Haiku 4.5: accuracy 0.756, ECE 0.094, AUROC 0.655. Sonnet 4.6: accuracy 0.791, ECE
  0.038, AUROC 0.694. Both are overconfident at the top of their range; Sonnet less
  so.
- **Verbalized confidence can collapse to a constant.** The first local-model run
  (Qwen3-30B-A3B-Instruct at 4-bit) emitted `95` as its verbalized confidence on every
  single one of 500 examples — verbal AUROC = 0.500 by construction (a constant can't
  rank anything). On the *same* model and the *same* examples, the logprob signal
  reached AUROC 0.648. The verbalized number is a model artifact; what's actually in
  the model's head lives in the token logits.
- **Quantization on a 30B MoE bit hard.** Qwen3-30B-A3B at Q4 scored 0.548 accuracy
  on irony — barely above the 0.49 base rate. Switching to dense Llama 3.1 8B
  Instruct at the same quantization is the current direction.

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
- Verbalized and logprob confidence are different signals with different failure
  modes. The local-model notebook compares them directly; the API notebooks have
  access to only the verbalized one.
