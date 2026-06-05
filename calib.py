"""Shared calibration utilities for the irony-detection notebooks.

Each notebook focuses on a different model or signal source — anthropic api
vs local mlx, different verbalized-confidence scales, single vs dual signal —
but the data loading, prompt shapes, parsing, calibration math, and plotting
are identical. All of that lives here.

The notebooks should mostly contain (a) markdown narrative, (b) the
model-specific setup (client construction, mlx model load, the prompt that
gets called per example), and (c) calls into this module.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


# data ----------------------------------------------------------------------

IRONY_POOL_SIZE = 4601  # all splits of cardiffnlp/tweet_eval irony combined


def load_irony_pool():
    """Load tweet_eval/irony and concatenate train + validation + test.

    Zero-shot eval makes the train/val/test distinction vestigial, so we pool
    everything for a single sampling surface of 4601 examples.

    :returns: A huggingface dataset with `text` and `label` columns.
    """
    from datasets import concatenate_datasets, load_dataset
    splits = load_dataset("cardiffnlp/tweet_eval", "irony")
    return concatenate_datasets([splits["train"], splits["validation"], splits["test"]])


def subsample(ds, n_samples: int, seed: int = 0) -> list[dict]:
    """Random subsample without replacement.

    :param ds: Huggingface dataset (typically from :func:`load_irony_pool`).
    :param n_samples: How many examples to draw (clamped to the dataset size).
    :param seed: RNG seed for reproducibility.
    :returns: List of dicts with `text` and `label` keys.
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(ds))
    idx = rng.choice(len(ds), size=n, replace=False)
    return [{"text": ds[int(i)]["text"], "label": ds[int(i)]["label"]} for i in idx]


# prompts -------------------------------------------------------------------

IRONY_RUBRIC = (
    "A tweet is IRONIC if its intended meaning differs from a literal reading — "
    "this includes sarcasm, verbal irony (saying the opposite of what is meant), "
    "and situational irony (an outcome contrary to expectation). A tweet is NOT "
    "IRONIC if it is meant literally, even if it is negative, funny, or exaggerated."
)


def verbalized_system_prompt(scale: int = 100) -> str:
    """System prompt asking for a json response with label + integer confidence.

    :param scale: Top of the confidence range; the model is told to emit an
        integer between 0 and `scale`. Typically 10 or 100.
    """
    return (
        f"You judge whether tweets are ironic. {IRONY_RUBRIC} "
        f"Respond with ONLY a JSON object and nothing else, in the form "
        f'{{"label": <1 for ironic, 0 for not ironic>, '
        f'"confidence": <integer 0-{scale} = your probability that your label is correct>}}.'
    )


LOGPROB_SYSTEM = (
    f"You judge whether tweets are ironic. {IRONY_RUBRIC} "
    "Respond with EXACTLY ONE CHARACTER: '1' if the tweet is ironic, '0' if it is not. "
    "No other text, no quotes, no explanation."
)


# parsing -------------------------------------------------------------------

_json_re = re.compile(r"\{.*\}", re.DOTALL)


def parse_verbalized_response(body: str, scale: int = 100) -> tuple[int | None, float | None]:
    """Parse a json-shaped model response into (label, normalized_confidence).

    :param body: Raw model response text.
    :param scale: The same scale the model was asked for; divides the integer
        to land in [0, 1].
    :returns: Tuple ``(label, conf)`` with label in {0, 1} and conf in [0, 1],
        or ``(None, None)`` if the response is unparseable or out of range.
    """
    match = _json_re.search(body)
    if not match:
        return None, None
    try:
        obj = json.loads(match.group(0))
        label = int(obj["label"])
        conf = float(obj["confidence"]) / scale
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None, None
    if label not in (0, 1) or not (0.0 <= conf <= 1.0):
        return None, None
    return label, conf


# anthropic helper ----------------------------------------------------------

def make_anthropic_classifier(client, model: str, scale: int = 100, max_retries: int = 3):
    """Build a callable that classifies one text via the anthropic messages api.

    :param client: An `anthropic.Anthropic` instance.
    :param model: Model id string (e.g. ``"claude-haiku-4-5-20251001"``).
    :param scale: Verbalized-confidence top end (10 or 100).
    :param max_retries: Retries on transient api / parse failures, with
        exponential backoff between attempts.
    :returns: A function ``(text) -> (label, conf) | (None, None)`` suitable
        for passing to :func:`run_loop`.
    """
    system = verbalized_system_prompt(scale)

    def classify(text: str) -> tuple[int | None, float | None]:
        for attempt in range(max_retries):
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=40,
                    temperature=0,
                    system=system,
                    messages=[{"role": "user", "content": text}],
                )
                return parse_verbalized_response(resp.content[0].text, scale=scale)
            except Exception:
                # back off briefly on rate limits / transient errors
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return None, None

    return classify


# run loop ------------------------------------------------------------------

def run_loop(
    samples: list[dict],
    classifiers: dict[str, Callable[[str], tuple[int | None, float | None]]],
    csv_path: str,
    progress_every: int = 100,
) -> tuple[list[dict], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Run one or more classifiers over a sample list, write results to csv.

    Only keeps rows where every classifier produced a parseable response, so
    cross-signal comparison stays apples-to-apples.

    :param samples: List of `{text, label}` dicts (from :func:`subsample`).
    :param classifiers: Dict mapping signal name (e.g. ``"verbal"``,
        ``"logprob"``) to a classifier callable.
    :param csv_path: Where to cache the per-row results.
    :param progress_every: Print a progress line every n examples.
    :returns: Tuple ``(results_rows, signals_dict)``. `signals_dict` maps each
        signal name to ``(confidence_array, correct_array)`` for downstream
        metrics and plotting.
    """
    results: list[dict] = []
    t_start = time.time()
    for i, ex in enumerate(samples):
        outputs = {name: fn(ex["text"]) for name, fn in classifiers.items()}
        # apples-to-apples: drop the row entirely if any signal failed to parse
        if any(pred is None for pred, _ in outputs.values()):
            continue
        row = {"text": ex["text"], "true_label": ex["label"]}
        for name, (pred, conf) in outputs.items():
            row[f"pred_{name}"] = pred
            row[f"conf_{name}"] = conf
            row[f"correct_{name}"] = int(pred == ex["label"])
        results.append(row)
        if (i + 1) % progress_every == 0:
            rate = (i + 1) / (time.time() - t_start)
            eta_min = (len(samples) - (i + 1)) / rate / 60
            print(f"{i + 1}/{len(samples)} done  ({rate:.2f} ex/s, eta {eta_min:.1f} min)")

    print(f"\nparsed {len(results)} / {len(samples)} responses "
          f"in {(time.time() - t_start) / 60:.1f} min")

    fieldnames = ["text", "true_label"]
    for name in classifiers:
        fieldnames.extend([f"pred_{name}", f"conf_{name}", f"correct_{name}"])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    signals = {
        name: (
            np.array([r[f"conf_{name}"] for r in results]),
            np.array([r[f"correct_{name}"] for r in results]),
        )
        for name in classifiers
    }
    return results, signals


def _load_results(csv_path: str) -> tuple[list[dict], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Read a cached run-loop csv back into the same `(results, signals)` shape.

    Handles two on-disk schemas: the post-refactor multi-signal layout
    (``pred_X``, ``conf_X``, ``correct_X`` per signal), and the legacy
    single-signal layout (``pred_label``, ``confidence``, ``correct``)
    produced by earlier versions. The legacy file is mapped onto a single
    ``"verbal"`` signal so downstream consumers see a consistent shape.
    """
    results: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        # signal names live in the column prefixes the run loop wrote
        signal_names = [n[len("pred_"):] for n in fieldnames if n.startswith("pred_")]
        # legacy: pred_label / confidence / correct → one implicit "verbal" signal
        legacy = signal_names == ["label"] and {"confidence", "correct"}.issubset(fieldnames)
        if legacy:
            signal_names = ["verbal"]
        for raw in reader:
            row: dict = {"text": raw["text"], "true_label": int(raw["true_label"])}
            if legacy:
                row["pred_verbal"] = int(raw["pred_label"])
                row["conf_verbal"] = float(raw["confidence"])
                row["correct_verbal"] = int(raw["correct"])
            else:
                for name in signal_names:
                    row[f"pred_{name}"] = int(raw[f"pred_{name}"])
                    row[f"conf_{name}"] = float(raw[f"conf_{name}"])
                    row[f"correct_{name}"] = int(raw[f"correct_{name}"])
            results.append(row)
    signals = {
        name: (
            np.array([r[f"conf_{name}"] for r in results]),
            np.array([r[f"correct_{name}"] for r in results]),
        )
        for name in signal_names
    }
    return results, signals


def run_or_load(
    samples: list[dict],
    classifiers: dict[str, Callable[[str], tuple[int | None, float | None]]],
    csv_path: str,
    progress_every: int = 100,
) -> tuple[list[dict], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Skip inference if a cached results csv already exists.

    If ``csv_path`` exists, load it and return immediately — `samples` and
    `classifiers` are ignored on the cache hit path, since signal names come
    from the CSV header. Otherwise call :func:`run_loop` to query the model
    and write the csv.

    To force a fresh run, delete the csv first.

    :param samples: List of `{text, label}` dicts (forwarded to :func:`run_loop`).
    :param classifiers: Dict mapping signal name to classifier callable
        (forwarded to :func:`run_loop`).
    :param csv_path: Cache file path. Existence triggers the load path.
    :param progress_every: Forwarded to :func:`run_loop`.
    :returns: Tuple ``(results_rows, signals_dict)`` matching the
        :func:`run_loop` return shape.
    """
    if os.path.exists(csv_path):
        print(f"loading cached results from {csv_path}")
        results, signals = _load_results(csv_path)
        print(f"loaded {len(results)} rows with signals: {list(signals.keys())}")
        return results, signals
    print(f"no cache at {csv_path} — running inference")
    return run_loop(samples, classifiers, csv_path, progress_every)


# metrics -------------------------------------------------------------------

def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 30,
    strategy: str = "range",
) -> tuple[float, dict]:
    """Compute expected calibration error.

    :param confidences: Stated probability the prediction is correct, in [0, 1].
    :param correct: Array of 0/1 flags; 1 where the prediction matched the label.
    :param n_bins: Target number of bins.
    :param strategy: Binning scheme. One of:

        - ``"range"`` (default) — N equal-width bins between min(conf) and
          max(conf). Bins are uniform width and concentrated where data lives.
        - ``"quantile"`` — N equal-count bins (dedupes ties).
        - ``"uniform"`` — N equal-width bins over the full [0, 1] interval.
    :returns: Tuple ``(ece, stats)`` where stats holds per-bin edges, mean
        confidence, accuracy, and count for plotting.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "range":
        lo, hi = float(confidences.min()), float(confidences.max())
        # degenerate case: all confidences identical
        if hi == lo:
            edges = np.array([lo, lo + 1e-12])
        else:
            edges = np.linspace(lo, hi, n_bins + 1)
    elif strategy == "quantile":
        # equal-count edges; dedupe to avoid zero-width bins on ties
        edges = np.unique(np.quantile(confidences, np.linspace(0.0, 1.0, n_bins + 1)))
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    n_actual = len(edges) - 1
    # clip so confidences equal to the upper edge land in the last bin
    idx = np.clip(np.digitize(confidences, edges) - 1, 0, n_actual - 1)

    bin_conf, bin_acc, bin_count = [], [], []
    ece, total = 0.0, len(confidences)
    for b in range(n_actual):
        mask = idx == b
        count = int(mask.sum())
        bin_count.append(count)
        if count == 0:
            bin_conf.append(np.nan)
            bin_acc.append(np.nan)
            continue
        mean_conf = confidences[mask].mean()
        acc = correct[mask].mean()
        bin_conf.append(mean_conf)
        bin_acc.append(acc)
        ece += (count / total) * abs(mean_conf - acc)
    return ece, {
        "edges": edges,
        "mean_confidence": np.array(bin_conf),
        "accuracy": np.array(bin_acc),
        "count": np.array(bin_count),
    }


def wilson_interval(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    More honest than the normal approximation at small n or extreme p, which
    is exactly the regime each calibration bin sits in.

    :param k: Number of successes.
    :param n: Number of trials.
    :param z: Z-score for the confidence level (1.96 ≈ 95%).
    :returns: Tuple ``(lower, upper)``; ``(nan, nan)`` if n == 0.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - half, center + half


def auroc(conf: np.ndarray, correct: np.ndarray) -> float:
    """AUROC of confidence vs correctness; nan if only one correctness class present."""
    if len(np.unique(correct)) < 2:
        return float("nan")
    return roc_auc_score(correct, conf)


def compute_metrics(
    signals: dict[str, tuple[np.ndarray, np.ndarray]],
    n_bins: int = 30,
    strategy: str = "range",
) -> dict[str, dict]:
    """Compute accuracy / ECE / AUROC / per-bin stats for each named signal.

    :param signals: Dict of ``{name: (confidences, correct)}``.
    :param n_bins: Forwarded to :func:`expected_calibration_error`.
    :param strategy: Forwarded to :func:`expected_calibration_error`.
    :returns: Dict mapping signal name to a dict with keys `conf`, `correct`,
        `accuracy`, `ece`, `auroc`, `stats`.
    """
    out = {}
    for name, (conf, correct) in signals.items():
        ece, stats = expected_calibration_error(conf, correct, n_bins=n_bins, strategy=strategy)
        out[name] = {
            "conf": conf,
            "correct": correct,
            "accuracy": float(correct.mean()),
            "ece": ece,
            "auroc": auroc(conf, correct),
            "stats": stats,
        }
    return out


def print_metrics(metrics: dict[str, dict]) -> None:
    """Print accuracy / ECE / AUROC in a comparable column layout.

    Works for one signal or many; columns auto-size to fit the longest name.
    """
    names = list(metrics.keys())
    col_width = max(12, max(len(n) for n in names) + 2)
    header = "".join(f"{n:>{col_width}}" for n in names)
    print(f"{'':14}{header}")
    for label, key in [("accuracy:", "accuracy"), ("ECE:", "ece"), ("AUROC:", "auroc")]:
        row = "".join(f"{metrics[n][key]:>{col_width}.3f}" for n in names)
        print(f"{label:14}{row}")
    print()
    for name in names:
        edges = metrics[name]["stats"]["edges"]
        print(f"{name + ' bins:':24} {len(edges) - 1} equal-width over "
              f"[{edges[0]:.2f}, {edges[-1]:.2f}]")


# diagnostics ---------------------------------------------------------------

def value_counts(conf: np.ndarray, scale: int = 100, label: str = "verbal") -> None:
    """Print discrete value distribution for a finite-scale verbalized signal."""
    ints = np.round(conf * scale).astype(int)
    uniq, cnt = np.unique(ints, return_counts=True)
    print(f"{label}: {len(uniq)} unique stated confidence values")
    for c, n in zip(uniq, cnt):
        print(f"  {c:3d}: {n:5d}")


def signal_summary(conf: np.ndarray, label: str) -> None:
    """One-line min / median / max + unique-count summary for a continuous signal."""
    print(f"{label}: min={conf.min():.3f}  median={np.median(conf):.3f}  "
          f"max={conf.max():.3f}  "
          f"({len(np.unique(np.round(conf, 4)))} unique values rounded to 4dp)")


# plotting ------------------------------------------------------------------

def plot_histograms(
    metrics: dict[str, dict],
    figsize: tuple[float, float] | None = None,
    fallback_width: float = 0.02,
) -> None:
    """Stacked histogram subplot per named signal, each fit to its own range.

    If a signal's binned range collapsed to a single tiny-width bin (constant
    model output), the bar is inflated to `fallback_width` so it stays visible
    and annotated.

    :param metrics: Output of :func:`compute_metrics`.
    :param figsize: Tuple ``(width, height)``; defaults to one row per signal.
    :param fallback_width: Bar width used for the degenerate single-bin case.
    """
    names = list(metrics.keys())
    n = len(names)
    if figsize is None:
        figsize = (8.5, 4.0 * n + 0.5)
    fig, axes = plt.subplots(n, 1, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        stats = metrics[name]["stats"]
        edges = stats["edges"]
        counts = stats["count"]
        widths = np.diff(edges)
        # detect the degenerate single-bin case and inflate the bar width
        if len(widths) == 1 and widths[0] < 1e-3:
            center = edges[0]
            ax.bar([center - fallback_width / 2], counts, width=fallback_width,
                   edgecolor="white")
            ax.annotate(
                f"all {int(counts[0])} examples at confidence = {center:.2f}\n"
                "(model emits a single value — no introspective range)",
                xy=(center, counts[0]), xytext=(0.5, 0.95), textcoords="axes fraction",
                ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray"),
            )
            pad = max(0.05, fallback_width)
            ax.set_xlim(max(0.0, center - pad), min(1.0, center + pad))
        else:
            ax.bar(edges[:-1], counts, width=widths, align="edge", edgecolor="white")
            ax.set_xlim(max(0.0, edges[0] - 0.02), min(1.0, edges[-1] + 0.02))
        ax.set_ylabel("count")
        ax.set_xlabel("stated confidence")
        ax.set_title(f"{name} confidence histogram")

    plt.tight_layout()
    plt.show()


def plot_reliability(
    metrics: dict[str, dict],
    figsize: tuple[float, float] = (6.5, 6.5),
    title: str | None = None,
) -> None:
    """Reliability diagram, one or more signals overlaid on a single square axes.

    Each signal is drawn as (mean_confidence, accuracy) points connected by a
    line, with 95% wilson intervals as error bars in light gray. The 45° line
    shows perfect calibration; points below it are overconfidence.

    :param metrics: Output of :func:`compute_metrics`.
    :param figsize: Tuple ``(width, height)``; kept square so the 45° line is
        meaningful.
    :param title: Optional override; otherwise auto-built from signal names.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")

    all_x_lo: list[float] = []
    all_x_hi: list[float] = []
    all_y_hi: list[float] = []
    fmts = ["o-", "s-", "^-", "D-", "v-"]

    for i, (name, m) in enumerate(metrics.items()):
        stats = m["stats"]
        acc = stats["accuracy"]
        counts = stats["count"]
        mean_conf = stats["mean_confidence"]
        lowers = np.full_like(acc, np.nan)
        uppers = np.full_like(acc, np.nan)
        for j, (a, c) in enumerate(zip(acc, counts)):
            if c == 0 or np.isnan(a):
                continue
            lowers[j], uppers[j] = wilson_interval(a * c, c)
        valid = ~np.isnan(acc)
        if not valid.any():
            continue
        yerr = np.vstack([acc - lowers, uppers - acc])
        label = f"{name} (ECE = {m['ece']:.3f}, AUROC = {m['auroc']:.3f})"
        ax.errorbar(
            mean_conf[valid], acc[valid], yerr=yerr[:, valid],
            fmt=fmts[i % len(fmts)], capsize=3, ecolor="lightgray",
            color=f"C{i}", label=label,
        )
        all_x_lo.extend([np.nanmin(mean_conf[valid]), np.nanmin(lowers[valid])])
        all_x_hi.extend([np.nanmax(mean_conf[valid]), np.nanmax(uppers[valid])])
        all_y_hi.append(np.nanmax(uppers[valid]))

    # zoom to a shared range covering all signals' data + cis, with small padding;
    # keeping axes equal preserves the meaning of the 45° line
    axis_lo = max(0.0, min(all_x_lo) - 0.02) if all_x_lo else 0.0
    axis_hi = min(1.0, max(*all_x_hi, *all_y_hi, 1.0) + 0.01) if all_x_hi else 1.0
    ax.set_xlabel("mean stated confidence (bin)")
    ax.set_ylabel("empirical accuracy (bin)")
    if title is None:
        title = "Reliability diagram"
        if len(metrics) > 1:
            title += f" — {' vs '.join(metrics.keys())}"
    ax.set_title(title)
    ax.set_xlim(axis_lo, axis_hi)
    ax.set_ylim(axis_lo, axis_hi)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
