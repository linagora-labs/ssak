"""Compute distributions of audio durations and text lengths across NeMo manifests.

Accepts the same kinds of inputs as the other tools in this folder: JSONL files,
directories (optionally recursive), or YAML training configs (resolved via
`resolve_manifest_paths`). Produces a global summary plus per-manifest detail,
written to a text report and (optionally) histogram plots.

Designed to run fast over thousands of manifests:
- Each manifest is parsed in a worker process (ProcessPoolExecutor).
- Per-row data is reduced into compact NumPy arrays inside the worker so only
  the arrays (not the parsed JSON) cross the process boundary.
- JSON is parsed directly (not via NemoDatasetRow) to skip dataclass overhead.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ssak.utils.nemo_dataset import resolve_manifest_paths

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PERCENTILES = (1, 5, 25, 50, 75, 90, 95, 99)


def _to_float(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def collect_from_manifest(manifest_path):
    """Parse a manifest and return (path, n_rows, durations, text_lengths).

    - durations: per-row total audio duration (float32 seconds). For multiturn
      rows, durations of all audio turns are summed.
    - text_lengths: characters per text item (int32). For multiturn rows, every
      text turn contributes one entry; for ASR rows, the `text` field does.
    """
    durations = []
    text_lengths = []
    n_rows = 0
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_rows += 1
                if "conversations" in row:
                    total_dur = 0.0
                    has_audio = False
                    for t in row["conversations"]:
                        ttype = t.get("type")
                        if ttype == "audio":
                            d = _to_float(t.get("duration"))
                            if d is not None:
                                total_dur += d
                                has_audio = True
                        elif ttype == "text":
                            v = t.get("value")
                            if isinstance(v, str):
                                text_lengths.append(len(v))
                    if has_audio:
                        durations.append(total_dur)
                else:
                    d = _to_float(row.get("duration"))
                    if d is not None:
                        durations.append(d)
                    t = row.get("text")
                    if isinstance(t, str):
                        text_lengths.append(len(t))
    except FileNotFoundError:
        return str(manifest_path), 0, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)
    return (
        str(manifest_path),
        n_rows,
        np.asarray(durations, dtype=np.float32),
        np.asarray(text_lengths, dtype=np.int32),
    )


def summarize(arr, name):
    if arr.size == 0:
        return {"name": name, "count": 0}
    pcts = np.percentile(arr, PERCENTILES)
    return {
        "name": name,
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "sum": float(arr.sum()),
        "percentiles": {p: float(v) for p, v in zip(PERCENTILES, pcts)},
    }


def format_summary(stats, unit=""):
    if stats["count"] == 0:
        return f"  {stats['name']}: no data"
    s = stats
    pcts = " ".join(f"p{p}={v:.2f}{unit}" for p, v in s["percentiles"].items())
    return (
        f"  {s['name']}: count={s['count']}, sum={s['sum']:.1f}{unit}, "
        f"min={s['min']:.2f}{unit}, max={s['max']:.2f}{unit}, "
        f"mean={s['mean']:.2f}{unit}, std={s['std']:.2f}{unit}, {pcts}"
    )


def write_histogram(arr, path, title, xlabel, bins=60, log_scale=False):
    """Save a histogram PNG. Returns True on success, False if array empty."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if arr.size == 0:
        return False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(arr, bins=bins)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def label_for(path, base):
    if base:
        try:
            return os.path.relpath(path, base)
        except ValueError:
            return str(path)
    return str(path)


def main():
    parser = argparse.ArgumentParser(
        description="Compute distributions of audio durations and text lengths across NeMo manifests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_paths", nargs="+", help="JSONL manifests, directories, or YAML configs.")
    parser.add_argument("--recursive", action="store_true", default=False, help="Recurse into directories.")
    parser.add_argument("--pattern", default="*.jsonl", help="Glob pattern for manifest files in directories (default: *.jsonl).")
    parser.add_argument("--output_dir", type=str, default="manifest_distributions", help="Directory where the report and plots are written (default: ./manifest_distributions).")
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Parallel worker processes for parsing manifests.")
    parser.add_argument("--no_plots", action="store_true", default=False, help="Skip generating histogram PNGs (text report only).")
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins (default: 60).")
    parser.add_argument("--log_y", action="store_true", default=False, help="Use a log scale on the y-axis of histograms (helpful for skewed distributions).")
    parser.add_argument("--plot_min_duration", type=float, default=None, help="Exclude durations below this value (seconds) from duration plots.")
    parser.add_argument("--plot_max_duration", type=float, default=None, help="Exclude durations above this value (seconds) from duration plots.")
    args = parser.parse_args()

    # Resolve all manifests across inputs
    all_manifests = []
    for inp in args.input_paths:
        found = resolve_manifest_paths(inp, pattern=args.pattern, recursive=args.recursive)
        if not found:
            logger.warning(f"No manifests found for: {inp}")
            continue
        all_manifests.extend(found)

    if not all_manifests:
        logger.error("No manifests to process.")
        sys.exit(1)

    # Deduplicate, preserving order
    seen = set()
    manifests = []
    for m in all_manifests:
        s = str(m)
        if s not in seen:
            seen.add(s)
            manifests.append(s)

    logger.info(f"Found {len(manifests)} manifest(s) to process with {args.num_workers} worker(s)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    if not args.no_plots:
        if plots_dir.exists():
            shutil.rmtree(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

    base = os.path.commonpath(manifests) if len(manifests) > 1 else os.path.dirname(manifests[0])

    per_manifest = []  # list of (label, n_rows, durations, text_lengths)
    global_durations = []
    global_text_lengths = []
    total_rows = 0

    if args.num_workers > 1 and len(manifests) > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(collect_from_manifest, m): m for m in manifests}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Parsing manifests"):
                path, n_rows, durs, texts = fut.result()
                label = label_for(path, base)
                per_manifest.append((label, n_rows, durs, texts))
                if durs.size:
                    global_durations.append(durs)
                if texts.size:
                    global_text_lengths.append(texts)
                total_rows += n_rows
    else:
        for m in tqdm(manifests, desc="Parsing manifests"):
            path, n_rows, durs, texts = collect_from_manifest(m)
            label = label_for(path, base)
            per_manifest.append((label, n_rows, durs, texts))
            if durs.size:
                global_durations.append(durs)
            if texts.size:
                global_text_lengths.append(texts)
            total_rows += n_rows

    per_manifest.sort(key=lambda x: x[0])

    global_durations = np.concatenate(global_durations) if global_durations else np.empty(0, dtype=np.float32)
    global_text_lengths = np.concatenate(global_text_lengths) if global_text_lengths else np.empty(0, dtype=np.int32)

    # ---- Write report ----
    report_path = output_dir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Manifest distributions ===\n")
        f.write(f"Manifests: {len(manifests)}\n")
        f.write(f"Total rows: {total_rows}\n")
        f.write(f"Common base: {base}\n\n")

        f.write("=== Global ===\n")
        f.write(format_summary(summarize(global_durations, "duration"), unit="s") + "\n")
        f.write(format_summary(summarize(global_text_lengths, "text length"), unit=" chars") + "\n\n")

        f.write("=== Per-manifest ===\n")
        for label, n_rows, durs, texts in per_manifest:
            f.write(f"\n[{label}] rows={n_rows}\n")
            f.write(format_summary(summarize(durs, "duration"), unit="s") + "\n")
            f.write(format_summary(summarize(texts, "text length"), unit=" chars") + "\n")

    logger.info(f"Wrote text report: {report_path}")

    # ---- Write per-manifest CSV (compact, easy to grep / sort) ----
    csv_path = output_dir / "summary.csv"
    cols = [
        "manifest", "rows",
        "dur_count", "dur_sum_s", "dur_min_s", "dur_max_s", "dur_mean_s", "dur_p50_s", "dur_p95_s", "dur_p99_s",
        "txt_count", "txt_min", "txt_max", "txt_mean", "txt_p50", "txt_p95", "txt_p99",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for label, n_rows, durs, texts in per_manifest:
            ds = summarize(durs, "duration")
            ts = summarize(texts, "text length")
            row = [
                label, str(n_rows),
                str(ds.get("count", 0)),
                f"{ds.get('sum', 0):.3f}" if ds["count"] else "",
                f"{ds.get('min', 0):.3f}" if ds["count"] else "",
                f"{ds.get('max', 0):.3f}" if ds["count"] else "",
                f"{ds.get('mean', 0):.3f}" if ds["count"] else "",
                f"{ds['percentiles'][50]:.3f}" if ds["count"] else "",
                f"{ds['percentiles'][95]:.3f}" if ds["count"] else "",
                f"{ds['percentiles'][99]:.3f}" if ds["count"] else "",
                str(ts.get("count", 0)),
                f"{ts.get('min', 0):.0f}" if ts["count"] else "",
                f"{ts.get('max', 0):.0f}" if ts["count"] else "",
                f"{ts.get('mean', 0):.1f}" if ts["count"] else "",
                f"{ts['percentiles'][50]:.0f}" if ts["count"] else "",
                f"{ts['percentiles'][95]:.0f}" if ts["count"] else "",
                f"{ts['percentiles'][99]:.0f}" if ts["count"] else "",
            ]
            if "," in row[0] or '"' in row[0]:
                row[0] = '"' + row[0].replace('"', '""') + '"'
            f.write(",".join(row) + "\n")
    logger.info(f"Wrote CSV summary: {csv_path}")

    # ---- Plots ----
    if not args.no_plots:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            logger.error("matplotlib not installed — cannot generate plots. `pip install matplotlib` or pass --no_plots.")
        else:
            def _filter_durations(a):
                if a.size == 0 or (args.plot_min_duration is None and args.plot_max_duration is None):
                    return a
                mask = np.ones(a.shape, dtype=bool)
                if args.plot_min_duration is not None:
                    mask &= a >= args.plot_min_duration
                if args.plot_max_duration is not None:
                    mask &= a <= args.plot_max_duration
                return a[mask]

            range_suffix = ""
            if args.plot_min_duration is not None or args.plot_max_duration is not None:
                lo = "" if args.plot_min_duration is None else f"{args.plot_min_duration:g}"
                hi = "" if args.plot_max_duration is None else f"{args.plot_max_duration:g}"
                range_suffix = f" [{lo}-{hi}s]"

            written = 0
            try:
                if write_histogram(_filter_durations(global_durations), plots_dir / "global_durations.png",
                                "Global duration distribution" + range_suffix, "duration (s)",
                                bins=args.bins, log_scale=args.log_y):
                    written += 1
                if write_histogram(global_text_lengths, plots_dir / "global_text_lengths.png",
                                "Global text length distribution", "text length (chars)",
                                bins=args.bins, log_scale=args.log_y):
                    written += 1

                for label, _, durs, texts in tqdm(per_manifest, desc="Plotting", leave=False):
                    label_path = Path(label)
                    sub_dir = plots_dir / label_path.parent
                    sub_dir.mkdir(parents=True, exist_ok=True)
                    stem = label_path.stem
                    durs_plot = _filter_durations(durs)
                    if durs_plot.size:
                        if write_histogram(durs_plot, sub_dir / f"{stem}__durations.png",
                                        f"{label} — durations" + range_suffix, "duration (s)",
                                        bins=args.bins, log_scale=args.log_y):
                            written += 1
                    if texts.size:
                        if write_histogram(texts, sub_dir / f"{stem}__text_lengths.png",
                                        f"{label} — text lengths", "text length (chars)",
                                        bins=args.bins, log_scale=args.log_y):
                            written += 1
            except Exception as e:
                logger.error(f"Plotting failed after {written} plot(s): {type(e).__name__}: {e}")
            if written:
                logger.info(f"Wrote {written} plot(s) to: {plots_dir}")
            else:
                logger.warning(f"No plots written to {plots_dir} (no data points or all arrays were empty).")

    # ---- Console summary ----
    print("\n=== Global ===")
    print(format_summary(summarize(global_durations, "duration"), unit="s"))
    print(format_summary(summarize(global_text_lengths, "text length"), unit=" chars"))
    print(f"\nReport: {report_path}")
    print(f"CSV:    {csv_path}")
    if not args.no_plots:
        print(f"Plots:  {plots_dir}")


if __name__ == "__main__":
    main()
