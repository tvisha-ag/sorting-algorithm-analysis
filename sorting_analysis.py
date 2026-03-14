"""
Comparative Analysis of Sorting Algorithms on Real-World Data
=============================================================
Empirical study analyzing QuickSort, MergeSort, and Timsort
across Random, Sorted, and Reverse-Sorted datasets.
"""

import time
import random
import sys
import statistics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────
# Increase recursion limit for large QuickSort inputs
# ─────────────────────────────────────────────
sys.setrecursionlimit(100_000)


# ══════════════════════════════════════════════
#  SORTING IMPLEMENTATIONS
# ══════════════════════════════════════════════

def quicksort(arr):
    """
    Lomuto partition scheme QuickSort.
    Average: O(n log n) | Worst: O(n²) on sorted/reverse-sorted.
    """
    arr = arr[:]

    def _qs(a, lo, hi):
        if lo < hi:
            pivot = a[hi]
            i = lo - 1
            for j in range(lo, hi):
                if a[j] <= pivot:
                    i += 1
                    a[i], a[j] = a[j], a[i]
            a[i + 1], a[hi] = a[hi], a[i + 1]
            p = i + 1
            _qs(a, lo, p - 1)
            _qs(a, p + 1, hi)

    _qs(arr, 0, len(arr) - 1)
    return arr


def mergesort(arr):
    """
    Top-down MergeSort.
    Guaranteed: O(n log n). Stable. Extra O(n) space.
    """
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    return _merge(left, right)


def _merge(left, right):
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def timsort(arr):
    """
    Python's built-in sort — a hybrid of MergeSort + InsertionSort.
    Exploits natural runs. Best: O(n). Worst: O(n log n). Stable.
    """
    a = arr[:]
    a.sort()
    return a


# ══════════════════════════════════════════════
#  DATASET GENERATORS
# ══════════════════════════════════════════════

def make_random(n):
    return [random.randint(0, 10 * n) for _ in range(n)]

def make_sorted(n):
    return list(range(n))

def make_reverse(n):
    return list(range(n, 0, -1))


# ══════════════════════════════════════════════
#  BENCHMARK ENGINE
# ══════════════════════════════════════════════

ALGORITHMS = {
    "QuickSort":  quicksort,
    "MergeSort":  mergesort,
    "Timsort":    timsort,
}

DATASETS = {
    "Random":         make_random,
    "Sorted":         make_sorted,
    "Reverse-Sorted": make_reverse,
}

SIZES   = [500, 1_000, 2_500, 5_000, 7_500, 10_000]
REPEATS = 5   # median of N runs per data point


def benchmark():
    """
    Returns nested dict:  results[algo][dataset] = list of (n, median_ms)
    """
    results = {algo: {ds: [] for ds in DATASETS} for algo in ALGORITHMS}

    total = len(SIZES) * len(ALGORITHMS) * len(DATASETS) * REPEATS
    done  = 0

    print(f"\n{'─'*58}")
    print(f"  Benchmarking  {len(ALGORITHMS)} algorithms × "
          f"{len(DATASETS)} datasets × {len(SIZES)} sizes × {REPEATS} runs")
    print(f"{'─'*58}\n")

    for n in SIZES:
        for ds_name, gen in DATASETS.items():
            data = gen(n)                       # one dataset per (n, ds)
            for algo_name, fn in ALGORITHMS.items():
                times = []
                for _ in range(REPEATS):
                    t0 = time.perf_counter()
                    fn(data)
                    times.append((time.perf_counter() - t0) * 1000)  # ms
                    done += 1

                med = statistics.median(times)
                results[algo_name][ds_name].append((n, med))

                bar = "█" * int(40 * done / total)
                print(f"\r  [{bar:<40}] {done}/{total}  "
                      f"n={n:>6}  {algo_name:<12} {ds_name:<16} "
                      f"{med:6.2f} ms", end="", flush=True)

    print(f"\n\n{'─'*58}")
    print("  Benchmarking complete ✓")
    print(f"{'─'*58}\n")
    return results


# ══════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════

PALETTE = {
    "QuickSort": "#FF6B6B",
    "MergeSort": "#4ECDC4",
    "Timsort":   "#FFE66D",
}

MARKERS = {
    "QuickSort": "o",
    "MergeSort": "s",
    "Timsort":   "^",
}

BG_DARK  = "#0D1117"
BG_PANEL = "#161B22"
GRID_CLR = "#21262D"
TEXT_CLR = "#E6EDF3"
MUTED    = "#8B949E"


def plot_results(results):
    fig = plt.figure(figsize=(18, 14), facecolor=BG_DARK)
    fig.suptitle(
        "Empirical Time Complexity: QuickSort vs MergeSort vs Timsort",
        fontsize=18, fontweight="bold", color=TEXT_CLR, y=0.97
    )

    ds_list   = list(DATASETS.keys())
    algo_list = list(ALGORITHMS.keys())

    # ── 3 main subplots (one per dataset) + 1 big overview ──────────────
    gs = GridSpec(2, 3, figure=fig,
                  top=0.91, bottom=0.1,
                  hspace=0.45, wspace=0.35)

    axes_top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_all   = fig.add_subplot(gs[1, :])   # full-width bottom row

    # ── Per-dataset panels ───────────────────────────────────────────────
    for ax, ds_name in zip(axes_top, ds_list):
        ax.set_facecolor(BG_PANEL)
        ax.set_title(ds_name, color=TEXT_CLR, fontsize=13, pad=8)

        for algo in algo_list:
            xs = [p[0] for p in results[algo][ds_name]]
            ys = [p[1] for p in results[algo][ds_name]]
            ax.plot(xs, ys,
                    color=PALETTE[algo],
                    marker=MARKERS[algo],
                    linewidth=2.2,
                    markersize=5,
                    label=algo,
                    alpha=0.92)

        ax.set_xlabel("n  (elements)", color=MUTED, fontsize=9)
        ax.set_ylabel("Time (ms)", color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.spines[:].set_color(GRID_CLR)
        ax.grid(color=GRID_CLR, linewidth=0.7, linestyle="--")
        ax.legend(fontsize=8, facecolor=BG_DARK,
                  edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

    # ── Big comparison panel (all datasets + algos) ─────────────────────
    ax_all.set_facecolor(BG_PANEL)
    ax_all.set_title(
        "All Algorithms × All Datasets  — T(n) Overview",
        color=TEXT_CLR, fontsize=13, pad=8
    )

    ds_alpha  = {"Random": 1.0, "Sorted": 0.65, "Reverse-Sorted": 0.40}
    ds_dashes = {"Random": (1, 0), "Sorted": (5, 2), "Reverse-Sorted": (2, 2)}

    for algo in algo_list:
        for ds_name in ds_list:
            xs = [p[0] for p in results[algo][ds_name]]
            ys = [p[1] for p in results[algo][ds_name]]
            ax_all.plot(
                xs, ys,
                color=PALETTE[algo],
                alpha=ds_alpha[ds_name],
                dashes=ds_dashes[ds_name],
                linewidth=1.8,
                marker=MARKERS[algo],
                markersize=4,
            )

    # Legend: algorithms
    algo_patches = [
        mpatches.Patch(color=PALETTE[a], label=a) for a in algo_list
    ]
    # Legend: line styles → datasets
    from matplotlib.lines import Line2D
    ds_handles = [
        Line2D([0], [0], color=TEXT_CLR, dashes=ds_dashes[d],
               alpha=ds_alpha[d], label=d)
        for d in ds_list
    ]

    legend1 = ax_all.legend(
        handles=algo_patches, loc="upper left",
        fontsize=9, facecolor=BG_DARK,
        edgecolor=GRID_CLR, labelcolor=TEXT_CLR, title="Algorithm",
        title_fontsize=8
    )
    legend1.get_title().set_color(MUTED)
    ax_all.add_artist(legend1)

    legend2 = ax_all.legend(
        handles=ds_handles, loc="upper center",
        fontsize=9, facecolor=BG_DARK,
        edgecolor=GRID_CLR, labelcolor=TEXT_CLR, title="Dataset Type",
        title_fontsize=8
    )
    legend2.get_title().set_color(MUTED)

    ax_all.set_xlabel("n  (elements)", color=MUTED, fontsize=10)
    ax_all.set_ylabel("Time (ms)", color=MUTED, fontsize=10)
    ax_all.tick_params(colors=MUTED, labelsize=9)
    ax_all.spines[:].set_color(GRID_CLR)
    ax_all.grid(color=GRID_CLR, linewidth=0.7, linestyle="--")

    # ── Caption ─────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.02,
        "Median of 5 runs per data point  ·  Sizes: 500 – 10,000 elements  "
        "·  Highlights constants hidden within Big-O notation",
        ha="center", fontsize=8, color=MUTED
    )

    out = "/mnt/user-data/outputs/sorting_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    print(f"  Chart saved → {out}")
    plt.close()
    return out


# ══════════════════════════════════════════════
#  PRINT SUMMARY TABLE
# ══════════════════════════════════════════════

def print_summary(results):
    n_max = SIZES[-1]
    print(f"\n{'═'*66}")
    print(f"  RESULTS  at  n = {n_max:,}")
    print(f"{'═'*66}")
    header = f"  {'Algorithm':<14}" + "".join(f"  {d:<18}" for d in DATASETS)
    print(header)
    print(f"{'─'*66}")
    for algo in ALGORITHMS:
        row = f"  {algo:<14}"
        for ds in DATASETS:
            val = results[algo][ds][-1][1]
            row += f"  {val:>8.2f} ms       "
        print(row)
    print(f"{'═'*66}\n")

    print("  Key Takeaways:")
    print("  ─────────────────────────────────────────────────────────")
    qs_rand = results["QuickSort"]["Random"][-1][1]
    qs_rev  = results["QuickSort"]["Reverse-Sorted"][-1][1]
    ms_rand = results["MergeSort"]["Random"][-1][1]
    ts_rand = results["Timsort"]["Random"][-1][1]
    ts_sort = results["Timsort"]["Sorted"][-1][1]

    print(f"  • QuickSort degradation  (random → reverse):  "
          f"{qs_rand:.1f} ms → {qs_rev:.1f} ms  "
          f"(×{qs_rev/qs_rand:.1f} slower)")
    print(f"  • MergeSort vs QuickSort on random data:  "
          f"{ms_rand:.1f} ms vs {qs_rand:.1f} ms")
    print(f"  • Timsort on pre-sorted data:  {ts_sort:.2f} ms  "
          f"(near O(n) – natural run detection)")
    print(f"  • Timsort constant advantage over MergeSort:  "
          f"×{ms_rand/ts_rand:.1f} on random\n")


# ══════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════

if __name__ == "__main__":
    random.seed(42)
    data = benchmark()
    print_summary(data)
    plot_results(data)
