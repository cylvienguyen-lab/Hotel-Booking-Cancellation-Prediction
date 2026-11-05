"""Write the monthly comparison tables and charts for H1 and H2."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter, MaxNLocator

from inference import CONTACT_LIMIT, publish_folder, replay_range, run_folder, windows
from monitoring import EXTERNAL_INPUTS, assessment_dates, monitoring_folder

BLUE, GRAY, GOLD = "#195596", "#5d636a", "#aa6530"
HOTELS = {"H1": "Resort", "H2": "City"}


def format_axis(ax, months):
    ax.set_xticks(range(len(months)), [pd.Timestamp(m).strftime("%b %Y") for m in months])
    ax.tick_params(length=0, pad=5, labelsize=9)
    ax.grid(axis="y", alpha=0.19)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)


def lines(ax, data, model, baseline=None, percent=False, baseline_label="Validation"):
    x = range(len(data))
    # Leave a gap when a month's outcomes are still pending. Zero would imply a measured result.
    values = data.get(model, pd.Series(np.nan, index=data.index))
    ax.plot(x, values, color=BLUE, marker="o", linewidth=2, label="Model")
    if baseline:
        ax.plot(x, data.get(baseline, pd.Series(np.nan, index=data.index)), color=GRAY,
                linestyle="--", marker="x", label=baseline_label)
        ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.margins(x=0.11, y=0.25)


def make_charts(data, stage):
    months = sorted(data.cutoff_month.unique())
    for kind, title, cols in [
        ("business", "Do the outreach list and planning forecasts help?",
         ["Outreach hit rate", "Daily arrivals error", "Nightly room-count error"]),
        ("model", "What changed in scores and inputs?",
         ["Ranking ability", "Score distribution shift", "External inputs available"])]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.1), sharey="col")
        for row, (hotel, name) in enumerate(HOTELS.items()):
            g = data[data.Hotel.eq(hotel)].set_index("cutoff_month").reindex(months)
            for j, ax in enumerate(axes[row]):
                format_axis(ax, months)
                if row == 0:
                    ax.set_title(cols[j], fontsize=12, loc="left", pad=13, fontweight="bold")
            ax, mid, right = axes[row]
            if kind == "business":
                # Outreach uses random selection as its baseline. The forecasts use the hotel's training rate.
                lines(ax, g, "precision_at_k", "random_precision", True, "Random")
                ax.set_ylabel(f"{hotel} {name}\nPrecision@{CONTACT_LIMIT[hotel]}")
                lines(mid, g, "arrivals_mae_model", "arrivals_mae_dummy", baseline_label="Dummy")
                mid.set_ylabel("MAE · bookings")
                lines(right, g, "room_nights_mae_model", "room_nights_mae_dummy", baseline_label="Dummy")
                right.set_ylabel("MAE · rooms per night")
            else:
                lines(ax, g, "ROC_AUC", "validation_ROC_AUC")
                ax.set_ylabel(f"{hotel} {name}\nROC AUC")
                lines(mid, g, "score_PSI")
                mid.set_ylabel("PSI vs validation")
                labels = ["EUR/USD", "Hotel searches", "Flight searches"]
                cols_quality = [col + "_available" for col in EXTERNAL_INPUTS]
                matrix = g[cols_quality].T.to_numpy(dtype=float)
                right.grid(False)
                right.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
                right.set_yticks(range(3), labels)
                for r, c in np.ndindex(matrix.shape):
                    v = matrix[r, c]
                    txt = f"{v:.0%}" if np.isfinite(v) else "—"
                    right.text(c, r, txt, ha="center", va="center", color="white" if v > .6 else "#1b2838")
        if kind == "business":
            for ax in axes[0]:
                ax.set_ylim(bottom=0)
        else:
            axes[0,0].set_ylim(0.45, 1)
            axes[0,1].set_ylim(bottom=0)
        fig.suptitle(title, x=.065, y=.98, fontsize=17, ha="left", fontweight="bold")
        fig.tight_layout(rect=(0, .04, 1, .91), h_pad=2.0, w_pad=1.7)
        fig.text(.065, .013, "T−14 scoring month   •   Source: metrics.csv", color=GRAY, fontsize=9)
        fig.savefig(stage / (kind + ".png"), dpi=140)
        plt.close(fig)


def main():
    start, end = replay_range()
    out = run_folder(start, end) / "trends"
    if out.is_dir():
        return
    snapshots, mixes = [], []
    for _, last, folder in windows():
        dates = assessment_dates(last)
        report = monitoring_folder(folder, dates)
        data = pd.read_csv(report / "monthly.csv")
        # Each assessment covers the same month's bookings. Keep the latest to avoid double-counting.
        current = data[data.assessment_date.eq(str(dates[-1]))].copy()
        if sorted(current.Hotel) != ["H1", "H2"]:
            raise ValueError("Expected both hotel snapshots for each month")
        snapshots.append(current)
        mixes.append(pd.read_csv(report / "input_mix.csv"))
    metrics = pd.concat(snapshots, ignore_index=True)
    if metrics.groupby("Hotel").model_version.nunique().ne(1).any():
        raise ValueError("Trend comparison spans multiple model versions for a hotel")
    mix = pd.concat(mixes, ignore_index=True)
    with publish_folder(out) as stage:
        metrics.to_csv(stage / "metrics.csv", index=False)
        mix.to_csv(stage / "input_mix.csv", index=False)
        make_charts(metrics, stage)
    print("Monitoring charts and CSVs:", out)


if __name__ == "__main__":
    main()
