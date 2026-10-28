"""Figures are drawn exclusively from saved CSV summaries."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .simulation import read_csv

COLORS = {"random":"#666666", "ucb1":"#0072B2", "epsilon_0.1":"#D55E00",
          "epsilon_0.01":"#009E73", "epsilon_0.05":"#CC79A7",
          "epsilon_0.2":"#E69F00", "epsilon_0.5":"#56B4E9"}


def label(name):
    return {"random":"Uniform random", "ucb1":"UCB1"}.get(
        name, f"$\\epsilon$-greedy ({name.removeprefix('epsilon_')})")


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
        "axes.spines.top":False, "axes.spines.right":False, "axes.grid":True,
        "grid.alpha":0.2, "figure.dpi":120, "savefig.dpi":300,
        "axes.titlesize":11, "legend.frameon":False, "pdf.fonttype":42})


def save(fig, folder, name):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder/f"{name}.png", bbox_inches="tight")
    fig.savefig(folder/f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def curve(ax, rows, metric, logx=False, normalize=False):
    for policy in dict.fromkeys(r["policy"] for r in rows):
        selected = [r for r in rows if r["policy"] == policy and r["metric"] == metric]
        selected.sort(key=lambda r:int(r["t"]))
        t = np.array([int(r["t"]) for r in selected])
        scale = t if normalize else 1
        y = np.array([float(r["mean"]) for r in selected])/scale
        lo = np.array([float(r["ci_low"]) for r in selected])/scale
        hi = np.array([float(r["ci_high"]) for r in selected])/scale
        ax.plot(t, y, color=COLORS[policy], label=label(policy), lw=1.7)
        ax.fill_between(t, lo, hi, color=COLORS[policy], alpha=0.15, linewidth=0)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("Round, $t$")


def comparison(data, figures, name="comparison"):
    style()
    rows = read_csv(Path(data)/f"{name}_curves.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    for ax, metric, ylabel in zip(axes, ["reward","pseudo_regret","recent_optimal_fraction"],
        ["Mean cumulative reward", "Mean cumulative pseudo-regret", "Optimal-arm selection frequency"]):
        curve(ax, rows, metric)
        ax.set_ylabel(ylabel)
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title("Trailing window of up to 1,000 rounds")
    axes[0].legend(fontsize=9)
    save(fig, figures, name)


def epsilon(data, figures):
    style()
    rows = read_csv(Path(data)/"epsilon_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    curve(axes[0], rows, "pseudo_regret", logx=True)
    curve(axes[1], rows, "recent_optimal_fraction", logx=True)
    axes[0].set_ylabel("Mean cumulative pseudo-regret")
    axes[1].set_ylabel("Optimal-arm selection frequency (trailing 1,000)")
    axes[1].set_ylim(0, 1.02)
    axes[0].legend(fontsize=8)
    save(fig, figures, "epsilon")


def gaps(data, figures):
    style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for column, env in enumerate(["easy", "difficult"]):
        rows = read_csv(Path(data)/f"gap_{env}_curves.csv")
        curve(axes[0,column], rows, "pseudo_regret")
        curve(axes[1,column], rows, "recent_optimal_fraction")
        axes[0,column].set_title(f"{env.capitalize()} environment")
        axes[0,column].set_ylabel("Mean cumulative pseudo-regret")
        axes[1,column].set_ylabel("Optimal-arm selection frequency (trailing 1,000)")
        axes[1,column].set_ylim(0, 1.02)
    axes[0,0].legend(fontsize=9)
    save(fig, figures, "reward_gaps")


def hoeffding(data, figures):
    style()
    rows = read_csv(Path(data)/"hoeffding.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, mu in zip(axes, sorted({float(r['mu']) for r in rows})):
        for delta, color in zip(sorted({float(r['delta']) for r in rows}), ["#0072B2","#D55E00","#009E73"]):
            s = sorted([r for r in rows if float(r['mu']) == mu and float(r['delta']) == delta], key=lambda r:int(r['n']))
            n = [int(r['n']) for r in s]
            p = np.array([float(r['estimate']) for r in s])
            low = np.array([float(r['ci_low']) for r in s])
            high = np.array([float(r['ci_high']) for r in s])
            ax.plot(n, p, "o-", ms=4, color=color, label=f"Empirical, $\\delta={delta}$")
            ax.fill_between(n, low, high, color=color, alpha=.15)
            ax.plot(n, [float(r['bound']) for r in s], "--", color=color, lw=1.3)
        ax.set(xscale="log", xlabel="Sample size, $n$", ylabel="Deviation probability", title=f"Bernoulli mean $\\mu={mu}$", ylim=(-.02,1.03))
    axes[0].legend(fontsize=8)
    fig.suptitle("Solid: empirical probabilities (95% Wilson bands); dashed: min(1, Hoeffding bound)", fontsize=10)
    save(fig, figures, "hoeffding")


def scaling(data, figures, means):
    style()
    rows = read_csv(Path(data)/"scaling_curves.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    curve(axes[0], rows, "pseudo_regret", logx=True)
    curve(axes[1], rows, "pseudo_regret", logx=True, normalize=True)
    ucb = [r for r in rows if r['policy']=="ucb1" and r['metric']=="pseudo_regret" and int(r['t']) >= len(means)]
    t = np.array([int(r['t']) for r in ucb])
    y = [float(r['mean']) for r in ucb]
    gaps = max(means)-np.array(means)
    positive = gaps[gaps>0]
    bound = 8*np.log(t)*np.sum(1/positive)+(1+np.pi**2/3)*positive.sum()
    axes[2].plot(t, y, color=COLORS['ucb1'], label="Observed UCB1")
    axes[2].plot(t, bound, color="#333333", ls="--", label="Gap-dependent upper bound")
    axes[2].set(xscale="log", xlabel="Horizon, $T$", ylabel="Cumulative pseudo-regret")
    axes[0].set_ylabel("Mean cumulative pseudo-regret")
    axes[1].set_ylabel("Mean pseudo-regret / $T$")
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    save(fig, figures, "regret_scaling")
