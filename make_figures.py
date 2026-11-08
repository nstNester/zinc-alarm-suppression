#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Рисунки для статьи: схема (рис. 1), граф и подграф G_W (рис. 2), результаты Монте-Карло (рис. 3)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
sys.path.insert(0, str(Path(__file__).parent))
import alarm_sim as A

res = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figs")
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

POS = {"N1": (0.0, 2.0), "N2": (1.7, 3.3), "N3": (3.4, 2.0), "N4": (5.1, 3.3), "N5": (6.9, 3.3),
       "N6": (5.1, 0.7), "N7": (6.9, 0.7), "N8": (8.6, 0.7)}
NAME = {"N1": "N1\nОбжиг", "N2": "N2\nГазоочистка,\nкислота", "N3": "N3\nНейтральное\nвыщелачивание",
        "N4": "N4\nКислое\nвыщелачивание", "N5": "N5\nВельц-печь", "N6": "N6\nОчистка\nраствора",
        "N7": "N7\nЭлектро-\nэкстракция", "N8": "N8\nПлавка,\nотливка"}
W_, H_ = 1.35, 0.9


def node(ax, n, color="#e8eef7", ec="#35506b", alpha=1.0):
    x, y = POS[n]
    ax.add_patch(FancyBboxPatch((x - W_ / 2, y - H_ / 2), W_, H_, boxstyle="round,pad=0.03", fc=color, ec=ec, lw=1.2, alpha=alpha, zorder=2))
    ax.text(x, y, NAME[n], ha="center", va="center", fontsize=8, zorder=3)


def edge(ax, a, b, rec=False, style="-", color="black", rad=0.0, lw=1.4, label=None, lab_off=(0, 0), alpha=1.0,
         path=None, lab_xy=None):
    """Прямая стрелка между блоками или кривая Безье по явным точкам path=[(start),(control),(end)]."""
    from matplotlib.path import Path as P
    if path is not None:
        pth = P(path, [P.MOVETO, P.CURVE3, P.CURVE3])
        ax.add_patch(FancyArrowPatch(path=pth, arrowstyle="-|>", mutation_scale=11, lw=lw, ls=style, color=color,
                                     zorder=1, alpha=alpha))
        if label and lab_xy:
            ax.text(*lab_xy, label, fontsize=7, color=color, ha="center", va="center", zorder=4,
                    bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))
        return
    (x1, y1), (x2, y2) = POS[a], POS[b]
    dx, dy = x2 - x1, y2 - y1
    sx = min(W_ / 2 / abs(dx) if dx else 9, H_ / 2 / abs(dy) if dy else 9) * 1.08
    p1 = (x1 + dx * sx, y1 + dy * sx)
    p2 = (x2 - dx * sx, y2 - dy * sx)
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11, lw=lw, ls=style, color=color,
                                 zorder=1, alpha=alpha))
    if label:
        ax.text((p1[0] + p2[0]) / 2 + lab_off[0], (p1[1] + p2[1]) / 2 + lab_off[1], label, fontsize=7, color=color,
                ha="center", va="center", zorder=4, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))


# явные маршруты рециклов (вне блоков)
REC_PATH = {
    ("N4", "N3"): ([(5.45, 2.83), (5.0, 2.0), (4.1, 1.75)], (4.95, 2.05)),
    ("N5", "N3"): ([(6.9, 3.78), (4.4, 5.3), (3.2, 2.47)], (5.0, 4.25)),
    ("N7", "N3"): ([(6.9, 0.23), (5.2, -1.1), (3.2, 1.53)], (5.0, -0.45)),
}
REC_LABEL = {("N4", "N3"): "рецикл раствора", ("N5", "N3"): "рецикл вельц-оксида", ("N7", "N3"): "рецикл электролита"}


# ---------- Рис. 1: схема ----------
fig, ax = plt.subplots(figsize=(9.2, 5.6))
ax.set_xlim(-1, 9.6); ax.set_ylim(-1.5, 5.6); ax.axis("off")
for n in POS:
    node(ax, n)
fwd = [("N1", "N2"), ("N1", "N3"), ("N2", "N4"), ("N4", "N5"), ("N3", "N6"), ("N6", "N7"), ("N7", "N8")]
for a, b in fwd:
    edge(ax, a, b)
edge(ax, None, None, path=[(3.75, 2.47), (3.95, 2.75), (4.45, 2.85)])   # N3 -> N4
for k, (pth, lab_xy) in REC_PATH.items():
    edge(ax, None, None, style="--", color="#b22222", path=pth, label=REC_LABEL[k], lab_xy=lab_xy)
ax.text(4.3, 5.55, "Обобщённая авторская схема по открытым источникам [13, 14, 16]; не схема конкретного предприятия.\n"
        "Сплошные стрелки - прямые потоки, штриховые красные - рецикловые.", fontsize=7.5, ha="center", va="top")
fig.tight_layout(); fig.savefig(out / "fig1_flowsheet.png", dpi=220); plt.close(fig)

# ---------- Рис. 2: граф и G_W ----------
cfg = A.Cfg()
G = A.build_graph(cfg)
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2))
for k, ax in enumerate(axs):
    ax.set_xlim(-1, 9.6); ax.set_ylim(-1.5, 5.6); ax.axis("off")
    for n in POS:
        node(ax, n)
    for a, b, d in G.edges(data=True):
        rec = d["kind"].startswith("рецикл")
        inW = d["tau"] <= cfg.W
        lab = f"{d['tau']:.0f}"
        if rec:
            pth, lxy = REC_PATH[(a, b)]
            if k == 0:
                edge(ax, None, None, style="--", color="#b22222", path=pth, label=lab, lab_xy=lxy)
            else:
                edge(ax, None, None, style=":", color="#9a9a9a", path=pth, lw=1.0)
        elif (a, b) == ("N3", "N4"):
            pth = [(3.75, 2.47), (3.95, 2.75), (4.45, 2.85)]
            if k == 0 or inW:
                edge(ax, None, None, path=pth, label=lab, lab_xy=(3.75, 2.95))
            else:
                edge(ax, None, None, style=":", color="#9a9a9a", path=pth, lw=1.0)
        else:
            if k == 0 or inW:
                edge(ax, a, b, label=lab, lab_off=(0, 0.17))
            else:
                edge(ax, a, b, style=":", color="#9a9a9a", lw=1.0, label=f"{lab}", lab_off=(0, 0.17))
axs[0].set_title("а) полный граф: τ_e, мин (контуры N3-N7 есть)", fontsize=9)
axs[1].set_title(f"б) подграф G_W при W = {cfg.W:.0f} мин (серым - исключённые дуги, граф ацикличен)", fontsize=9)
fig.tight_layout(); fig.savefig(out / "fig2_graph.png", dpi=220); plt.close(fig)
print("рис. 1-2 готовы")

# ---------- Рис. 3: Монте-Карло ----------
mcp = res / "montecarlo.csv"
if mcp.exists():
    mc = pd.read_csv(mcp)
    scn = [s for s in ["S2", "S3", "S4", "S6", "S7", "S8"] if s in set(mc.scenario)]
    def q(x, k=1.0):
        x = np.asarray(x, float) * k
        x = x[~np.isnan(x)]
        return (np.median(x), np.percentile(x, 2.5), np.percentile(x, 97.5)) if len(x) else (np.nan,) * 3
    fig, axs = plt.subplots(1, 3, figsize=(14, 5.2))
    cols = {"B2": "#c0392b", "P": "#7f8c8d", "P2": "#1f77b4"}
    def panel(ax, key, title, ylab, variants, k=1.0, ylim=None):
        for i, v in enumerate(variants):
            xs, med, lo, hi = [], [], [], []
            for j, s in enumerate(scn):
                m, l, h = q(mc[mc.scenario == s][f"{key}_{v}"], k)
                xs.append(j + (i - (len(variants) - 1) / 2) * 0.25); med.append(m); lo.append(m - l); hi.append(h - m)
            ax.errorbar(xs, med, yerr=[lo, hi], fmt="o", color=cols[v], capsize=3, label=v, ms=5)
        ax.set_xticks(range(len(scn))); ax.set_xticklabels(scn)
        ax.set_title(title, fontsize=11); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
        if ylim: ax.set_ylim(*ylim)
        ax.legend(fontsize=10, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.08), frameon=False)
    panel(axs[0], "redM1", "а) снижение M1 относительно B1", "%", ["B2", "P", "P2"])
    panel(axs[1], "fs", "б) доля ложных подавлений (M7)", "%", ["B2", "P", "P2"], k=100, ylim=(-5, 105))
    mm = mc[mc.scenario.isin(["S7", "S8"])]
    ax = axs[2]
    for i, v in enumerate(["B2", "P", "P2"]):
        xs, med, lo, hi = [], [], [], []
        for j, s in enumerate(["S7", "S8"]):
            m, l, h = q(mm[mm.scenario == s][f"m62_{v}"])
            xs.append(j + (i - 1) * 0.22); med.append(m); lo.append(m - l); hi.append(h - m)
        ax.errorbar(xs, med, yerr=[lo, hi], fmt="o", color=cols[v], capsize=3, label=v, ms=5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["S7", "S8"]); ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.05, 1.12); ax.grid(alpha=0.3)
    ax.set_title("в) видимость второй первопричины (M6)", fontsize=11); ax.legend(fontsize=10, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle("Монте-Карло: медиана и 2,5-97,5-й процентили по 150 репликам", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "fig3_montecarlo.png", dpi=220); plt.close(fig)
    print("рис. 3 готов")
