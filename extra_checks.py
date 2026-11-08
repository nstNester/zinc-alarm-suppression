#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Дополнительные проверки к статье: S9 при W = 180, окно разметки причинности, определение времени распространения."""
import sys
from dataclasses import replace
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import alarm_sim as A

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results"); out.mkdir(parents=True, exist_ok=True)
cfg = A.Cfg(runs=100)
cols = ["M1", "M3", "false_supp", "supp_recall", "M6_visible", "M6_second"]

# 1. S9: независимые отказы в одной компоненте связности; при W = 180 мин G_W содержит цикл
df, _ = A.run_experiment(replace(cfg, W=180.0), scenarios=["S9"])
df.groupby(["scenario", "variant"])[cols].mean().to_csv(out / "extra_S9_W180.csv")

# 2. Чувствительность разметки причинности к окну 10 / 30 / 60 мин
rows = []
for win in (10, 30, 60):
    df, _ = A.run_experiment(replace(cfg, label_win=win), scenarios=["S2", "S3", "S6", "S7", "S8"])
    g = df[df.variant.isin(["B2", "P", "P2"])].groupby(["scenario", "variant"])[["false_supp", "supp_recall"]].mean().reset_index()
    g.insert(0, "label_win", win)
    rows.append(g)
pd.concat(rows).to_csv(out / "extra_label_window.csv", index=False)

# 3. Время распространения: tau = d + T (среднее время запаздывания) вместо d + ln2*T  <=>  mis_lag = 1/ln2
rows = []
for name, mis in (("ln2*T (принято)", 1.0), ("T (среднее запаздывание)", 1.0 / A.LN2)):
    df, _ = A.run_experiment(replace(cfg, runs=30, mis_lag=mis), scenarios=["S3", "S6", "S7", "S8"])
    g = df.groupby(["scenario", "variant"])[["M1", "false_supp", "M6_second"]].mean()
    for scn in ("S3", "S6", "S7", "S8"):
        b1 = g.loc[(scn, "B1"), "M1"]
        for v in ("P", "P2"):
            rows.append(dict(tau_def=name, scenario=scn, variant=v, red_M1_pct=100 * (1 - g.loc[(scn, v), "M1"] / b1),
                             false_supp=g.loc[(scn, v), "false_supp"], M6_second=g.loc[(scn, v), "M6_second"]))
pd.DataFrame(rows).to_csv(out / "extra_tau_definition.csv", index=False)

# 4. N5 (вельц-печь, близкая к вытеснению) как запаздывание: T5 = 60 мин, дуга N4->N5 получает запаздывание 470 мин
names = list(A.NODES)
nodeT = tuple(60.0 if n == "N5" else A.NODES[n][0] for n in names)
idx45 = next(i for i, e in enumerate(A.EDGES) if e[0] == "N4" and e[1] == "N5")
dmult = tuple(470.0 / A.EDGES[i][2] if i == idx45 else 1.0 for i in range(len(A.EDGES)))
rows = []
for name, c in (("N5: апериодическое звено T = 450 мин (принято)", cfg),
                ("N5: запаздывание 470 мин + T = 60 мин", replace(cfg, node_T=nodeT, edge_d_mult=dmult))):
    df, _ = A.run_experiment(c, scenarios=["S2", "S3", "S4", "S6", "S7", "S8"])
    g = df.groupby(["scenario", "variant"])[["M1", "false_supp", "M6_second"]].mean()
    for scn in ("S2", "S3", "S4", "S6", "S7", "S8"):
        b1 = g.loc[(scn, "B1"), "M1"]
        for v in ("B2", "P", "P2"):
            rows.append(dict(N5_model=name, scenario=scn, variant=v, red_M1_pct=100 * (1 - g.loc[(scn, v), "M1"] / b1),
                             false_supp=g.loc[(scn, v), "false_supp"], M6_second=g.loc[(scn, v), "M6_second"]))
pd.DataFrame(rows).to_csv(out / "extra_N5_deadtime.csv", index=False)
print("готово")
