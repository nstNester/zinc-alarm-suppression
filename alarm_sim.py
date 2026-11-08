#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alarm_sim.py - имитационный эксперимент для статьи «Рационализация системы тревог АСУ ТП
безотходного цинкового производства с замкнутыми рециклами на основе графовой модели».

Параметры взяты из открытых источников там, где источник найден; остальные помечены
«допущение» и покрываются анализом чувствительности (--sens). Таблицы параметров с
источниками записываются в results/params_*.csv (приложение А статьи).

Варианты:  B0 - уставка на границе нормальной области, без гистерезиса и задержки;
           B1 - рационализация без топологии: уставка вне области, гистерезис, задержка, приоритеты;
           P  - B1 + подавление следствий по подграфу G_W (дуги с временем распространения <= W).
Зависимости: numpy, pandas, networkx, scipy (matplotlib - необязательно).
Запуск:  python alarm_sim.py --runs 100 --out results [--sens --sens-runs 30]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

LN2 = math.log(2.0)
ITS = "ИТС 13-2020"
WAELZ = "Tzouganatos et al., JOM 65(12), 2013"
PAT1 = "US 4545963"
PAT2 = "US 4610721"
PAT3 = "US 4004991"
ASM = "допущение"

# ---- узлы: (постоянная времени узла T, мин; источник) ------------------------------
NODES = {
    "N1": (10, ASM + " (кипящий слой, быстрая динамика)"),
    "N2": (10, ASM),
    "N3": (60, f"{PAT2}: время выщелачивания 1-4 ч, принято нижнее значение"),
    "N4": (120, f"{PAT2}: время выщелачивания 1-4 ч, принято 2 ч"),
    "N5": (450, f"{WAELZ}: время пребывания в вельц-печи 6-9 ч, принято 7,5 ч"),
    "N6": (90, f"{PAT3}, US 4149945: стадии обработки раствора 1,4-3 ч, принято 1,5 ч (по аналогии)"),
    "N7": (60, ASM),
    "N8": (10, ASM),
}
# ---- дуги: (из, в, транспортная задержка d, мин; коэффициент передачи g; вид; источник) --
EDGES = [
    ("N1", "N2", 2, 0.8, "газ", ASM),
    ("N1", "N3", 10, 0.7, "огарок", ASM),
    ("N2", "N4", 15, 0.5, "кислота", ASM),
    ("N3", "N4", 8, 0.8, "пульпа", ASM),
    ("N4", "N5", 20, 0.2, "кек", f"{ITS}: в остатке выщелачивания 5-30 % Zn, принято 20 %"),
    ("N3", "N6", 6, 0.8, "раствор", ASM),
    ("N6", "N7", 10, 0.8, "раствор", ASM),
    ("N7", "N8", 5, 0.6, "катодный цинк", ASM),
    ("N4", "N3", 120, 0.3, "рецикл раствора", ASM + " (буферные ёмкости)"),
    ("N5", "N3", 180, 0.25, "рецикл вельц-оксида", ASM),
    ("N7", "N3", 120, 0.4, "рецикл электролита", f"{PAT1}: отработанный электролит возвращается на выщелачивание; задержка - допущение"),
]
RECYCLE_KINDS = {"рецикл раствора", "рецикл вельц-оксида", "рецикл электролита"}

TAG_DEFAULTS = dict(k=1.0, sigma=0.35, phi=0.9, thr0=1.0, thr1=1.5, db=0.2, d_on=3)


def mk(name, node, S, Tc, resp, unit, band, src, **kw):
    d = dict(name=name, node=node, S=S, Tc=Tc, resp=resp, unit=unit, band=band, src=src, **TAG_DEFAULTS)
    d.update(kw)
    return d


# S - тяжесть (1..4); Tc - класс времени реакции (0: >30, 1: 10-30, 2: <10 мин);
# resp: fast - быстро реагирующая переменная (T, давление, расход), node - медленная (состав)
TAGS = [
    mk("N1.T_roast", "N1", 3, 1, "fast", "°C", "900-1000", f"{ITS}, п. 2.2.2.1"),
    mk("N1.draft", "N1", 2, 1, "fast", "Па", "-", ASM),
    mk("N2.SO2_conv", "N2", 3, 1, "fast", "%", "-", ASM),
    mk("N2.acid_conc", "N2", 2, 0, "node", "г/дм3", "-", ASM),
    mk("N3.pH", "N3", 2, 1, "node", "pH", "4,0-4,5", f"{ITS}, п. 2.2.2.3"),
    mk("N3.T", "N3", 2, 0, "fast", "°C", "около 50", f"{ITS}, п. 2.2.2.3 (допуск - допущение)"),
    mk("N4.H2SO4", "N4", 3, 1, "node", "г/дм3", "до 80", f"{ITS}, п. 2.2.2.2 (полуширина - допущение)"),
    mk("N4.T", "N4", 2, 1, "fast", "°C", "-", ASM),
    mk("N5.kiln_T", "N5", 3, 2, "fast", "°C", "около 1200", WAELZ),
    mk("N5.dust_flow", "N5", 1, 0, "node", "т/ч", "-", ASM),
    mk("N6.Cu_Cd", "N6", 2, 1, "node", "мг/дм3", "-", ASM),
    mk("N6.Zn_dust", "N6", 1, 0, "node", "% от катодов", "1,5-6", f"{ITS}, п. 2.2.2.5"),
    mk("N6.H2_LEL", "N6", 4, 2, "fast", "% НПВ", "ниже НПВ", f"{ITS}, п. 2.2.2.5 (непрерывный контроль H2)"),
    mk("N7.cell_T", "N7", 3, 1, "fast", "°C", "-", ASM),
    mk("N7.acid_mist", "N7", 4, 2, "fast", "мг/м3", "-", f"{ITS}, п. 2.2.2.7 (выбросы аэрозоля); уровень - допущение"),
    mk("N7.Zn_electrolyte", "N7", 2, 0, "node", "г/дм3", "40-60", PAT1),
    mk("N8.furnace_T", "N8", 2, 0, "fast", "°C", "-", ASM),
    mk("N8.level", "N8", 1, 0, "node", "м", "-", ASM),
]
MATRIX = {1: [1, 1, 2], 2: [1, 2, 2], 3: [2, 3, 3], 4: [3, 3, 3]}
SCENARIOS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]
DEFAULT_SCENARIOS = SCENARIOS[:8]   # S9 прогоняется отдельно при W = 180 мин (extra_checks.py)
METRICS = ["M1", "M2", "M3", "M4", "M5"]


@dataclass
class Cfg:
    horizon: int = 2880
    runs: int = 100
    seed: int = 12345
    W: float = 120.0           # горизонт причинной связности, мин (дуги с tau_char <= W)
    k_margin: float = 30.0     # запас по времени при подавлении, мин
    lag_fast: float = 5.0      # постоянная времени быстрых переменных, мин
    tau_scale: float = 1.0
    lag_scale: float = 1.0
    thr_scale: float = 1.0
    sigma_scale: float = 1.0
    stale_min: int = 1440      # порог «стоящей» тревоги (>24 ч): Emerson, ISA-18.2
    bin_min: int = 10
    flood_thr: int = 10        # лавина: > 10 тревог за 10 мин: ISA-18.2, EEMUA 191
    root_window: int = 180
    mis_tau: float = 1.0       # ошибка модели: граф подавления строится с задержками x mis_tau
    mis_lag: float = 1.0       # ... и с постоянными времени x mis_lag (сигналы генерируются по истинным)
    t_window: float = 150.0    # окно базового варианта B2 (группировка по времени без графа), мин
    node_T: tuple | None = None        # переопределение постоянных времени узлов (Монте-Карло)
    edge_d_mult: tuple | None = None   # множители транспортных задержек по дугам
    edge_g_mult: tuple | None = None   # множители коэффициентов передачи по дугам
    phi_override: float | None = None  # общий коэффициент AR(1)
    amp_scale: float = 1.0             # масштаб амплитуд возмущений
    label_win: int = 30                # окно разметки истинной причинности, мин


def node_T(cfg: Cfg, nm: str) -> float:
    if cfg.node_T:
        return cfg.node_T[list(NODES).index(nm)]
    return NODES[nm][0] * cfg.lag_scale


def build_graph(cfg: Cfg) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(NODES)
    for ei, (a, b, d, g, kind, _) in enumerate(EDGES):
        dm = cfg.edge_d_mult[ei] if cfg.edge_d_mult else 1.0
        gm = cfg.edge_g_mult[ei] if cfg.edge_g_mult else 1.0
        d_eff = max(1, int(round(d * cfg.tau_scale * dm)))
        tau_char = d_eff + LN2 * node_T(cfg, b)    # время половинного отклика приёмника
        G.add_edge(a, b, d=d_eff, gain=min(0.95, g * gm), kind=kind, tau=tau_char)
    return G


def window_graph(G, W):
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes)
    H.add_edges_from((a, b, dd) for a, b, dd in G.edges(data=True) if dd["tau"] <= W)
    return H


def graph_report(cfg: Cfg):
    G = build_graph(cfg)
    H = window_graph(G, cfg.W)
    L = []
    scc = [sorted(c) for c in nx.strongly_connected_components(G) if len(c) > 1]
    L.append(f"Полный граф: узлов {G.number_of_nodes()}, дуг {G.number_of_edges()}; "
             f"нетривиальные компоненты сильной связности: {scc}")
    for a, b, dd in sorted(G.edges(data=True), key=lambda e: e[2]["tau"]):
        L.append(f"  дуга {a}->{b} ({dd['kind']}): d={dd['d']} мин, g={dd['gain']}, tau_char={dd['tau']:.0f} мин")
    for cyc in nx.simple_cycles(G):
        pairs = list(zip(cyc, cyc[1:] + cyc[:1]))
        gain = math.prod(G[a][b]["gain"] for a, b in pairs)
        L.append(f"  контур {'->'.join(cyc)}: усиление {gain:.3f} ({'OK' if gain < 1 else 'НЕУСТОЙЧИВО'})")
        if gain >= 1:
            raise ValueError("Усиление контура >= 1")
    taus = sorted({round(dd["tau"], 6) for _, _, dd in G.edges(data=True)})
    w_cycle = next((w for w in taus if not nx.is_directed_acyclic_graph(window_graph(G, w))), None)
    L.append(f"G_W ацикличен при W < {w_cycle:.0f} мин" if w_cycle else "G_W ацикличен при любом W")
    acyc = nx.is_directed_acyclic_graph(H)
    L.append(f"При W = {cfg.W:.0f} мин: дуг в G_W {H.number_of_edges()}, ацикличен: {acyc}")
    return "\n".join(L), G, H


def build_signals(scn, rng, cfg: Cfg, G):
    T, n = cfg.horizon, len(NODES)
    names = list(NODES)
    idx = {nm: i for i, nm in enumerate(names)}
    ext = np.zeros((n, T))
    meta = {"src": None, "t0": None}
    drift = np.zeros((len(TAGS), T))
    extra = np.zeros((len(TAGS), T))

    def pulse(node, t0, dur, amp):
        ext[idx[node], t0:min(T, t0 + dur)] += amp * cfg.amp_scale

    if scn == "S2":
        src = str(rng.choice(["N1", "N3", "N6", "N7"]))
        t0 = int(rng.integers(300, 1200))
        pulse(src, t0, int(rng.integers(120, 360)), rng.uniform(3, 6))
        meta = {"src": src, "t0": t0}
    elif scn == "S3":
        t0 = int(rng.integers(300, 1200))
        pulse("N7", t0, int(rng.integers(180, 480)), rng.uniform(4, 8))
        meta = {"src": "N7", "t0": t0}
    elif scn == "S4":
        for i, nm in enumerate(names):
            pulse(nm, 60 * i + int(rng.integers(0, 30)), int(rng.integers(120, 300)), rng.uniform(2, 5))
    elif scn == "S5":
        j = int(rng.integers(0, len(TAGS)))
        t0 = int(rng.integers(200, 600))
        D = rng.uniform(2, 4)
        drift[j] = D * np.clip((np.arange(T) - t0) / 600.0, 0, 1)
        extra[j, t0:] = rng.normal(0, 0.15, T - t0)
    elif scn == "S6":     # общая причина: все узлы отклоняются одновременно
        t0 = int(rng.integers(300, 1200))
        dur = int(rng.integers(240, 480))
        for nm in names:
            pulse(nm, t0, dur, rng.uniform(3, 6))
        meta = {"src": None, "t0": t0}
    elif scn == "S7":     # два независимых отказа в узлах, не связанных в G_W
        a_, b_ = [("N2", "N5"), ("N5", "N8"), ("N2", "N6"), ("N4", "N6"), ("N5", "N7"),
                  ("N2", "N3")][int(rng.integers(0, 6))]
        t0 = int(rng.integers(300, 900))
        t1 = t0 + int(rng.integers(10, 60))
        pulse(a_, t0, int(rng.integers(120, 360)), rng.uniform(3, 6))
        pulse(b_, t1, int(rng.integers(120, 360)), rng.uniform(3, 6))
        meta = {"src": a_, "t0": t0, "src2": b_, "t02": t1}
    elif scn == "S8":     # независимые отказы: нижний узел связан с верхним путём в G_W (неблагоприятно для правил по графу)
        a_, b_ = [("N1", "N3"), ("N3", "N6"), ("N6", "N7"), ("N3", "N4"), ("N1", "N7"),
                  ("N7", "N8")][int(rng.integers(0, 6))]
        t0 = int(rng.integers(300, 900))
        t1 = t0 + int(rng.integers(10, 60))
        pulse(a_, t0, int(rng.integers(120, 360)), rng.uniform(3, 6))
        pulse(b_, t1, int(rng.integers(120, 360)), rng.uniform(3, 6))
        meta = {"src": a_, "t0": t0, "src2": b_, "t02": t1}
    elif scn == "S9":     # независимые отказы в узлах одной компоненты сильной связности (прогонять при W = 180)
        a_, b_ = [("N3", "N7"), ("N4", "N6"), ("N3", "N4"), ("N6", "N7"), ("N4", "N7"),
                  ("N6", "N3")][int(rng.integers(0, 6))]
        t0 = int(rng.integers(300, 900))
        t1 = t0 + int(rng.integers(10, 60))
        pulse(a_, t0, int(rng.integers(120, 360)), rng.uniform(3, 6))
        pulse(b_, t1, int(rng.integers(120, 360)), rng.uniform(3, 6))
        meta = {"src": a_, "t0": t0, "src2": b_, "t02": t1}

    edges = [(idx[a], idx[b], dd["d"], dd["gain"]) for a, b, dd in G.edges(data=True)]
    aN = np.array([math.exp(-1.0 / node_T(cfg, nm)) for nm in names])
    aF = math.exp(-1.0 / cfg.lag_fast)
    s = np.zeros((n, T))
    f = np.zeros((n, T))
    prop = np.zeros((n, T))     # часть входа, пришедшая по схеме (для истинной причинности)
    for t in range(T):
        pr = np.zeros(n)
        for a, b, d, g in edges:
            if t - d >= 0:
                pr[b] += g * s[a, t - d]
        prop[:, t] = pr
        u = ext[:, t] + pr
        sp = s[:, t - 1] if t > 0 else 0.0
        fp = f[:, t - 1] if t > 0 else 0.0
        s[:, t] = aN * sp + (1 - aN) * u
        f[:, t] = aF * fp + (1 - aF) * u

    sig = np.array([tg["sigma"] for tg in TAGS]) * cfg.sigma_scale
    phi = np.array([tg["phi"] if cfg.phi_override is None else cfg.phi_override for tg in TAGS])
    eps = rng.standard_normal((len(TAGS), T))
    noise = np.zeros_like(eps)
    noise[:, 0] = sig * eps[:, 0]
    for t in range(1, T):
        noise[:, t] = phi * noise[:, t - 1] + np.sqrt(1 - phi ** 2) * sig * eps[:, t]
    x = np.zeros((len(TAGS), T))
    for i, tg in enumerate(TAGS):
        base = f[idx[tg["node"]]] if tg["resp"] == "fast" else s[idx[tg["node"]]]
        x[i] = tg["k"] * base + noise[i] + drift[i] + extra[i]
    return x, meta, prop, ext


def detect(x, thr, db, d_on):
    ev, active, cnt, t_on = [], False, 0, 0
    for t, v in enumerate(np.abs(x)):
        if not active:
            if v > thr:
                cnt += 1
                if cnt >= d_on:
                    active, t_on, cnt = True, t, 0
            else:
                cnt = 0
        elif v < thr - db:
            ev.append((t_on, t))
            active = False
    if active:
        ev.append((t_on, len(x)))
    return ev


def make_events(x, cfg, variant):
    ev = []
    for i, tg in enumerate(TAGS):
        if variant == "B0":
            thr, db, d_on = tg["thr0"], 0.0, 1
        else:
            thr, db, d_on = tg["thr1"] * cfg.thr_scale, tg["db"], tg["d_on"]
        for t_on, t_off in detect(x[i], thr, db, d_on):
            ev.append(dict(tag=tg["name"], node=tg["node"], t_on=t_on, t_off=t_off,
                           prio=MATRIX[tg["S"]][tg["Tc"]], visible=True))
    ev.sort(key=lambda e: (e["t_on"], e["tag"]))
    return ev


def apply_suppression(ev, H, cfg, lower=False):
    """lower=True (вариант P2): следствие не может возникнуть раньше суммарной транспортной задержки по пути."""
    dist = dict(nx.floyd_warshall(H, weight="tau"))
    dmin = dict(nx.floyd_warshall(H, weight="d"))
    scc_id = {}
    for i, comp in enumerate(nx.strongly_connected_components(H)):
        for nd in comp:
            scc_id[nd] = i
    for j, ej in enumerate(ev):
        if ej["prio"] >= 3:
            continue
        for ei in ev[:j]:
            if ei["t_off"] <= ej["t_on"] or ei["node"] == ej["node"]:
                continue
            if scc_id[ei["node"]] == scc_id[ej["node"]]:
                continue
            d = dist[ei["node"]][ej["node"]]
            if math.isinf(d):
                continue
            lo = dmin[ei["node"]][ej["node"]] if lower else 0
            if lo <= ej["t_on"] - ei["t_on"] <= d + cfg.k_margin and ei["prio"] >= ej["prio"]:
                ej["visible"] = False
                break
    return ev


def label_conseq(ev, prop, ext, win=30):
    """Истинная причинность из модели: тревога - следствие, если за win мин до срабатывания
    вход узла в среднем приходил в основном по схеме, а не от собственного возмущения."""
    idx = {nm: i for i, nm in enumerate(NODES)}
    for e in ev:
        i = idx[e["node"]]
        a, b = max(0, e["t_on"] - win), e["t_on"] + 1
        e["conseq"] = bool(prop[i, a:b].mean() > ext[i, a:b].mean())
    return ev


def apply_time_suppression(ev, cfg):
    """Базовый вариант B2: группировка по времени без графа - подавить j, если в окне t_window
    активна тревога другого узла не ниже по приоритету."""
    for j, ej in enumerate(ev):
        if ej["prio"] >= 3:
            continue
        for ei in ev[:j]:
            if ei["t_off"] <= ej["t_on"] or ei["node"] == ej["node"]:
                continue
            if ej["t_on"] - ei["t_on"] <= cfg.t_window and ei["prio"] >= ej["prio"]:
                ej["visible"] = False
                break
    return ev


def supp_quality(ev):
    """Доля ложных подавлений (подавлена тревога, не являющаяся следствием) и полнота подавления."""
    sup = [e for e in ev if not e["visible"]]
    con = [e for e in ev if e.get("conseq")]
    false = (sum(1 for e in sup if not e["conseq"]) / len(sup)) if sup else np.nan
    recall = (sum(1 for e in con if not e["visible"]) / len(con)) if con else np.nan
    return false, recall


def metrics(ev, cfg):
    vis = [e for e in ev if e["visible"]]
    nb = cfg.horizon // cfg.bin_min
    cnt = np.zeros(nb)
    for e in vis:
        b = e["t_on"] // cfg.bin_min
        if b < nb:
            cnt[b] += 1
    tot = len(vis)
    top5 = sum(c for _, c in Counter(e["tag"] for e in vis).most_common(5))
    return dict(M1=cnt.mean(), M2=float((cnt > cfg.flood_thr).mean()), M3=float(cnt.max()),
                M4=sum(1 for e in vis if e["t_off"] - e["t_on"] >= cfg.stale_min),
                M5=(top5 / tot) if tot else np.nan,
                n_total=len(ev), n_visible=tot, n_suppressed=len(ev) - tot)


def _root_one(ev, src, t0, cfg):
    c = [e for e in ev if e["node"] == src and t0 <= e["t_on"] <= t0 + cfg.root_window]
    return c[0] if c else None


def root_metric(ev, meta, cfg):
    """M6: видимость тревоги первопричины (для S7 - обеих первопричин), задержка первой и видимость второй."""
    if meta["src"] is None:
        return np.nan, np.nan, np.nan
    e1 = _root_one(ev, meta["src"], meta["t0"], cfg)
    e2 = _root_one(ev, meta["src2"], meta["t02"], cfg) if meta.get("src2") else None
    vis = [float(e["visible"]) for e in (e1, e2) if e is not None]
    if not vis:
        return np.nan, np.nan, np.nan
    delay = (e1["t_on"] - meta["t0"]) if (e1 is not None and e1["visible"]) else np.nan
    second = float(e2["visible"]) if e2 is not None else np.nan
    return float(np.mean(vis)), delay, second


def run_experiment(cfg: Cfg, scenarios=None, design_cfg: Cfg | None = None):
    scenarios = scenarios or DEFAULT_SCENARIOS
    _, G, H = graph_report(cfg)
    if design_cfg is not None:      # проектировщик знает номинальный граф, а не истинный
        H = window_graph(build_graph(design_cfg), cfg.W)
    elif cfg.mis_tau != 1.0 or cfg.mis_lag != 1.0:
        Gm = build_graph(replace(cfg, tau_scale=cfg.tau_scale * cfg.mis_tau, lag_scale=cfg.lag_scale * cfg.mis_lag))
        H = window_graph(Gm, cfg.W)
    rows, prio_rows = [], []
    for si, scn in enumerate(SCENARIOS):
        if scn not in scenarios:
            continue
        for r in range(cfg.runs):
            rng = np.random.default_rng([cfg.seed, si, r])
            x, meta, prop, ext = build_signals(scn, rng, cfg, G)
            ev_b0 = make_events(x, cfg, "B0")
            ev_b1 = label_conseq(make_events(x, cfg, "B1"), prop, ext, win=cfg.label_win)
            ev_p = apply_suppression([dict(e) for e in ev_b1], H, cfg)
            ev_b2 = apply_time_suppression([dict(e) for e in ev_b1], cfg)
            ev_p2 = apply_suppression([dict(e) for e in ev_b1], H, cfg, lower=True)
            for name, ev in (("B0", ev_b0), ("B1", ev_b1), ("B2", ev_b2), ("P", ev_p), ("P2", ev_p2)):
                m = metrics(ev, cfg)
                rv, rd, r2 = root_metric(ev, meta, cfg)
                fs, rc = supp_quality(ev) if name in ("B2", "P", "P2") else (np.nan, np.nan)
                rows.append(dict(scenario=scn, run=r, variant=name, **m, M6_visible=rv, M6_delay=rd,
                                 M6_second=r2, false_supp=fs, supp_recall=rc))
            for e in ev_b1:
                prio_rows.append(dict(scenario=scn, prio=e["prio"]))
    df = pd.DataFrame(rows)
    pr = pd.DataFrame(prio_rows)
    if len(pr):
        pr = pr.groupby(["scenario", "prio"]).size().rename("n").reset_index()
        pr["share"] = pr["n"] / pr.groupby("scenario")["n"].transform("sum")
    return df, pr


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return out
    order = np.argsort(p[ok])
    adj, prev = np.empty(m), 0.0
    for rank, pv in enumerate(p[ok][order]):
        prev = max(prev, min(1.0, (m - rank) * pv))
        adj[rank] = prev
    tmp = np.empty(m)
    tmp[order] = adj
    out[ok] = tmp
    return out


def paired_tests(df, a, b):
    out = []
    for scn, g in df.groupby("scenario"):
        pa = g[g.variant == a].sort_values("run")
        pb = g[g.variant == b].sort_values("run")
        for m in METRICS:
            d = pa[m].values - pb[m].values
            d = d[~np.isnan(d)]
            p = np.nan if (len(d) == 0 or np.allclose(d, 0)) else stats.wilcoxon(d).pvalue
            out.append(dict(scenario=scn, metric=m, A=a, B=b, n=len(d),
                            median_diff=float(np.median(d)) if len(d) else np.nan, p=p))
    t = pd.DataFrame(out)
    t["p_holm"] = holm(t["p"].values)
    return t


def reductions(df):
    g = df.groupby(["scenario", "variant"])[METRICS].mean()
    pct = lambda x, y: (100 * (1 - x / y)) if y else np.nan
    rows = []
    for scn in df.scenario.unique():
        for m in METRICS:
            v = {k: g.loc[(scn, k), m] for k in ("B0", "B1", "B2", "P", "P2")}
            rows.append(dict(scenario=scn, metric=m, **v, P_vs_B1_pct=pct(v["P"], v["B1"]),
                             P2_vs_B1_pct=pct(v["P2"], v["B1"]), B2_vs_B1_pct=pct(v["B2"], v["B1"]),
                             P2_vs_B2_pct=pct(v["P2"], v["B2"]), B1_vs_B0_pct=pct(v["B1"], v["B0"])))
    return pd.DataFrame(rows)


def sensitivity(cfg, runs):
    grid = {"tau_scale": [0.5, 1.0, 2.0], "lag_scale": [0.5, 1.0, 2.0], "k_margin": [0.0, 30.0, 60.0, 120.0],
            "thr_scale": [0.8, 1.0, 1.25], "sigma_scale": [0.75, 1.0, 1.5], "W": [60.0, 120.0, 180.0, 360.0],
            "mis_tau": [0.5, 1.0, 2.0], "mis_lag": [0.5, 1.0, 2.0]}
    rows = []
    for par, vals in grid.items():
        for v in vals:
            df, _ = run_experiment(replace(cfg, runs=runs, **{par: v}), scenarios=["S3", "S6"])
            g = df.groupby(["scenario", "variant"])[["M1", "M2", "M6_visible"]].mean()
            for scn in ("S3", "S6"):
                for var in ("B1", "P"):
                    rows.append(dict(param=par, value=v, scenario=scn, variant=var, **g.loc[(scn, var)].to_dict()))
    return pd.DataFrame(rows)


MC_NODE_T = {"N1": (5, 30), "N2": (5, 30), "N3": (60, 240), "N4": (60, 240), "N5": (360, 540),
             "N6": (84, 180), "N7": (30, 120), "N8": (5, 30)}   # N3-N6, N5: диапазоны из источников (табл. 1)


def sample_true_cfg(base: Cfg, rng):
    """Истинные параметры процесса: все допущения разыгрываются одновременно."""
    for _ in range(200):
        cfg = replace(
            base,
            edge_d_mult=tuple(float(np.exp(rng.uniform(np.log(0.5), np.log(2.0)))) for _ in EDGES),
            edge_g_mult=tuple(float(rng.uniform(0.5, 1.5)) for _ in EDGES),
            node_T=tuple(float(rng.uniform(*MC_NODE_T[nm])) for nm in NODES),
            sigma_scale=float(rng.uniform(0.7, 1.4)), phi_override=float(rng.uniform(0.8, 0.95)),
            amp_scale=float(rng.uniform(0.7, 1.3)))
        try:
            graph_report(cfg)
            return cfg
        except ValueError:
            continue
    raise RuntimeError("не удалось подобрать устойчивые параметры")


def monte_carlo(cfg: Cfg, n_rep: int, runs: int):
    rows = []
    scns = ["S2", "S3", "S4", "S6", "S7", "S8"]
    for r in range(n_rep):
        rng = np.random.default_rng([cfg.seed, 999, r])
        true = replace(sample_true_cfg(replace(cfg, runs=runs, seed=cfg.seed + 1000 + r), rng))
        df, _ = run_experiment(true, scenarios=scns, design_cfg=replace(cfg))
        g = df.groupby(["scenario", "variant"]).agg(M1=("M1", "mean"), M3=("M3", "mean"), fs=("false_supp", "mean"),
                                                    rc=("supp_recall", "mean"), m6=("M6_visible", "mean"),
                                                    m62=("M6_second", "mean"))
        for scn in scns:
            b1 = g.loc[(scn, "B1")]
            pct = lambda v, k: (100 * (1 - g.loc[(scn, v), k] / b1[k])) if b1[k] else np.nan
            row = dict(rep=r, scenario=scn)
            for v in ("P", "P2", "B2"):
                row.update({f"redM1_{v}": pct(v, "M1"), f"redM3_{v}": pct(v, "M3"),
                            f"fs_{v}": g.loc[(scn, v), "fs"], f"rc_{v}": g.loc[(scn, v), "rc"],
                            f"m6_{v}": g.loc[(scn, v), "m6"], f"m62_{v}": g.loc[(scn, v), "m62"]})
            rows.append(row)
    return pd.DataFrame(rows)


def write_param_tables(out: Path):
    pd.DataFrame([dict(node=k, T_min=v[0], source=v[1]) for k, v in NODES.items()]).to_csv(out / "params_nodes.csv", index=False)
    pd.DataFrame([dict(src=a, dst=b, d_min=d, gain=g, kind=k, source=s) for a, b, d, g, k, s in EDGES]).to_csv(out / "params_edges.csv", index=False)
    pd.DataFrame([{k: v for k, v in tg.items()} for tg in TAGS]).to_csv(out / "params_tags.csv", index=False)


def cascade_figure(cfg, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    _, G, H = graph_report(cfg)
    rng = np.random.default_rng([cfg.seed, SCENARIOS.index("S3"), 0])
    x, _, _, _ = build_signals("S3", rng, cfg, G)
    b1 = make_events(x, cfg, "B1")
    sets = {"B0": make_events(x, cfg, "B0"), "B1": b1, "P": apply_suppression([dict(e) for e in b1], H, cfg)}
    nb = cfg.horizon // cfg.bin_min
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for name, ev in sets.items():
        c = np.zeros(nb)
        for e in ev:
            if e["visible"] and e["t_on"] // cfg.bin_min < nb:
                c[e["t_on"] // cfg.bin_min] += 1
        ax.step(np.arange(nb) * cfg.bin_min / 60, c, where="post", label=name)
    ax.set_xlabel("Время, ч")
    ax.set_ylabel(f"Тревог за {cfg.bin_min} мин")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--horizon", type=int, default=2880)
    ap.add_argument("--W", type=float, default=120.0)
    ap.add_argument("--k", type=float, default=30.0, dest="k_margin")
    ap.add_argument("--out", default="results")
    ap.add_argument("--sens", action="store_true")
    ap.add_argument("--sens-runs", type=int, default=30)
    ap.add_argument("--mc", type=int, default=0, help="число реплик Монте-Карло")
    ap.add_argument("--mc-runs", type=int, default=3)
    a = ap.parse_args()
    cfg = Cfg(runs=a.runs, seed=a.seed, horizon=a.horizon, W=a.W, k_margin=a.k_margin)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    text, _, _ = graph_report(cfg)
    print(text)
    (out / "graph_report.txt").write_text(text, encoding="utf-8")
    (out / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    write_param_tables(out)
    df, pr = run_experiment(cfg)
    df.to_csv(out / "runs.csv", index=False)
    pr.to_csv(out / "priorities_B1.csv", index=False)
    summ = df.groupby(["scenario", "variant"])[METRICS + ["M6_visible", "M6_delay", "n_total", "n_visible", "n_suppressed", "false_supp", "supp_recall", "M6_second"]].agg(["mean", "std"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.to_csv(out / "summary.csv")
    reductions(df).to_csv(out / "reductions.csv", index=False)
    pd.concat([paired_tests(df, "P", "B1"), paired_tests(df, "B2", "B1"), paired_tests(df, "P2", "B1"), paired_tests(df, "P2", "B2"),
               paired_tests(df, "B1", "B0")]).to_csv(out / "tests.csv", index=False)
    cascade_figure(cfg, out / "fig_cascade.png")
    if a.sens:
        sensitivity(cfg, a.sens_runs).to_csv(out / "sensitivity.csv", index=False)
    if a.mc:
        monte_carlo(cfg, a.mc, a.mc_runs).to_csv(out / "montecarlo.csv", index=False)
    print(f"Готово: {out.resolve()}")


if __name__ == "__main__":
    main()
