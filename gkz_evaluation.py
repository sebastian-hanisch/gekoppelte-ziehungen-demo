"""Kennzahlen, Verdikt-Logik und Experimente über feste Ströme (reine Funktionen, kein Streamlit).

Zentrale Größe ist der Varianzfaktor eines gespiegelten Paares gegen zwei unabhängige Auswertungen: 1 + rho, mit rho der Korrelation
von X(u) und X(1-u). Bei den gemeinsamen Zufallszahlen ist es das Verhältnis der Varianzen einer Differenz bei unabhängigen und gekoppelten Ziehungen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import gkz_algorithm as A
import gkz_constants as C
from gkz_scenario import Netz, extra_edge, named_netz, with_edge


def de(x: float, nd: int = 2) -> str:
    """Zahl im deutschen Format: Punkt als Tausendertrenner, Komma als Dezimalzeichen, echtes Minuszeichen."""
    return f"{x:,.{nd}f}".translate(str.maketrans(",.", ".,")).replace("-", "−")


def de_e(x: float) -> str:
    """Wissenschaftliche Schreibweise mit Dezimalkomma, z. B. 4,2e-10."""
    return f"{x:.1e}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    gitter: str = C.DEFAULT_GRID
    p: float = C.DEFAULT_P
    n: int = C.DEFAULT_N
    ziel: str = C.DEFAULT_TARGET
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    netz: Netz
    truth: float                 # exakte Wahrscheinlichkeit des Ziels (Aufzählung)
    mean_plain: float            # Mittel der Schätzungen über die Wiederholungen
    mean_anti: float
    sd_plain: float              # Streuung der Schätzung bei N unabhängigen Auswertungen
    sd_anti: float               # Streuung bei N/2 gespiegelten Paaren (gleich viele Auswertungen)
    var_ratio: float             # gemessen: Var(gespiegelt) / Var(unabhängig)
    rho: float                   # gepoolt aus den gezogenen Paaren
    rho_exact: float | None      # exakte Korrelation (3^m-Aufzählung), nur bis EXACT_PAIR_EDGE_LIMIT Kanten
    plain: tuple[float, ...]
    anti: tuple[float, ...]

    @property
    def predicted_ratio(self) -> float:
        """Das aus der Korrelation vorhergesagte Varianzverhältnis 1 + rho (exakt, wenn die exakte Korrelation vorliegt)."""
        return 1.0 + (self.rho_exact if self.rho_exact is not None else self.rho)


# --- Wahrheit -----------------------------------------------------------------------------------------------------------------------------

def netz_counts(netz: Netz, limit: int = C.LIVE_EDGE_LIMIT) -> dict:
    """N_k je Ziel durch Aufzählung (höchstens `limit` Kanten; der Entwurfsvergleich erlaubt C.DESIGN_EDGE_LIMIT)."""
    if netz.m > limit:
        raise ValueError(f"{netz.m} Kanten: aufgezählt wird nur bis {limit}")
    return {k: tuple(int(x) for x in v) for k, v in A.target_counts(netz.n, netz.edges).items()}


def exact_rho(netz: Netz, name: str, p: float) -> float | None:
    """Exakte Korrelation von X(u) und X(1-u) (3^m-Aufzählung) oder None, wenn das Netz dafür zu groß ist."""
    if netz.m > C.EXACT_PAIR_EDGE_LIMIT:
        return None
    return A.rho_from_moments(A.exact_pair_moments(name, netz.n, netz.edges, p))


# --- Hauptanalyse -------------------------------------------------------------------------------------------------------------------------

def analyse(settings: Settings, netz: Netz | None = None, counts: dict | None = None) -> Analysis:
    """REPS Schätzungen mit N unabhängigen Auswertungen gegen REPS Schätzungen mit N/2 gespiegelten Paaren, gleicher Aufwand."""
    netz = netz if netz is not None else named_netz(settings.gitter)
    counts = counts if counts is not None else netz_counts(netz)
    sp = A.spread(settings.seed, C.REPS, settings.n, settings.p, settings.ziel, netz.n, netz.edges)
    plain, anti = sp["plain"], sp["anti"]
    var_plain, var_anti = float(plain.var(ddof=1)), float(anti.var(ddof=1))
    ratio = var_anti / var_plain if var_plain > 0 else float("nan")
    return Analysis(
        netz, A.probability_from_counts(counts[settings.ziel], settings.p), float(plain.mean()), float(anti.mean()),
        float(np.sqrt(var_plain)), float(np.sqrt(var_anti)), ratio, sp["rho"], exact_rho(netz, settings.ziel, settings.p),
        tuple(float(x) for x in plain), tuple(float(x) for x in anti),
    )


def verdict_spread(a: Analysis, n: int) -> tuple[str, str]:
    """Senkt die Spiegelung die Streuung, und stimmt das gemessene Verhältnis mit der Vorhersage aus der Korrelation überein?"""
    pred = a.predicted_ratio
    rho = a.rho_exact if a.rho_exact is not None else a.rho
    quelle = "exakt" if a.rho_exact is not None else "aus den Paaren geschätzt"
    kopf = (f"Bei {de(n, 0)} Auswertungen streut die gespiegelte Schätzung mit {de(a.sd_anti, 5)} statt {de(a.sd_plain, 5)} "
            f"(Varianzverhältnis {de(a.var_ratio, 2)} gemessen, {de(pred, 2)} vorhergesagt: Korrelation ρ = {de(rho, 3)}, {quelle}).")
    if pred < 0.95:
        return "success", kopf + f" Die Spiegelung senkt die Varianz um {de((1 - pred) * 100, 0)} %."
    if pred > 1.05:
        return "warning", kopf + f" Die Spiegelung ERHÖHT die Varianz um {de((pred - 1) * 100, 0)} %: das Ziel ist nicht monoton."
    return "info", kopf + " Die Spiegelung bringt hier praktisch nichts."


# --- Experiment 1: Streuung gegen N -----------------------------------------------------------------------------------------------------------

def n_sweep(settings: Settings, netz: Netz) -> dict:
    rows = []
    for N in C.SWEEP_N:
        sp = A.spread(settings.seed, C.SWEEP_REPS, N, settings.p, settings.ziel, netz.n, netz.edges)
        vp, va = float(sp["plain"].var(ddof=1)), float(sp["anti"].var(ddof=1))
        rows.append({"N": N, "sd_plain": float(np.sqrt(vp)), "sd_anti": float(np.sqrt(va)), "ratio": va / vp, "rho": sp["rho"]})
    ns = np.log([r["N"] for r in rows])
    return {
        "rows": rows,
        "slope_plain": float(np.polyfit(ns, np.log([r["sd_plain"] for r in rows]), 1)[0]),
        "slope_anti": float(np.polyfit(ns, np.log([r["sd_anti"] for r in rows]), 1)[0]),
        "rho_exact": exact_rho(netz, settings.ziel, settings.p),
    }


def verdict_n_sweep(exp: dict) -> tuple[str, str]:
    ratios = [r["ratio"] for r in exp["rows"]]
    text = (f"Das Varianzverhältnis liegt bei allen N zwischen {de(min(ratios), 2)} und {de(max(ratios), 2)}; die Steigung der Streuung in der doppelt-logarithmischen "
            f"Auftragung ist {de(exp['slope_plain'], 2)} (unabhängig) und {de(exp['slope_anti'], 2)} (gespiegelt), Soll −0,50. Die Spiegelung ändert die "
            f"Konstante, nicht die Rate. Das Rauschen der Verhältnisse bei {C.SWEEP_REPS} Wiederholungen beträgt rund {de(200 / np.sqrt(C.SWEEP_REPS - 1), 0)} %.")
    return "info", text


# --- Experiment 2: Korrelation gegen p ------------------------------------------------------------------------------------------------------------

def rho_curve(settings: Settings, netz: Netz) -> dict:
    """Varianzfaktor 1 + rho gegen p: exakt für Netze bis EXACT_PAIR_EDGE_LIMIT Kanten, sonst aus RHO_PAIRS Paaren je p."""
    rows = []
    for p in C.RHO_PS:
        rho = exact_rho(netz, settings.ziel, p)
        exakt = rho is not None
        if not exakt:
            xa, xb = A.pair_values(A.stream_seed(22, netz.m, int(p * 1000)), C.RHO_PAIRS, p, settings.ziel, netz.n, netz.edges)
            rho = float(np.corrcoef(xa, xb)[0, 1])
        rows.append({"p": p, "rho": rho, "factor": 1.0 + rho, "exakt": exakt})
    return {"rows": rows}


def verdict_rho_curve(exp: dict, name: str) -> tuple[str, str]:
    rows = exp["rows"]
    best = min(rows, key=lambda r: r["factor"])
    worst = max(rows, key=lambda r: r["factor"])
    last = rows[-1]
    text = (f"Der Varianzfaktor ist am kleinsten bei p = {de(best['p'], 2)} ({de(best['factor'], 3)}) und am größten bei p = {de(worst['p'], 2)} ({de(worst['factor'], 3)}); "
            f"bei p = {de(last['p'], 2)} (seltene Ausfälle) ist er {de(last['factor'], 4)}.")
    if name == "zusammenhaengend":
        text += " Für ein monotones Ziel ist ρ ≤ 0 garantiert; groß ist der Gewinn nur in der Mitte."
    return "info", text


# --- Experiment 3: welche Funktion? ------------------------------------------------------------------------------------------------------------

SCALAR_LABELS = {"quadrat": "u² (Skalar, monoton, glatt)", "symmetrisch": "(2u−1)² (Skalar, symmetrisch um ½)"}


def function_table(settings: Settings, netz: Netz) -> dict:
    """Korrelation und Varianzfaktor für vier Funktionen derselben Zufallszahlen: zwei Netzziele und zwei Skalare."""
    rows = []
    for name, label, monoton in (("zusammenhaengend", "Netz hängt zusammen (monoton)", True), ("zwei_teile", "Netz zerfällt in genau zwei Teile (nicht monoton)", False)):
        rho = exact_rho(netz, name, settings.p)
        exakt = rho is not None
        if not exakt:
            xa, xb = A.pair_values(A.stream_seed(33, netz.m, int(settings.p * 1000)), C.FUNC_PAIRS, settings.p, name, netz.n, netz.edges)
            rho = float(np.corrcoef(xa, xb)[0, 1]) if xa.std() > 0 else float("nan")
        rows.append({"name": label, "monoton": monoton, "rho": rho, "factor": 1.0 + rho, "exakt": exakt})
    for key, monoton in (("quadrat", True), ("symmetrisch", False)):
        rho = A.scalar_rho(key, A.stream_seed(34, len(key)), C.SCALAR_DRAWS)
        rows.append({"name": SCALAR_LABELS[key], "monoton": monoton, "rho": rho, "factor": 1.0 + rho, "exakt": False})
    return {"rows": rows}


def verdict_functions(exp: dict) -> tuple[str, str]:
    rows = exp["rows"]
    schlecht = [r for r in rows if r["factor"] > 1.0 + 1e-9]
    teile = "; ".join(f"{r['name']}: {de(r['factor'], 3)}" for r in rows)
    if schlecht:
        namen = ", ".join(r["name"] for r in schlecht)
        return "warning", f"Varianzfaktoren (unter 1 ist ein Gewinn): {teile}. Über 1 liegt: {namen}."
    return "info", f"Varianzfaktoren (unter 1 ist ein Gewinn): {teile}. Keine der vier Funktionen wird bei diesem p schlechter."


# --- Experiment 4: gemeinsame Zufallszahlen -------------------------------------------------------------------------------------------------------

def crn_experiment(settings: Settings, netz: Netz, counts: dict) -> dict:
    """Differenz von X(p + delta) und X(p): Gewinnfaktor gegen delta (Formel und Messung) und Entwurfsvergleich mit einer Zusatzkante."""
    conn = counts["zusammenhaengend"]
    r0 = A.probability_from_counts(conn, settings.p)
    rows = []
    for delta in C.CRN_DELTAS:
        p1 = min(settings.p + delta, C.P_CAP)
        r1 = A.probability_from_counts(conn, p1)
        dr = r1 - r0
        var_crn, var_ind = dr * (1 - dr), r0 * (1 - r0) + r1 * (1 - r1)
        meas = A.difference_experiment(A.stream_seed(44, netz.m, int(delta * 1000)), C.CRN_DRAWS, settings.p, p1, netz.n, netz.edges)
        rows.append({"delta": delta, "dR": dr, "var_crn": var_crn, "var_ind": var_ind, "factor": var_ind / var_crn if var_crn > 0 else float("nan"),
                     "var_crn_meas": meas["var_crn"], "var_ind_meas": meas["var_ind"], "factor_meas": meas["var_ind"] / meas["var_crn"] if meas["var_crn"] > 0 else float("nan")})
    return {"rows": rows, "R": r0, "design": design_comparison(settings, netz, counts)}


def design_comparison(settings: Settings, netz: Netz, counts: dict, extra: tuple[int, int] | None = None) -> dict:
    """Netz mit einer Zusatzkante gegen das Netz ohne sie, mit gemeinsamen Zufallszahlen (feste Zuordnung je Ziehung und Kante)."""
    extra = extra if extra is not None else extra_edge(settings.gitter)
    netz_b = with_edge(netz, extra)
    counts_b = netz_counts(netz_b, C.DESIGN_EDGE_LIMIT)
    ra = A.probability_from_counts(counts["zusammenhaengend"], settings.p)
    rb = A.probability_from_counts(counts_b["zusammenhaengend"], settings.p)
    dr = rb - ra
    meas = A.design_experiment(A.stream_seed(45, netz.m), C.CRN_DRAWS, settings.p, netz.n, netz.edges, netz_b.edges)
    var_crn, var_ind = dr * (1 - dr), ra * (1 - ra) + rb * (1 - rb)
    return {"edge": extra, "R_a": ra, "R_b": rb, "dR": dr, "var_crn": var_crn, "var_ind": var_ind, "factor": var_ind / var_crn if var_crn > 0 else float("nan"),
            "var_crn_meas": meas["var"], "corr": meas["corr"], "mean_meas": meas["mean"]}


def verdict_crn(exp: dict) -> tuple[str, str]:
    rows = exp["rows"]
    first, last = rows[0], rows[-1]
    worst = max(abs(r["var_crn_meas"] / r["var_crn"] - 1) for r in rows)
    d = exp["design"]
    text = (f"Der Gewinn hängt am Abstand der Varianten: bei δ = {de(first['delta'], 2)} ist die Differenz {de(first['factor'], 1)}-mal ruhiger, bei δ = {de(last['delta'], 2)} {de(last['factor'], 1)}-mal. "
            f"Die Formel Var(X_B − X_A) = ΔR(1−ΔR) trifft die Messung auf höchstens {de(worst * 100, 1)} % genau. "
            f"Mit einer Zusatzkante {d['edge'][0]}–{d['edge'][1]} (ΔR = {de(d['dR'], 4)}) ist der Gewinn {de(d['factor'], 1)}.")
    return "success", text


# --- Experiment 5: Synchronisation --------------------------------------------------------------------------------------------------------------

def sync_result(settings: Settings, netz: Netz) -> dict:
    """Richtige (feste Schrittweite) gegen falsche (Schrittweite = eigene Kantenzahl) Zuordnung der Zufallszahlen beim Entwurfsvergleich."""
    netz_b = with_edge(netz, extra_edge(settings.gitter))
    res = A.sync_experiment(A.stream_seed(55, netz.m), C.CRN_DRAWS, settings.p, netz.n, netz.edges, netz_b.edges)
    counts_a, counts_b = netz_counts(netz), netz_counts(netz_b, C.DESIGN_EDGE_LIMIT)
    ra = A.probability_from_counts(counts_a["zusammenhaengend"], settings.p)
    rb = A.probability_from_counts(counts_b["zusammenhaengend"], settings.p)
    return {"fest": res["fest"], "lage": res["lage"], "dR": rb - ra, "var_ind": ra * (1 - ra) + rb * (1 - rb), "edge": extra_edge(settings.gitter)}


def verdict_sync(exp: dict) -> tuple[str, str]:
    fest, lage = exp["fest"], exp["lage"]
    text = (f"Mit fester Zuordnung (Zufallszahl Nr. i·(m+1) + e in beiden Varianten) ist die Korrelation der beiden Ziele {de(fest['corr'], 3)} und die Varianz der Differenz {de(fest['var'], 4)}. "
            f"Liest jede Variante den Strom mit ihrer eigenen Kantenzahl als Schrittweite, verschiebt sich alles: Korrelation {de(lage['corr'], 3)}, Varianz {de(lage['var'], 4)} "
            f"(unabhängig wäre {de(exp['var_ind'], 4)}), also {de(lage['var'] / fest['var'], 1)}-mal so groß. Der Mittelwert der Differenz bleibt unverzerrt ({de(lage['mean'], 4)} gegen {de(fest['mean'], 4)}); nur die Streuung verrät den Fehler.")
    return "warning", text


# --- Spielwiese ---------------------------------------------------------------------------------------------------------------------------

def custom_result(settings: Settings, netz: Netz, extra: tuple[int, int] | None = None) -> dict:
    """Ein eigenes Netz: Wahrheit, Korrelation und Varianzfaktor der Spiegelung (exakt bis EXACT_PAIR_EDGE_LIMIT Kanten, sonst aus Paaren) und,
    wenn eine Zusatzkante angegeben ist, der Entwurfsvergleich mit gemeinsamen Zufallszahlen."""
    counts = netz_counts(netz)
    truth = A.probability_from_counts(counts[settings.ziel], settings.p)
    rho = exact_rho(netz, settings.ziel, settings.p)
    exakt = rho is not None
    if not exakt:
        xa, xb = A.pair_values(A.stream_seed(66, netz.m), C.FUNC_PAIRS // 2, settings.p, settings.ziel, netz.n, netz.edges)
        rho = float(np.corrcoef(xa, xb)[0, 1]) if xa.std() > 0 else float("nan")
    out = {"truth": truth, "rho": rho, "factor": 1.0 + rho, "exakt": exakt}
    if extra is not None:
        out["design"] = design_comparison(settings, netz, counts, extra)
    return out
