"""Kern der Demo (numpy; scipy wird hier nicht gebraucht).

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b):
  Zufallsstrom       stream_seed, uniforms, edge_uniforms        SplitMix64, zählerbasiert; Zahl (Ziehung i, Kante e) ist Nr. i*stride + e des Stroms
  Ziele              connected, components, target_values        "Netz hängt zusammen" (monoton), "genau zwei Teile" (nicht monoton)
  Wahrheit           target_counts, probability_from_counts      Aufzählung aller 2^m Kantenmengen
  Paare              pair_values, exact_pair_moments             Kante e hält im ersten Netz bei u_e < p, im gespiegelten bei u_e > 1 - p
  Schätzer           spread                                      N unabhängige Auswertungen gegen N/2 gespiegelte Paare, über feste Ströme
  Kopplung           difference_experiment, sync_experiment      gleiche Zufallszahlen für zwei Varianten, richtig und falsch zugeordnet
  Skalare            scalar_rho                                  f(u) und f(1 - u) für einfache Funktionen
Zufall kommt nur über einen übergebenen Strom (seed), nie aus einem globalen Generator.
"""

from __future__ import annotations

import numpy as np

GAMMA = 0x9E3779B97F4A7C15
MASK = (1 << 64) - 1


# --- Zufallsstrom -------------------------------------------------------------------------------------------------------------------------

def mix_int(z: int) -> int:
    """SplitMix64-Mischfunktion mit Python-Ganzzahlen."""
    z &= MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return (z ^ (z >> 31)) & MASK


def _mix_arr(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def stream_seed(*keys: int) -> int:
    """Seed eines Stroms aus einem Schlüsseltupel (Einstellung, Wiederholung, ...)."""
    s = 0x1234_5678_9ABC_DEF0
    for k in keys:
        s = mix_int(s ^ mix_int(int(k) + GAMMA))
    return s


def uniforms(seed: int, count: int) -> np.ndarray:
    """`count` gleichverteilte Zahlen in [0, 1) (53 Bit) aus dem Strom `seed`; Ausgabe i hängt nur von (seed, i) ab."""
    idx = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = _mix_arr(np.uint64(seed) + idx * np.uint64(GAMMA))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def edge_uniforms(seed: int, N: int, m: int, stride: int | None = None) -> np.ndarray:
    """(N, m) Zufallszahlen: die Zahl der Kante e in Ziehung i ist Nr. i*stride + e des Stroms.

    `stride` (Standard m) ist die Zahl der Zahlen, die der Strom je Ziehung reserviert. Zwei Varianten teilen sich ihre Zufallszahlen nur dann,
    wenn sie DENSELBEN Strom mit DERSELBEN Schrittweite lesen (Synchronisation)."""
    stride = m if stride is None else stride
    if stride < m:
        raise ValueError("stride muss mindestens m sein")
    return uniforms(seed, N * stride).reshape(N, stride)[:, :m]


# --- Ziele ------------------------------------------------------------------------------------------------------------------------------

def _labels(present: np.ndarray, n: int, edges) -> np.ndarray:
    """Kleinste-Marke-Ausbreitung: Marke je Knoten und Zeile; Knoten mit gleicher Marke liegen in derselben Komponente."""
    B = present.shape[0]
    lab = np.tile(np.arange(n, dtype=np.int16), (B, 1))
    while True:
        before = int(lab.sum())
        for e, (u, v) in enumerate(edges):
            low = np.minimum(lab[:, u], lab[:, v])
            pe = present[:, e]
            lab[:, u] = np.where(pe, low, lab[:, u])
            lab[:, v] = np.where(pe, low, lab[:, v])
        if int(lab.sum()) == before:
            return lab


def components(present: np.ndarray, n: int, edges) -> np.ndarray:
    """Zahl der Zusammenhangskomponenten je Zeile (`present` ist (B, m) bool); Wurzeln sind Knoten, deren Marke ihr eigener Index ist."""
    if n == 0:
        return np.zeros(present.shape[0], dtype=int)
    return (_labels(present, n, edges) == np.arange(n)).sum(axis=1)


def connected(present: np.ndarray, n: int, edges) -> np.ndarray:
    """Zusammenhang je Zeile: genau eine Komponente."""
    return components(present, n, edges) <= 1


def target_values(name: str, present: np.ndarray, n: int, edges) -> np.ndarray:
    """0/1-Wert des Ziels je Zeile: "zusammenhaengend" (monoton in den haltenden Kanten) oder "zwei_teile" (genau zwei Komponenten, NICHT monoton)."""
    comp = components(present, n, edges)
    if name == "zusammenhaengend":
        return (comp <= 1).astype(np.float64)
    if name == "zwei_teile":
        return (comp == 2).astype(np.float64)
    raise ValueError(f"unbekanntes Ziel {name!r}")


# --- Wahrheit ----------------------------------------------------------------------------------------------------------------------------

def target_counts(n: int, edges, chunk: int = 1 << 18) -> dict:
    """Für beide Ziele die Zahl N_k der Kantenmengen mit k Kanten, in denen das Ziel eintritt (Aufzählung aller 2^m Mengen)."""
    m = len(edges)
    counts = {name: np.zeros(m + 1, dtype=np.int64) for name in ("zusammenhaengend", "zwei_teile")}
    shifts = np.arange(m, dtype=np.uint32)
    for start in range(0, 1 << m, chunk):
        masks = np.arange(start, min(start + chunk, 1 << m), dtype=np.uint32)
        present = ((masks[:, None] >> shifts) & 1).astype(bool)
        comp = components(present, n, edges)
        k = present.sum(axis=1)
        counts["zusammenhaengend"] += np.bincount(k[comp <= 1], minlength=m + 1)
        counts["zwei_teile"] += np.bincount(k[comp == 2], minlength=m + 1)
    return counts


def probability_from_counts(counts, p: float) -> float:
    """P(Ziel) = Summe_k N_k p^k (1-p)^(m-k), wenn jede Kante unabhängig mit p hält."""
    counts = np.asarray(counts, dtype=np.float64)
    k = np.arange(len(counts))
    return float(np.sum(counts * p**k * (1.0 - p) ** (len(counts) - 1 - k)))


# --- Paare -------------------------------------------------------------------------------------------------------------------------------

def pair_values(seed: int, N: int, p: float, name: str, n: int, edges) -> tuple[np.ndarray, np.ndarray]:
    """Zielwerte X(u) und X(1-u) für N Zufallsvektoren u: Kante e hält im ersten Netz bei u_e < p, im gespiegelten bei u_e > 1 - p."""
    u = edge_uniforms(seed, N, len(edges))
    return target_values(name, u < p, n, edges), target_values(name, u > 1.0 - p, n, edges)


def exact_pair_moments(name: str, n: int, edges, p: float) -> dict:
    """E X, E X', E X X' EXAKT durch Aufzählung der 3^m Kategorien je Kante (hält im ersten / im zweiten / in beiden / in keinem Netz).

    Für p > 0,5: (hält, hält) mit 2p-1, (hält, nicht) mit 1-p, (nicht, hält) mit 1-p, (nicht, nicht) unmöglich.
    Für p <= 0,5: (nicht, nicht) mit 1-2p, (hält, nicht) mit p, (nicht, hält) mit p, (hält, hält) unmöglich."""
    m = len(edges)
    cats = [((1, 1), 2 * p - 1), ((1, 0), 1 - p), ((0, 1), 1 - p)] if p > 0.5 else [((0, 0), 1 - 2 * p), ((1, 0), p), ((0, 1), p)]
    probs = np.array([c[1] for c in cats])
    idx = np.arange(3**m, dtype=np.int64)
    a = np.zeros((3**m, m), dtype=bool)
    b = np.zeros((3**m, m), dtype=bool)
    weight = np.ones(3**m)
    for e in range(m):
        d = (idx // 3**e) % 3
        for c, ((ha, hb), _) in enumerate(cats):
            a[d == c, e] = bool(ha)
            b[d == c, e] = bool(hb)
        weight *= probs[d]
    xa, xb = target_values(name, a, n, edges), target_values(name, b, n, edges)
    return {"EX": float(np.sum(weight * xa)), "EXp": float(np.sum(weight * xb)), "EXXp": float(np.sum(weight * xa * xb)), "mass": float(weight.sum())}


def rho_from_moments(mo: dict) -> float:
    """Korrelation von X(u) und X(1-u) aus E X, E X', E X X' (beide Varianzen sind gleich: die Verteilungen von X und X' sind es)."""
    var = mo["EX"] * (1 - mo["EX"])
    return (mo["EXXp"] - mo["EX"] * mo["EXp"]) / var if var > 0 else float("nan")


def path_rho(m: int, p: float) -> float:
    """Geschlossene Korrelation für einen Pfad mit m Kanten (Reihenschaltung, p > 0,5): ((2p-1)^m - p^(2m)) / (p^m (1-p^m))."""
    return ((2 * p - 1) ** m - p ** (2 * m)) / (p**m * (1 - p**m))


# --- Schätzer: unabhängig gegen gespiegelt ------------------------------------------------------------------------------------------------------

def spread(seed: int, reps: int, N: int, p: float, name: str, n: int, edges) -> dict:
    """Je Wiederholung eine Schätzung aus N unabhängigen Auswertungen und eine aus N/2 gespiegelten Paaren (gleich viele Auswertungen).

    Rückgabe: die beiden Schätzungsreihen und die über alle Paare gepoolte Korrelation von X(u) und X(1-u)."""
    if N % 2:
        raise ValueError("N muss gerade sein: ein gespiegeltes Paar zählt zwei Auswertungen")
    half = N // 2
    m = len(edges)
    plain, anti = np.empty(reps), np.empty(reps)
    sums = np.zeros(5)      # Summe xa, xb, xa^2, xb^2, xa*xb über alle Paare
    for rep in range(reps):
        up = edge_uniforms(stream_seed(1, seed, N, rep), N, m)
        plain[rep] = target_values(name, up < p, n, edges).mean()
        ua = edge_uniforms(stream_seed(2, seed, N, rep), half, m)
        xa, xb = target_values(name, ua < p, n, edges), target_values(name, ua > 1.0 - p, n, edges)
        anti[rep] = 0.5 * (xa.mean() + xb.mean())
        sums += (xa.sum(), xb.sum(), (xa * xa).sum(), (xb * xb).sum(), (xa * xb).sum())
    cnt = reps * half
    ma, mb = sums[0] / cnt, sums[1] / cnt
    cov = sums[4] / cnt - ma * mb
    va, vb = sums[2] / cnt - ma * ma, sums[3] / cnt - mb * mb
    rho = float(cov / np.sqrt(va * vb)) if va > 0 and vb > 0 else float("nan")
    return {"plain": plain, "anti": anti, "rho": rho}


# --- Kopplung zweier Varianten --------------------------------------------------------------------------------------------------------------

def difference_experiment(seed: int, N: int, p_a: float, p_b: float, n: int, edges) -> dict:
    """Differenz X_B - X_A für zwei Haltewahrscheinlichkeiten (p_b > p_a) bei GEMEINSAMEN gegen UNABHÄNGIGEN Zufallszahlen.

    Gleiche Zufallszahlen: B hält mindestens dort, wo A hält, die Differenz liegt in {0, 1}. Rückgabe: Mittel und Varianz je Ziehung."""
    m = len(edges)
    u = edge_uniforms(seed, N, m)
    u_other = edge_uniforms(stream_seed(seed, 1), N, m)
    xa = target_values("zusammenhaengend", u < p_a, n, edges)
    d_crn = target_values("zusammenhaengend", u < p_b, n, edges) - xa
    d_ind = target_values("zusammenhaengend", u_other < p_b, n, edges) - xa
    return {"mean_crn": float(d_crn.mean()), "var_crn": float(d_crn.var(ddof=1)), "var_ind": float(d_ind.var(ddof=1)), "mean_ind": float(d_ind.mean())}


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    """Korrelation zweier 0/1-Reihen; NaN (ohne Warnung), wenn eine Reihe konstant ist."""
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def design_experiment(seed: int, N: int, p: float, n: int, edges_a, edges_b) -> dict:
    """Differenz X_B - X_A, wenn B dieselben Kanten wie A plus Kanten am Ende der Liste hat; die Zufallszahlen sind je (Ziehung, Kante) fest."""
    ma, mb = len(edges_a), len(edges_b)
    ub = edge_uniforms(seed, N, mb)
    ua = ub[:, :ma]
    xa = target_values("zusammenhaengend", ua < p, n, edges_a)
    xb = target_values("zusammenhaengend", ub < p, n, edges_b)
    d = xb - xa
    return {"mean": float(d.mean()), "var": float(d.var(ddof=1)), "corr": _corr(xa, xb)}


def sync_experiment(seed: int, N: int, p: float, n: int, edges_a, edges_b) -> dict:
    """Dasselbe wie design_experiment, aber mit der Zuordnung RICHTIG (feste Schrittweite je Ziehung) und FALSCH (jede Variante liest den Strom
    mit ihrer eigenen Kantenzahl als Schrittweite: bei m + 1 Kanten verschiebt sich alles)."""
    ma, mb = len(edges_a), len(edges_b)
    out = {}
    for label, stride_a in (("fest", mb), ("lage", ma)):
        ua = edge_uniforms(seed, N, ma, stride=stride_a)
        ub = edge_uniforms(seed, N, mb)
        xa = target_values("zusammenhaengend", ua < p, n, edges_a)
        xb = target_values("zusammenhaengend", ub < p, n, edges_b)
        d = xb - xa
        out[label] = {"mean": float(d.mean()), "var": float(d.var(ddof=1)), "corr": _corr(xa, xb)}
    return out


# --- Skalare Funktionen ------------------------------------------------------------------------------------------------------------------

SCALARS = {"quadrat": lambda u: u**2, "symmetrisch": lambda u: (2.0 * u - 1.0) ** 2}


def scalar_rho(name: str, seed: int, N: int) -> float:
    """Korrelation von f(u) und f(1-u) für u gleichverteilt, aus N Ziehungen."""
    u = uniforms(seed, N)
    f = SCALARS[name]
    return float(np.corrcoef(f(u), f(1.0 - u))[0, 1])
