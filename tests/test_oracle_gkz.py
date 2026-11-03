"""Orakel-Tests (ORACLE-PLAYBOOK §5): Kernroutinen gegen einen UNABHÄNGIGEN Rechenweg.

  Komponenten und Ziele   gegen networkx.number_connected_components auf zufälligen Kantenmengen
  Aufzählung N_k          gegen Matrix-Baum-Satz (Determinante) und gegen Bruchrechnung mit Union-Find (fractions)
  exakte Paarmomente      gegen eine eigene Bruchrechnung über alle Kategorienfolgen (itertools.product) und gegen große Simulationen
  gemeinsame Zufallszahlen  Var(X_B - X_A) = dR (1 - dR) gegen die exakte Verteilung der gekoppelten Differenz (Bruchrechnung)
Mini-Instanzen, feste Stände (range statt Zufall), insgesamt wenige Sekunden."""

import itertools
import random
from fractions import Fraction

import numpy as np
import pytest

import gkz_algorithm as A
from gkz_scenario import grid_netz, named_netz


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _components_unionfind(n, edges, up):
    parent = list(range(n))
    for (u, v), keep in zip(edges, up):
        if keep:
            parent[_find(parent, u)] = _find(parent, v)
    return len({_find(parent, x) for x in range(n)})


def _target(name, comps):
    return 1 if (comps == 1 if name == "zusammenhaengend" else comps == 2) else 0


def _exact_moments_fractions(name, n, edges, p: Fraction):
    """E X, E X', E X X' durch Aufzählung aller Kategorienfolgen: Der Bereich [0, 1) der Zufallszahl u zerfällt an den Punkten p und 1 - p in Stücke;
    in jedem Stück ist festgelegt, ob die Kante im ersten Netz (u < p) und im zweiten (u > 1 - p) hält. Die Länge des Stücks ist seine Wahrscheinlichkeit."""
    cuts = sorted({Fraction(0), p, 1 - p, Fraction(1)})
    cats = []
    for a, b in zip(cuts, cuts[1:]):
        mid = (a + b) / 2
        cats.append(((mid < p), (mid > 1 - p), b - a))
    ex = exp = exxp = Fraction(0)
    for combo in itertools.product(cats, repeat=len(edges)):
        w = Fraction(1)
        up_a, up_b = [], []
        for ha, hb, length in combo:
            w *= length
            up_a.append(ha)
            up_b.append(hb)
        xa = _target(name, _components_unionfind(n, edges, up_a))
        xb = _target(name, _components_unionfind(n, edges, up_b))
        ex += w * xa
        exp += w * xb
        exxp += w * xa * xb
    return ex, exp, exxp


def _spanning_trees(n, edges):
    lap = np.zeros((n, n))
    for u, v in edges:
        lap[u, u] += 1
        lap[v, v] += 1
        lap[u, v] -= 1
        lap[v, u] -= 1
    return round(np.linalg.det(lap[1:, 1:]))


def test_komponenten_gegen_networkx_auf_zufaelligen_kantenmengen():
    nx = pytest.importorskip("networkx")
    rng = random.Random(2026)
    for _ in range(40):
        n = rng.randint(2, 7)
        alle = list(itertools.combinations(range(n), 2))
        edges = tuple(rng.sample(alle, rng.randint(1, min(len(alle), 12))))
        stats = np.array([[rng.random() < 0.55 for _ in edges] for _ in range(30)])
        ours = A.components(stats, n, edges)
        for row, got in zip(stats, ours):
            g = nx.Graph()
            g.add_nodes_from(range(n))
            g.add_edges_from(e for e, keep in zip(edges, row) if keep)
            assert nx.number_connected_components(g) == int(got)


def test_spannbaumzahl_gegen_matrix_baum_satz_auf_gittern_und_zufallsgraphen():
    rng = random.Random(7)
    netze = [grid_netz(2, 3), grid_netz(3, 3)]
    for netz in netze:
        assert A.target_counts(netz.n, netz.edges)["zusammenhaengend"][netz.n - 1] == _spanning_trees(netz.n, netz.edges)
    for _ in range(15):
        n = rng.randint(3, 6)
        alle = list(itertools.combinations(range(n), 2))
        edges = tuple(rng.sample(alle, rng.randint(n - 1, min(10, len(alle)))))
        counts = A.target_counts(n, edges)["zusammenhaengend"]
        assert counts[n - 1] == _spanning_trees(n, edges) and counts[: n - 1].sum() == 0


def test_zaehlung_beider_ziele_gegen_bruchrechnung_mit_union_find():
    for netz in (grid_netz(2, 2), grid_netz(2, 3)):
        counts = A.target_counts(netz.n, netz.edges)
        for name in ("zusammenhaengend", "zwei_teile"):
            for p in (Fraction(1, 3), Fraction(3, 5)):
                total = Fraction(0)
                for mask in range(1 << netz.m):
                    up = [(mask >> e) & 1 for e in range(netz.m)]
                    if _target(name, _components_unionfind(netz.n, netz.edges, up)):
                        k = sum(up)
                        total += p**k * (1 - p) ** (netz.m - k)
                assert A.probability_from_counts(counts[name], float(p)) == pytest.approx(float(total), rel=1e-12), (netz.name, name, p)


@pytest.mark.parametrize("p", [Fraction(3, 5), Fraction(4, 5), Fraction(2, 5), Fraction(1, 2)])
@pytest.mark.parametrize("name", ["zusammenhaengend", "zwei_teile"])
def test_exakte_paarmomente_gegen_bruchrechnung_ueber_alle_kategorienfolgen(name, p):
    netz = grid_netz(2, 2)                      # 4 Kanten: höchstens 3^4 Kategorienfolgen je Intervallzerlegung
    ex, exp, exxp = _exact_moments_fractions(name, netz.n, netz.edges, p)
    mo = A.exact_pair_moments(name, netz.n, netz.edges, float(p))
    assert mo["EX"] == pytest.approx(float(ex), abs=1e-12)
    assert mo["EXp"] == pytest.approx(float(exp), abs=1e-12)
    assert mo["EXXp"] == pytest.approx(float(exxp), abs=1e-12)


def test_exakte_korrelation_gegen_grosse_simulation_am_gitter_2x3():
    netz = named_netz("2x3")
    for name, p in (("zusammenhaengend", 0.6), ("zwei_teile", 0.6), ("zusammenhaengend", 0.8)):
        exakt = A.rho_from_moments(A.exact_pair_moments(name, netz.n, netz.edges, p))
        xa, xb = A.pair_values(A.stream_seed(7, int(p * 10)), 150_000, p, name, netz.n, netz.edges)
        assert np.corrcoef(xa, xb)[0, 1] == pytest.approx(exakt, abs=0.012)         # 150 000 Paare: Rauschen unter 0,003


def test_gekoppelte_differenz_gegen_exakte_verteilung_in_bruchrechnung():
    """Gitter 2x2 (Kreis mit 4 Kanten), p = 1/2 gegen p' = 3/4 mit gemeinsamen u: Je Kante (hält in beiden, nur in B, in keiner) mit 1/2, 1/4, 1/4. D = X_B - X_A in {0, 1}."""
    netz = grid_netz(2, 2)
    p_a, p_b = Fraction(1, 2), Fraction(3, 4)
    cats = [((True, True), p_a), ((False, True), p_b - p_a), ((False, False), 1 - p_b)]
    e_d = e_d2 = Fraction(0)
    for combo in itertools.product(cats, repeat=netz.m):
        w = Fraction(1)
        up_a, up_b = [], []
        for (ha, hb), pr in combo:
            w *= pr
            up_a.append(ha)
            up_b.append(hb)
        d = _target("zusammenhaengend", _components_unionfind(netz.n, netz.edges, up_b)) - _target("zusammenhaengend", _components_unionfind(netz.n, netz.edges, up_a))
        e_d += w * d
        e_d2 += w * d * d
    counts = A.target_counts(netz.n, netz.edges)["zusammenhaengend"]
    dr = A.probability_from_counts(counts, 0.75) - A.probability_from_counts(counts, 0.5)
    assert float(e_d) == pytest.approx(dr, abs=1e-12)
    assert float(e_d2 - e_d**2) == pytest.approx(dr * (1 - dr), abs=1e-12)         # Var(D) = dR (1 - dR)
