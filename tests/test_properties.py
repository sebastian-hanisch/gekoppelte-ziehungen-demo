"""Eigenschaftstests (ORACLE-PLAYBOOK §5): Aussagen, die für ALLE Eingaben gelten müssen, an festen Zufallsständen geprüft."""

import numpy as np
import pytest

import gkz_algorithm as A
import gkz_constants as C
import gkz_evaluation as ev
from gkz_scenario import grid_netz, named_netz, parse_edges, with_edge


@pytest.mark.parametrize("gitter", [(2, 3), (3, 3)])
def test_fuer_ein_monotones_ziel_ist_die_korrelation_des_paares_nie_positiv(gitter):
    netz = grid_netz(*gitter)
    for p in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        rho = A.rho_from_moments(A.exact_pair_moments("zusammenhaengend", netz.n, netz.edges, p))
        assert rho <= 1e-12, (gitter, p, rho)


def test_beide_schaetzer_sind_erwartungstreu_innerhalb_von_vier_standardfehlern():
    netz = named_netz("2x3")
    counts = ev.netz_counts(netz)
    truth = A.probability_from_counts(counts["zusammenhaengend"], 0.7)
    sp = A.spread(11, 400, 100, 0.7, "zusammenhaengend", netz.n, netz.edges)
    se = np.sqrt(sp["plain"].var(ddof=1) / 400)
    assert abs(sp["plain"].mean() - truth) < 4 * se
    assert abs(sp["anti"].mean() - truth) < 4 * se


def test_gepoolte_korrelation_stimmt_mit_der_exakten_ueberein_und_sagt_das_varianzverhaeltnis_voraus():
    netz = named_netz("2x3")
    exakt = A.rho_from_moments(A.exact_pair_moments("zusammenhaengend", netz.n, netz.edges, 0.6))
    sp = A.spread(11, 300, 400, 0.6, "zusammenhaengend", netz.n, netz.edges)
    assert sp["rho"] == pytest.approx(exakt, abs=0.02)
    ratio = sp["anti"].var(ddof=1) / sp["plain"].var(ddof=1)
    assert ratio == pytest.approx(1 + exakt, rel=0.3)          # Rauschen der Varianzverhältnisse bei 300 Wiederholungen: rund 8 %


def test_gemeinsame_zahlen_sind_monoton_gekoppelt_die_differenz_ist_null_oder_eins():
    netz = named_netz("3x3")
    u = A.edge_uniforms(5, 2000, netz.m)
    xa = A.target_values("zusammenhaengend", u < 0.6, netz.n, netz.edges)
    xb = A.target_values("zusammenhaengend", u < 0.8, netz.n, netz.edges)
    assert set(np.unique(xb - xa)) <= {0.0, 1.0}
    assert (xb >= xa).all()


def test_zusatzkante_mit_festen_zufallszahlen_macht_das_netz_nie_schlechter():
    netz = named_netz("3x3")
    netz_b = with_edge(netz, (0, 4))
    ub = A.edge_uniforms(5, 2000, netz_b.m)
    xa = A.target_values("zusammenhaengend", ub[:, : netz.m] < 0.7, netz.n, netz.edges)
    xb = A.target_values("zusammenhaengend", ub < 0.7, netz_b.n, netz_b.edges)
    assert (xb >= xa).all()


def test_falsche_zuordnung_aendert_den_mittelwert_der_differenz_nicht_aber_ihre_streuung():
    netz = named_netz("3x3")
    netz_b = with_edge(netz, (0, 4))
    res = A.sync_experiment(5, 60_000, 0.7, netz.n, netz.edges, netz_b.edges)
    assert res["lage"]["mean"] == pytest.approx(res["fest"]["mean"], abs=0.012)
    assert res["lage"]["var"] > 4 * res["fest"]["var"]
    assert abs(res["lage"]["corr"]) < 0.05 < res["fest"]["corr"]


def test_gleicher_strom_gleiche_zahlen_und_laengere_folge_beginnt_gleich():
    a = A.edge_uniforms(5, 50, 4)
    assert np.array_equal(a, A.edge_uniforms(5, 50, 4)) and np.array_equal(a[:10], A.edge_uniforms(5, 10, 4))


def test_analyse_ist_deterministisch():
    s = ev.Settings(n=100)
    assert ev.analyse(s) == ev.analyse(s)


def test_gewinnfaktor_waechst_mit_kleinerem_abstand():
    settings = ev.Settings()
    netz = named_netz(settings.gitter)
    counts = ev.netz_counts(netz)
    rows = ev.crn_experiment(settings, netz, counts)["rows"]
    faktoren = [r["factor"] for r in rows]                       # CRN_DELTAS fallen: der Gewinn wächst
    assert all(b > a for a, b in zip(faktoren, faktoren[1:]))


def test_parse_edges_ist_unabhaengig_von_reihenfolge_und_trennzeichen():
    a, b = parse_edges("0-1\n1-2\n2-0"), parse_edges("2 0\n1,2\n0;1")
    assert sorted(a.edges) == sorted(b.edges) and a.n == b.n == 3
    assert C.CUSTOM_EDGE_LIMIT >= 3
