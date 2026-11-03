"""Kern gegen Handrechnung auf den Mini-Instanzen (conftest.py) und gegen geschlossene Formeln; je Einheit mindestens ein Test."""

import warnings

import numpy as np
import pytest

import gkz_algorithm as A


# --- Zufallsstrom --------------------------------------------------------------------------------------------------------------------------

def test_splitmix64_bekannter_startwert():
    assert A.mix_int(A.GAMMA) == 0xE220A8397B1DCDAF       # Seed 0, erste Ausgabe des Verfahrens


@pytest.mark.parametrize("seed", [0, 7, 2**63 + 12345])
def test_vektorisierter_strom_gleich_ganzzahl_referenz(seed):
    ref = [(A.mix_int((seed + i * A.GAMMA) & A.MASK) >> 11) * (1.0 / (1 << 53)) for i in range(1, 301)]
    assert np.array_equal(A.uniforms(seed, 300), np.array(ref))


def test_edge_uniforms_zahl_i_e_ist_nummer_i_mal_schrittweite_plus_e():
    flach = A.uniforms(9, 5 * 7)
    u = A.edge_uniforms(9, 5, 4, stride=7)
    for i in range(5):
        for e in range(4):
            assert u[i, e] == flach[i * 7 + e]


def test_edge_uniforms_schrittweite_gleich_kantenzahl_ist_der_standard():
    assert np.array_equal(A.edge_uniforms(3, 6, 4), A.edge_uniforms(3, 6, 4, stride=4))


def test_edge_uniforms_lehnt_zu_kleine_schrittweite_ab():
    with pytest.raises(ValueError):
        A.edge_uniforms(3, 6, 4, stride=3)


def test_stream_seed_ist_deterministisch_und_unterscheidet_schluessel():
    assert A.stream_seed(1, 2) == A.stream_seed(1, 2)
    assert len({A.stream_seed(1, 2), A.stream_seed(2, 1), A.stream_seed(1), A.stream_seed(1, 3)}) == 4


# --- Ziele --------------------------------------------------------------------------------------------------------------------------------

def test_komponenten_und_ziele_dreieck_von_hand(dreieck):
    zeilen = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)
    assert list(A.components(zeilen, 3, dreieck.edges)) == [1, 1, 2, 3]
    assert list(A.target_values("zusammenhaengend", zeilen, 3, dreieck.edges)) == [1.0, 1.0, 0.0, 0.0]
    assert list(A.target_values("zwei_teile", zeilen, 3, dreieck.edges)) == [0.0, 0.0, 1.0, 0.0]


def test_pfad_zerfaellt_bei_jeder_fehlenden_kante_in_zwei_teile(pfad):
    voll = np.ones((1, 4), dtype=bool)
    assert A.target_values("zwei_teile", voll, 5, pfad.edges)[0] == 0.0
    for e in range(4):
        z = voll.copy()
        z[0, e] = False
        assert A.target_values("zwei_teile", z, 5, pfad.edges)[0] == 1.0
        assert A.target_values("zusammenhaengend", z, 5, pfad.edges)[0] == 0.0


def test_ein_knoten_ist_zusammenhaengend_und_hat_eine_komponente():
    z = np.zeros((2, 0), dtype=bool)
    assert A.components(z, 1, ()).tolist() == [1, 1]
    assert A.target_values("zusammenhaengend", z, 1, ()).tolist() == [1.0, 1.0]


def test_unbekanntes_ziel_wird_abgelehnt(dreieck):
    with pytest.raises(ValueError):
        A.target_values("quatsch", np.ones((1, 3), dtype=bool), 3, dreieck.edges)


# --- Wahrheit -----------------------------------------------------------------------------------------------------------------------------

def test_zaehlung_dreieck_kreis_pfad_k4(dreieck, kreis4, pfad, k4):
    assert A.target_counts(dreieck.n, dreieck.edges)["zusammenhaengend"].tolist() == [0, 0, 3, 1]
    assert A.target_counts(dreieck.n, dreieck.edges)["zwei_teile"].tolist() == [0, 3, 0, 0]
    assert A.target_counts(kreis4.n, kreis4.edges)["zusammenhaengend"].tolist() == [0, 0, 0, 4, 1]
    assert A.target_counts(pfad.n, pfad.edges)["zusammenhaengend"].tolist() == [0, 0, 0, 0, 1]
    assert A.target_counts(k4.n, k4.edges)["zusammenhaengend"].tolist() == [0, 0, 0, 16, 15, 6, 1]


def test_zaehlung_in_kleinen_stuecken_gleich_ganzer_aufzaehlung(k4):
    klein = A.target_counts(k4.n, k4.edges, chunk=5)
    gross = A.target_counts(k4.n, k4.edges)
    assert all(klein[k].tolist() == gross[k].tolist() for k in gross)


def test_wahrscheinlichkeit_dreieck_von_hand(dreieck):
    counts = A.target_counts(dreieck.n, dreieck.edges)
    assert A.probability_from_counts(counts["zusammenhaengend"], 1 / 3) == pytest.approx(7 / 27, rel=1e-12)
    assert A.probability_from_counts(counts["zwei_teile"], 1 / 3) == pytest.approx(3 * (1 / 3) * (2 / 3) ** 2, rel=1e-12)   # genau eine Kante hält


# --- Paare --------------------------------------------------------------------------------------------------------------------------------

def test_pair_values_das_zweite_netz_nutzt_die_gespiegelten_zahlen(dreieck):
    u = A.edge_uniforms(11, 40, 3)
    xa, xb = A.pair_values(11, 40, 0.6, "zusammenhaengend", dreieck.n, dreieck.edges)
    assert xa.tolist() == [1.0 if (x < 0.6).sum() >= 2 else 0.0 for x in u]
    assert xb.tolist() == [1.0 if ((1 - x) < 0.6).sum() >= 2 else 0.0 for x in u]


def test_exakte_momente_dreieck_von_hand(dreieck):
    # p = 0,6: je Kante (beide, nur erstes, nur zweites) mit 0,2 / 0,4 / 0,4. X = 1, wenn mindestens 2 Kanten halten.
    mo = A.exact_pair_moments("zusammenhaengend", dreieck.n, dreieck.edges, 0.6)
    assert mo["mass"] == pytest.approx(1.0, abs=1e-12)
    assert mo["EX"] == pytest.approx(0.648, abs=1e-12) and mo["EXp"] == pytest.approx(0.648, abs=1e-12)   # 3 * 0,36 * 0,4 + 0,216
    # beide Netze verbunden: i Kanten in beiden, j nur im ersten, k nur im zweiten (i + j + k = 3); nötig sind i + j >= 2 und i + k >= 2:
    # (3,0,0): 0,008; (2,1,0) und (2,0,1): je 3 * 0,04 * 0,4 = 0,048; (1,1,1): 6 * 0,2 * 0,4 * 0,4 = 0,192; Summe 0,296
    assert mo["EXXp"] == pytest.approx(0.008 + 2 * 0.048 + 0.192, abs=1e-12)


def test_pfad_korrelation_aufzaehlung_gleich_geschlossene_formel():
    for m in (1, 3, 5):
        for p in (0.6, 0.8, 0.95):
            edges = [(i, i + 1) for i in range(m)]
            rho = A.rho_from_moments(A.exact_pair_moments("zusammenhaengend", m + 1, edges, p))
            assert rho == pytest.approx(A.path_rho(m, p), abs=1e-9)
    assert A.path_rho(1, 0.8) == pytest.approx(-(1 - 0.8) / 0.8)          # m = 1: rho = -(1-p)/p


def test_exakte_momente_fuer_p_unter_einhalb_nutzen_die_andere_kategorienfolge(dreieck):
    mo = A.exact_pair_moments("zusammenhaengend", dreieck.n, dreieck.edges, 0.4)
    assert mo["mass"] == pytest.approx(1.0, abs=1e-12)
    assert mo["EX"] == pytest.approx(A.probability_from_counts(A.target_counts(dreieck.n, dreieck.edges)["zusammenhaengend"], 0.4), abs=1e-12)


# --- Schätzer -----------------------------------------------------------------------------------------------------------------------------

def test_spread_ist_deterministisch_und_hat_die_richtige_form(dreieck):
    a = A.spread(5, 30, 20, 0.7, "zusammenhaengend", dreieck.n, dreieck.edges)
    b = A.spread(5, 30, 20, 0.7, "zusammenhaengend", dreieck.n, dreieck.edges)
    assert a["plain"].shape == a["anti"].shape == (30,)
    assert np.array_equal(a["plain"], b["plain"]) and np.array_equal(a["anti"], b["anti"]) and a["rho"] == b["rho"]


def test_spread_gleich_unabhaengige_neuberechnung_der_gespiegelten_schaetzung(dreieck):
    sp = A.spread(5, 4, 20, 0.7, "zusammenhaengend", dreieck.n, dreieck.edges)
    for rep in range(4):
        u = A.edge_uniforms(A.stream_seed(2, 5, 20, rep), 10, 3)
        hand = 0.0
        for zeile in u:
            hand += int(sum(int(x < 0.7) for x in zeile) >= 2) + int(sum(int((1 - x) < 0.7) for x in zeile) >= 2)     # int(): np.bool_ + np.bool_ wäre ein ODER
        assert sp["anti"][rep] == pytest.approx(hand / 20)


def test_spread_verlangt_gerades_n(dreieck):
    with pytest.raises(ValueError):
        A.spread(5, 3, 21, 0.7, "zusammenhaengend", dreieck.n, dreieck.edges)


# --- Kopplung zweier Varianten ----------------------------------------------------------------------------------------------------------

def test_differenz_bei_gemeinsamen_zahlen_liegt_in_null_eins_und_hat_die_bernoulli_varianz(kreis4):
    r = A.difference_experiment(3, 500, 0.6, 0.8, kreis4.n, kreis4.edges)
    m = r["mean_crn"]
    assert r["var_crn"] == pytest.approx(m * (1 - m) * 500 / 499, rel=1e-12)       # D in {0,1}: Var = m (1-m) n/(n-1)
    assert r["var_ind"] > r["var_crn"]


def test_design_experiment_ist_synchron_und_sync_fest_gleich_design(kreis4):
    edges_b = list(kreis4.edges) + [(0, 2)]
    d = A.design_experiment(3, 800, 0.7, kreis4.n, list(kreis4.edges), edges_b)
    s = A.sync_experiment(3, 800, 0.7, kreis4.n, list(kreis4.edges), edges_b)
    assert s["fest"] == pytest.approx(d, rel=0) and d["corr"] > 0.5
    assert s["lage"]["corr"] < d["corr"] - 0.3


# --- Skalare ------------------------------------------------------------------------------------------------------------------------------

def test_skalare_korrelation_symmetrisch_exakt_eins_und_quadrat_nahe_minus_sieben_achtel():
    assert A.scalar_rho("symmetrisch", 1, 100_000) == pytest.approx(1.0, abs=1e-12)
    assert A.scalar_rho("quadrat", 1, 400_000) == pytest.approx(-7 / 8, abs=0.003)


def test_kopplung_mit_konstanter_variante_gibt_nan_ohne_warnung(kreis4):
    """Zwei getrennte Paare sind nie zusammenhängend: X_A ist konstant 0, die Korrelation ist nicht definiert (NaN), und numpy warnt nicht."""
    edges_a = [(0, 1), (2, 3)]
    edges_b = edges_a + [(1, 3)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = A.design_experiment(3, 500, 0.7, 4, edges_a, edges_b)
        s = A.sync_experiment(3, 500, 0.7, 4, edges_a, edges_b)
    assert np.isnan(d["corr"]) and np.isnan(s["fest"]["corr"]) and np.isnan(s["lage"]["corr"])
    assert d["mean"] >= 0.0
