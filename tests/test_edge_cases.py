"""Randfälle: Grenzen der Regler, entartete Netze, Eingabefehler der Spielwiese."""

import math

import numpy as np
import pytest

import gkz_algorithm as A
import gkz_constants as C
import gkz_evaluation as ev
from gkz_scenario import Netz, extra_edge, named_netz, parse_edges, parse_extra_edge, with_edge


def test_kleinste_und_groesste_auswertungszahl_laufen_durch():
    for n in (C.N_MIN, C.N_MAX):
        a = ev.analyse(ev.Settings(n=n))
        assert 0.0 <= a.mean_plain <= 1.0 and 0.0 <= a.mean_anti <= 1.0


def test_hoechstes_und_niedrigstes_p_laufen_durch():
    for p in (C.P_MIN, C.P_MAX):
        a = ev.analyse(ev.Settings(p=p, n=100))
        assert 0.0 < a.truth < 1.0


@pytest.mark.parametrize("ziel", list(C.TARGETS))
def test_jedes_gitter_und_jedes_ziel_laeuft_durch(ziel):
    for key in C.GRIDS:
        a = ev.analyse(ev.Settings(gitter=key, ziel=ziel, n=100))
        assert 0.0 <= a.truth <= 1.0


def test_ein_ziel_das_nie_eintritt_hat_keine_streuung_und_eine_undefinierte_korrelation():
    netz = Netz("zwei Paare", 4, ((0, 1), (2, 3)), tuple((float(i), 0.0) for i in range(4)))   # nie zusammenhängend
    counts = ev.netz_counts(netz)
    a = ev.analyse(ev.Settings(n=100), netz=netz, counts=counts)
    assert a.truth == 0.0 and a.sd_plain == 0.0 and a.sd_anti == 0.0 and math.isnan(a.var_ratio) and math.isnan(a.rho)
    assert math.isnan(ev.custom_result(ev.Settings(), netz)["factor"])


def test_eigenes_netz_mit_zusatzkante_laeuft_durch():
    netz = parse_edges("0-1\n1-2\n2-3")
    res = ev.custom_result(ev.Settings(p=0.8, n=100), netz, (0, 3))
    assert res["design"]["dR"] > 0 and res["design"]["factor"] > 1.0
    assert res["exakt"] is True


def test_netz_oberhalb_der_aufzaehlungsgrenze_wird_abgelehnt():
    n = 12
    edges = tuple((i, j) for i in range(6) for j in range(i + 1, 7))[: C.LIVE_EDGE_LIMIT + 1]       # Teil des K7: 18 Kanten
    with pytest.raises(ValueError):
        ev.netz_counts(Netz("zu gross", n, edges, tuple((0.0, 0.0) for _ in range(n))))


def test_exakte_paarmomente_gibt_es_nur_bis_zur_grenze_sonst_none():
    assert ev.exact_rho(named_netz("3x3"), "zusammenhaengend", 0.7) is not None          # 12 Kanten: 3^12
    assert ev.exact_rho(named_netz("3x4"), "zusammenhaengend", 0.7) is None              # 17 Kanten


def test_ungerade_auswertungszahl_wird_abgelehnt():
    netz = named_netz("2x3")
    with pytest.raises(ValueError):
        A.spread(1, 5, 101, 0.7, "zusammenhaengend", netz.n, netz.edges)


@pytest.mark.parametrize("text,fragment", [
    ("", "Keine Kanten"),
    ("# nur Kommentar\n\n", "Keine Kanten"),
    ("0-0", "Schleife"),
    ("0-1\n1-0", "Doppelte Kante"),
    ("0-1\nabc", "Zeile 2"),
    ("0-1-2", "Zeile 1"),
    ("\n".join(f"{i}-{i + 1}" for i in range(C.CUSTOM_EDGE_LIMIT + 1)), "Kanten: höchstens"),
    ("\n".join(f"0-{i}" for i in range(1, C.CUSTOM_NODE_LIMIT + 1)), "Knoten: höchstens"),
])
def test_eingabefehler_der_spielwiese_haben_eine_klare_meldung(text, fragment):
    with pytest.raises(ValueError) as err:
        parse_edges(text)
    assert fragment in str(err.value)


def test_zusatzkante_pruefungen():
    netz = named_netz("2x3")
    assert parse_extra_edge("", netz) is None
    assert parse_extra_edge("0-4", netz) == (0, 4)
    for text, fragment in (("0-1", "gibt es schon"), ("0-9", "passt nicht"), ("2-2", "Schleife"), ("a-b", "Zeile 1")):
        with pytest.raises(ValueError) as err:
            parse_extra_edge(text, netz)
        assert fragment in str(err.value)


def test_standard_zusatzkante_jedes_gitters_ist_neu():
    for key in C.GRIDS:
        netz = named_netz(key)
        assert with_edge(netz, extra_edge(key)).m == netz.m + 1


def test_knoten_werden_von_null_an_umnummeriert():
    netz = parse_edges("10-20\n20-30")
    assert netz.n == 3 and netz.edges == ((0, 1), (1, 2))


def test_worst_case_am_regler_maximum_laeuft_durch():
    """Hänger-Netz am Slider-Maximum (kein Zeittest, nur ein Netz): 3x4, 2000 Auswertungen, 200 Wiederholungen."""
    a = ev.analyse(ev.Settings(n=C.N_MAX))
    assert len(a.plain) == C.REPS and np.isfinite(a.var_ratio)
