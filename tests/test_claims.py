"""Jede Zahl aus der README und den App-Texten, nachgerechnet über dieselben Auswertungsfunktionen wie die App (DEMO-PLAYBOOK §4).

Ströme sind fest (SplitMix64) und die Rechnung besteht aus Vergleichen und Zählungen: die Zahlen sind plattformunabhängig bis auf die letzte Stelle der
Gleitkommasummen. Verhältnisse verrauschter Messungen haben trotzdem Bänder (Rauschen der Varianzverhältnisse bei 200 Wiederholungen: rund 10 %).
Zeiten stehen nie in einem Test."""

import numpy as np
import pytest

import gkz_algorithm as A
import gkz_constants as C
import gkz_evaluation as ev
from gkz_scenario import named_netz, parse_edges

STANDARD = C.PRESETS["Standardfall (Voreinstellung)"]


def _analyse(preset):
    return ev.analyse(ev.Settings(**C.PRESETS[preset]))


# --- Die vier Presets ----------------------------------------------------------------------------------------------------------------------

def test_preset_standardfall():
    a = _analyse("Standardfall (Voreinstellung)")
    assert a.netz.m == 17 and a.rho_exact is None
    assert a.truth == pytest.approx(0.45602, abs=5e-6)
    assert a.sd_plain == pytest.approx(0.01569, abs=2e-5) and a.sd_anti == pytest.approx(0.01436, abs=2e-5)
    assert a.var_ratio == pytest.approx(0.838, abs=0.01)
    assert a.rho == pytest.approx(-0.178, abs=0.001) and a.predicted_ratio == pytest.approx(0.822, abs=0.001)
    assert ev.verdict_spread(a, 1000)[0] == "success" and "um 18 %" in ev.verdict_spread(a, 1000)[1]


def test_preset_kraeftiger_gewinn_exakt():
    a = _analyse("Kräftiger Gewinn")
    assert a.netz.m == 12 and a.rho_exact == pytest.approx(-0.2545, abs=5e-5) and a.predicted_ratio == pytest.approx(0.7455, abs=5e-5)
    assert a.truth == pytest.approx(0.26791, abs=5e-6)
    assert a.var_ratio == pytest.approx(0.675, abs=0.01)
    assert "um 25 %" in ev.verdict_spread(a, 1000)[1]


def test_preset_seltenes_ereignis():
    a = _analyse("Seltenes Ereignis")
    assert a.truth == pytest.approx(0.99958, abs=5e-6) and a.rho == pytest.approx(-0.0004, abs=2e-4)
    assert a.predicted_ratio == pytest.approx(0.9996, abs=2e-4) and a.var_ratio == pytest.approx(1.045, abs=0.01)
    assert ev.verdict_spread(a, 2000)[0] == "info"


def test_preset_nicht_monotones_ziel_exakt():
    a = _analyse("Nicht monotones Ziel")
    assert a.netz.m == 7 and a.rho_exact == pytest.approx(0.1312, abs=5e-5) and a.predicted_ratio == pytest.approx(1.1312, abs=5e-5)
    assert a.truth == pytest.approx(0.34836, abs=5e-6) and a.var_ratio == pytest.approx(1.185, abs=0.01)
    kind, text = ev.verdict_spread(a, 1000)
    assert kind == "warning" and "ERHÖHT die Varianz um 13 %" in text


def test_beide_schaetzer_treffen_die_wahrheit():
    for preset in C.PRESETS:
        a = _analyse(preset)
        assert abs(a.mean_plain - a.truth) < 4 * a.sd_plain / np.sqrt(C.REPS) + 1e-9, preset
        assert abs(a.mean_anti - a.truth) < 4 * a.sd_anti / np.sqrt(C.REPS) + 1e-9, preset


def test_rauschen_der_varianzverhaeltnisse():
    assert 100 / np.sqrt(C.REPS / 2) == pytest.approx(10.0)
    assert 100 / np.sqrt(C.SWEEP_REPS / 2) == pytest.approx(10.0)


# --- Experiment 1: Streuung gegen N -----------------------------------------------------------------------------------------------------------

def test_experiment_1_verhaeltnisse_und_steigung():
    settings = ev.Settings()
    exp = ev.n_sweep(settings, named_netz(settings.gitter))
    assert [r["N"] for r in exp["rows"]] == [100, 300, 1000, 3000]
    assert [r["ratio"] for r in exp["rows"]] == pytest.approx([0.571, 0.738, 0.838, 0.746], abs=0.01)
    assert exp["slope_plain"] == pytest.approx(-0.50, abs=0.01) and exp["slope_anti"] == pytest.approx(-0.46, abs=0.01)
    assert all(-0.21 < r["rho"] < -0.17 for r in exp["rows"])                    # die Korrelation hängt nicht an N
    assert ev.verdict_n_sweep(exp)[0] == "info"


# --- Experiment 2: Korrelation gegen p -----------------------------------------------------------------------------------------------------

def test_experiment_2_kurve_fuer_3x4_aus_paaren():
    settings = ev.Settings()
    exp = ev.rho_curve(settings, named_netz("3x4"))
    assert [r["p"] for r in exp["rows"]] == list(C.RHO_PS) and not any(r["exakt"] for r in exp["rows"])
    assert [r["factor"] for r in exp["rows"]] == pytest.approx([0.936, 0.8126, 0.8143, 0.9162, 0.9854, 0.9929, 0.9996], abs=0.0015)
    best = min(exp["rows"], key=lambda r: r["factor"])
    assert best["p"] == 0.6 and best["factor"] == pytest.approx(0.8126, abs=0.0015)


def test_experiment_2_kurve_fuer_3x3_exakt():
    exp = ev.rho_curve(ev.Settings(gitter="3x3"), named_netz("3x3"))
    assert all(r["exakt"] for r in exp["rows"])
    assert [r["factor"] for r in exp["rows"]] == pytest.approx([0.8824, 0.7455, 0.7889, 0.904, 0.9806, 0.9962, 0.9999], abs=5e-4)


def test_gewinn_haengt_am_netz_bei_p_gleich_0_6():
    faktoren = {}
    for key in ("2x3", "3x3"):
        netz = named_netz(key)
        faktoren[key] = 1 + ev.exact_rho(netz, "zusammenhaengend", 0.6)
    assert faktoren["2x3"] == pytest.approx(0.6564, abs=5e-5) and faktoren["3x3"] == pytest.approx(0.7455, abs=5e-5)
    assert faktoren["2x3"] < faktoren["3x3"] < 0.8126                  # kleinere Netze gewinnen mehr; 0,8126 ist der Wert des 3x4-Gitters


def test_pfad_geschlossene_formel_zahlen():
    assert A.path_rho(3, 0.6) == pytest.approx(-0.2283, abs=5e-5)
    assert A.path_rho(5, 0.8) == pytest.approx(-0.1344, abs=5e-5)
    assert A.path_rho(1, 0.8) == pytest.approx(-0.25)


# --- Experiment 3: welche Funktion? -------------------------------------------------------------------------------------------------------

def test_experiment_3_funktionen_am_standardfall():
    settings = ev.Settings()
    exp = ev.function_table(settings, named_netz(settings.gitter))
    f = {r["name"]: r["factor"] for r in exp["rows"]}
    assert f["Netz hängt zusammen (monoton)"] == pytest.approx(0.8191, abs=0.003)
    assert f["Netz zerfällt in genau zwei Teile (nicht monoton)"] == pytest.approx(1.0149, abs=0.003)
    assert f[ev.SCALAR_LABELS["quadrat"]] == pytest.approx(0.1249, abs=0.002) and f[ev.SCALAR_LABELS["symmetrisch"]] == pytest.approx(2.0, abs=1e-9)


def test_experiment_3_funktionen_beim_nicht_monotonen_preset_exakt():
    exp = ev.function_table(ev.Settings(**C.PRESETS["Nicht monotones Ziel"]), named_netz("2x3"))
    f = [r["factor"] for r in exp["rows"]]
    assert f[:2] == pytest.approx([0.6564, 1.1312], abs=5e-5) and all(r["exakt"] for r in exp["rows"][:2])
    kind, text = ev.verdict_functions(exp)
    assert kind == "warning" and "Über 1 liegt: Netz zerfällt in genau zwei Teile (nicht monoton), (2u−1)² (Skalar, symmetrisch um ½)" in text


def test_skalare_exakt():
    assert A.scalar_rho("quadrat", 3, C.SCALAR_DRAWS) == pytest.approx(-7 / 8, abs=0.002)
    assert A.scalar_rho("symmetrisch", 3, 1000) == 1.0


def test_exakte_paarkorrelation_bei_zwei_teilen_3x3_ist_klein_positiv():
    rho = ev.exact_rho(named_netz("3x3"), "zwei_teile", 0.6)
    assert rho == pytest.approx(0.0268, abs=5e-5) and ev.exact_rho(named_netz("3x3"), "zwei_teile", 0.7) == pytest.approx(0.0024, abs=5e-5)


# --- Experiment 4 und 5: gemeinsame Zufallszahlen -----------------------------------------------------------------------------------------------

def test_experiment_4_gewinn_gegen_delta_und_entwurfsvergleich():
    settings = ev.Settings()
    netz = named_netz(settings.gitter)
    exp = ev.crn_experiment(settings, netz, ev.netz_counts(netz))
    assert [r["factor"] for r in exp["rows"]] == pytest.approx([1.2, 2.12, 3.84, 8.92, 17.38], abs=0.01)
    assert [r["factor_meas"] for r in exp["rows"]] == pytest.approx([1.2, 2.11, 3.82, 8.79, 17.0], abs=0.15)
    assert all(abs(r["var_crn_meas"] / r["var_crn"] - 1) < 0.03 for r in exp["rows"])                # Formel gegen Messung: höchstens 3 %
    d = exp["design"]
    assert d["edge"] == (0, 5) and d["dR"] == pytest.approx(0.0519, abs=5e-5) and d["factor"] == pytest.approx(10.1, abs=0.05)
    assert d["corr"] == pytest.approx(0.8995, abs=0.003) and d["R_a"] == pytest.approx(0.4560, abs=5e-5) and d["R_b"] == pytest.approx(0.5079, abs=5e-5)
    assert ev.verdict_crn(exp)[0] == "success"


def test_experiment_5_synchronisationsfalle():
    settings = ev.Settings()
    exp = ev.sync_result(settings, named_netz(settings.gitter))
    assert exp["fest"]["corr"] == pytest.approx(0.903, abs=0.003) and exp["fest"]["var"] == pytest.approx(0.0485, abs=0.0006)
    assert abs(exp["lage"]["corr"]) < 0.01 and exp["lage"]["var"] == pytest.approx(0.4988, abs=0.006)
    assert exp["var_ind"] == pytest.approx(0.4980, abs=5e-4)
    assert exp["lage"]["var"] / exp["fest"]["var"] == pytest.approx(10.3, abs=0.2)
    assert exp["lage"]["mean"] == pytest.approx(exp["fest"]["mean"], abs=0.003)


def test_spielwiese_standardnetz():
    settings = ev.Settings()
    netz = parse_edges("0-1\n1-2\n2-3\n3-0\n0-2")
    res = ev.custom_result(settings, netz, (1, 3))
    assert res["exakt"] and res["truth"] == pytest.approx(0.77518, abs=5e-6)
    assert res["rho"] == pytest.approx(-0.1971, abs=5e-5) and res["factor"] == pytest.approx(0.8029, abs=5e-5)
    d = res["design"]
    assert d["R_a"] == pytest.approx(0.7752, abs=5e-5) and d["R_b"] == pytest.approx(0.8925, abs=5e-5) and d["dR"] == pytest.approx(0.1173, abs=5e-5)
    assert d["factor"] == pytest.approx(2.61, abs=0.01)


def test_gitter_kantenzahlen():
    assert [named_netz(k).m for k in C.GRIDS] == [7, 12, 13, 17]
    assert C.EXACT_PAIR_EDGE_LIMIT == 12 and C.LIVE_EDGE_LIMIT == 17 and C.DESIGN_EDGE_LIMIT == 18
