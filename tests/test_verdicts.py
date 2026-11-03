"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt gesetzte Werte, keine Seeds. Damit prüft der Test die Grenzen und jede genannte Zahl direkt."""

import gkz_evaluation as ev
from gkz_scenario import named_netz


def _analysis(pred_rho, var_ratio=0.8, rho=None, sd_plain=0.02, sd_anti=0.018, exact=True):
    """rho_exact = pred_rho (exakte Korrelation) oder None; rho = aus den Paaren geschätzte Korrelation."""
    return ev.Analysis(named_netz("2x3"), 0.5, 0.5, 0.5, sd_plain, sd_anti, var_ratio, pred_rho if rho is None else rho,
                       pred_rho if exact else None, (0.5,), (0.5,))


def test_zahlenformat_ist_deutsch():
    assert ev.de(1234567.891, 2) == "1.234.567,89" and ev.de(0.05, 3) == "0,050" and ev.de(-0.5, 1) == "−0,5"
    assert ev.de_e(4.17e-10) == "4,2e-10"


def test_vorhergesagtes_verhaeltnis_nutzt_die_exakte_korrelation_wenn_sie_vorliegt():
    assert _analysis(-0.2, rho=-0.5).predicted_ratio == 0.8
    assert _analysis(-0.2, rho=-0.5, exact=False).predicted_ratio == 0.5


def test_spread_verdikt_gewinn_nennt_alle_zahlen():
    kind, text = ev.verdict_spread(_analysis(-0.25, var_ratio=0.78, sd_plain=0.0216, sd_anti=0.0191), 1000)
    assert kind == "success"
    for teil in ("Bei 1.000 Auswertungen", "0,01910 statt 0,02160", "Varianzverhältnis 0,78 gemessen, 0,75 vorhergesagt", "ρ = −0,250, exakt", "um 25 %"):
        assert teil in text, teil


def test_spread_verdikt_schwellen_sind_0_95_und_1_05():
    assert ev.verdict_spread(_analysis(-0.05), 100)[0] == "info"           # 0,95 genau: kein Gewinn mehr
    assert ev.verdict_spread(_analysis(-0.0501), 100)[0] == "success"
    assert ev.verdict_spread(_analysis(0.05), 100)[0] == "info"            # 1,05 genau
    kind, text = ev.verdict_spread(_analysis(0.13, var_ratio=1.2), 100)
    assert kind == "warning" and "ERHÖHT die Varianz um 13 %" in text


def test_spread_verdikt_nennt_die_quelle_der_korrelation():
    assert "aus den Paaren geschätzt" in ev.verdict_spread(_analysis(-0.2, exact=False), 100)[1]


def test_n_sweep_verdikt():
    exp = {"rows": [{"ratio": 0.8}, {"ratio": 0.9}, {"ratio": 0.85}], "slope_plain": -0.5, "slope_anti": -0.49, "rho_exact": -0.15}
    kind, text = ev.verdict_n_sweep(exp)
    assert kind == "info" and "zwischen 0,80 und 0,90" in text and "−0,50 (unabhängig) und −0,49 (gespiegelt)" in text and "Konstante, nicht die Rate" in text and "10 %" in text


def test_rho_kurve_verdikt():
    rows = [{"p": 0.5, "factor": 0.94}, {"p": 0.6, "factor": 0.81}, {"p": 0.99, "factor": 0.9996}]
    kind, text = ev.verdict_rho_curve({"rows": rows}, "zusammenhaengend")
    assert kind == "info"
    for teil in ("am kleinsten bei p = 0,60 (0,810)", "am größten bei p = 0,99 (1,000)", "bei p = 0,99 (seltene Ausfälle) ist er 0,9996"):
        assert teil in text, teil
    assert "ρ ≤ 0 garantiert" in text
    assert "ρ ≤ 0 garantiert" not in ev.verdict_rho_curve({"rows": rows}, "zwei_teile")[1]


def test_funktionen_verdikt_warnt_ueber_eins():
    rows = [{"name": "A", "factor": 0.81}, {"name": "B", "factor": 1.13}, {"name": "C", "factor": 2.0}]
    kind, text = ev.verdict_functions({"rows": rows})
    assert kind == "warning" and "A: 0,810" in text and "B: 1,130" in text and "Über 1 liegt: B, C" in text
    kind, text = ev.verdict_functions({"rows": rows[:1]})
    assert kind == "info" and "Keine der vier Funktionen" in text
    assert ev.verdict_functions({"rows": [{"name": "A", "factor": 1.0}]})[0] == "info"      # genau 1 ist kein Verlust


def test_crn_verdikt_nennt_gewinn_formelgenauigkeit_und_entwurfsvergleich():
    rows = [{"delta": 0.2, "factor": 1.0, "var_crn": 0.19, "var_crn_meas": 0.1895},
            {"delta": 0.01, "factor": 14.5, "var_crn": 0.025, "var_crn_meas": 0.0253}]
    design = {"edge": (0, 5), "dR": 0.0374, "factor": 10.0}
    kind, text = ev.verdict_crn({"rows": rows, "design": design})
    assert kind == "success"
    for teil in ("bei δ = 0,20 ist die Differenz 1,0-mal ruhiger", "bei δ = 0,01 14,5-mal", "höchstens 1,2 % genau", "Zusatzkante 0–5 (ΔR = 0,0374)", "ist der Gewinn 10,0"):
        assert teil in text, teil


def test_sync_verdikt_nennt_korrelationen_und_faktor():
    exp = {"fest": {"corr": 0.9, "var": 0.036, "mean": 0.0375}, "lage": {"corr": 0.001, "var": 0.358, "mean": 0.0385}, "var_ind": 0.3587, "dR": 0.0374, "edge": (0, 5)}
    kind, text = ev.verdict_sync(exp)
    assert kind == "warning"
    for teil in ("Korrelation der beiden Ziele 0,900", "Varianz der Differenz 0,0360", "Korrelation 0,001, Varianz 0,3580", "unabhängig wäre 0,3587", "9,9-mal so groß", "unverzerrt (0,0385 gegen 0,0375)"):
        assert teil in text, teil
