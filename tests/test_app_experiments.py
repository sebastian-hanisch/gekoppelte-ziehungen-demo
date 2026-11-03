"""AppTest der Experimente, der Spielwiese, des PDF-Exports und des Permalinks (Einzelheiten, die der allgemeine Rauchtest nicht trifft)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import gkz_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at) -> str:
    return " ".join(e.value for kind in (at.success, at.info, at.warning, at.error) for e in kind)


@pytest.mark.parametrize("key,fragment", [
    ("run_sweep", "Konstante, nicht die Rate"),
    ("run_rho", "Der Varianzfaktor ist am kleinsten bei p ="),
    ("run_func", "Varianzfaktoren (unter 1 ist ein Gewinn)"),
    ("run_crn", "Die Formel Var(X_B − X_A) = ΔR(1−ΔR) trifft die Messung"),
    ("run_sync", "verschiebt sich alles"),
])
def test_experiment_auf_klick_zeigt_sein_verdikt(key, fragment):
    at = _app()
    assert fragment not in _texte(at)                       # vor dem Klick steht noch nichts da
    at.button(key=key).click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert fragment in _texte(at)


def test_experiment_bleibt_nach_einer_aenderung_der_einstellungen_sichtbar():
    at = _app()
    at.button(key="run_func").click().run(timeout=TIMEOUT)
    at.sidebar.select_slider(key="n_select").set_value(300).run(timeout=TIMEOUT)
    assert not at.exception
    assert "Varianzfaktoren (unter 1 ist ein Gewinn)" in _texte(at)


def test_kernfrage_verdikt_wechselt_mit_dem_preset():
    at = _app()
    assert "senkt die Varianz" in _texte(at)
    at.sidebar.button(key="preset_Seltenes Ereignis").click().run(timeout=TIMEOUT)
    assert "praktisch nichts" in _texte(at)
    at.sidebar.button(key="preset_Nicht monotones Ziel").click().run(timeout=TIMEOUT)
    assert "ERHÖHT die Varianz" in _texte(at)


def test_ziel_wechsel_laeuft_fuer_beide_ziele_durch():
    at = _app()
    for ziel in C.TARGETS:
        at.sidebar.selectbox(key="ziel_select").set_value(ziel).run(timeout=TIMEOUT)
        assert not at.exception, (ziel, [str(e) for e in at.exception])


def test_spielwiese_gueltiges_netz_zusatzkante_und_fehlermeldungen():
    at = _app()
    assert not at.error
    assert "Entwurfsvergleich mit der Zusatzkante 1–3" in _texte(at)
    at.text_input(key="eigene_zusatzkante").set_value("").run(timeout=TIMEOUT)
    assert not at.exception and "Entwurfsvergleich" not in _texte(at)
    at.text_input(key="eigene_zusatzkante").set_value("0-1").run(timeout=TIMEOUT)          # gibt es schon
    assert not at.exception and "gibt es schon" in " ".join(e.value for e in at.error)
    at.text_area(key="eigene_kanten").set_value("0-0").run(timeout=TIMEOUT)
    assert not at.exception and "Schleife" in " ".join(e.value for e in at.error)


def test_spielwiese_nicht_zusammenhaengendes_netz_laeuft_durch():
    at = _app()
    at.text_area(key="eigene_kanten").set_value("0-1\n2-3").run(timeout=TIMEOUT)
    at.text_input(key="eigene_zusatzkante").set_value("").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "nie oder immer ein" in _texte(at)


def test_pdf_wird_erst_nach_klick_erzeugt():
    at = _app()
    assert not at.get("download_button")
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.get("download_button")) == 1
    at.sidebar.select_slider(key="n_select").set_value(300).run(timeout=TIMEOUT)     # andere Einstellung: altes PDF nicht mehr anbieten
    assert not at.get("download_button")


def test_permalink_wird_begrenzt_und_eingerastet():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["p"] = "5"
    at.query_params["n"] = "700"
    at.query_params["g"] = "9x9"            # unbekanntes Gitter: verworfen
    at.query_params["z"] = "quatsch"        # unbekanntes Ziel: verworfen
    at.query_params["seed"] = "-4"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.slider(key="p_slider").value == C.P_MAX
    assert at.sidebar.select_slider(key="n_select").value == 1000                    # 700 liegt näher an 1000 als an 300
    assert at.sidebar.selectbox(key="gitter_select").value == C.DEFAULT_GRID
    assert at.sidebar.selectbox(key="ziel_select").value == C.DEFAULT_TARGET
    assert at.sidebar.number_input(key="seed_input").value == 0
