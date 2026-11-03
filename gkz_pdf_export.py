"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import gkz_constants as C
from gkz_evaluation import Analysis, Settings, de, verdict_spread

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "€": "EUR", "„": '"', "“": '"', "²": "^2", "³": "3", "×": "x", "√": "Wurzel", "≈": "~",
            "σ": "Sigma", "ρ": "rho", "∞": "unendlich", "Σ": "Summe", "δ": "delta", "Δ": "Delta", "½": "1/2", "̂": ""}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings: Settings, a: Analysis) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Gekoppelte Ziehungen: antithetische Paare"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(
        f"{a.netz.name}, {a.netz.n} Knoten, {a.netz.m} Kanten; Ziel: {C.TARGETS[settings.ziel]}; jede Kante hält mit Wahrscheinlichkeit {de(settings.p, 2)}; "
        f"{de(settings.n, 0)} Auswertungen je Schätzung, {C.REPS} Wiederholungen, Strom {settings.seed}. Alles simuliert, keine Messdaten."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 9)
    widths = (70, 40, 40)
    for h, w in zip(("Größe", "unabhängig", "gespiegelt"), widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for label, x, y in (("Mittel der Schätzungen", a.mean_plain, a.mean_anti), ("Streuung der Schätzung", a.sd_plain, a.sd_anti)):
        for text, w in zip((label, de(x, 5), de(y, 5)), widths):
            pdf.cell(w, 6, _s(text), border=1)
        pdf.ln()
    pdf.ln(2)
    pdf.multi_cell(0, 5, _s(f"Wahrheit (Aufzählung aller Kantenmengen): {de(a.truth, 5)}. Varianzverhältnis gemessen {de(a.var_ratio, 3)}, vorhergesagt {de(a.predicted_ratio, 3)}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.multi_cell(0, 5, _s(verdict_spread(a, settings.n)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: gekoppelte-ziehungen-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
