"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

GRIDS = {"2x3": (2, 3), "3x3": (3, 3), "2x5": (2, 5), "3x4": (3, 4)}
DEFAULT_GRID = "3x4"
LIVE_EDGE_LIMIT = 17          # Aufzählung aller Kantenmengen live bis hierher (2^17 = 131 072, rund 0,3 s); eigenes Netz ebenso
EXACT_PAIR_EDGE_LIMIT = 12    # exakte gemeinsame Verteilung eines Paares: 3^m Kategorien, 3^12 = 531 441
DESIGN_EDGE_LIMIT = LIVE_EDGE_LIMIT + 1   # der Entwurfsvergleich zählt auch das Netz mit der Zusatzkante auf (2^18 = 262 144 Kantenmengen, rund 0,6 s)
CUSTOM_EDGE_LIMIT = 17
CUSTOM_NODE_LIMIT = 14

TARGETS = {"zusammenhaengend": "Netz hängt zusammen", "zwei_teile": "Netz zerfällt in genau zwei Teile"}
DEFAULT_TARGET = "zusammenhaengend"

P_MIN, P_MAX, DEFAULT_P = 0.50, 0.99, 0.70
N_CHOICES = (100, 300, 1000, 2000)        # Zahl der Auswertungen je Schätzung (gerade: das Paar zählt zwei)
N_MIN, N_MAX, DEFAULT_N = N_CHOICES[0], N_CHOICES[-1], 1000
SEED_MAX, DEFAULT_SEED = 999999, 35
REPS = 200                                # Wiederholungen je Schätzung (feste Ströme), Rauschen der Varianzverhältnisse 2/sqrt(REPS − 1) ≈ 14 %

# Experimente
SWEEP_N = (100, 300, 1000, 3000)
SWEEP_REPS = 200
RHO_PS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
RHO_PAIRS = 100_000                       # Paare je p, wenn die gemeinsame Verteilung nicht exakt aufgezählt wird
FUNC_PAIRS = 100_000
SCALAR_DRAWS = 1_000_000
CRN_DELTAS = (0.2, 0.1, 0.05, 0.02, 0.01)
CRN_DRAWS = 100_000
P_CAP = 0.999                             # p + delta wird auf diesen Wert begrenzt

COLOR_PLAIN = "#4c78a8"
COLOR_ANTI = "#2F6B65"
COLOR_TRUTH = "#222222"
COLOR_BAD = "#C0392B"
COLOR_OTHER = "rgba(150,150,150,0.55)"

PRESETS = {
    "Standardfall (Voreinstellung)": {"gitter": DEFAULT_GRID, "p": DEFAULT_P, "n": DEFAULT_N, "ziel": DEFAULT_TARGET, "seed": DEFAULT_SEED},
    "Kräftiger Gewinn": {"gitter": "3x3", "p": 0.60, "n": 1000, "ziel": DEFAULT_TARGET, "seed": DEFAULT_SEED},
    "Seltenes Ereignis": {"gitter": DEFAULT_GRID, "p": 0.99, "n": 2000, "ziel": DEFAULT_TARGET, "seed": DEFAULT_SEED},
    "Nicht monotones Ziel": {"gitter": "2x3", "p": 0.60, "n": 1000, "ziel": "zwei_teile", "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Gitter 3×4, jede Kante hält mit 70 %, 1000 Auswertungen: das Netz hängt zusammen. Mittlerer Gewinn der Spiegelung.",
    "Kräftiger Gewinn": "Gitter 3×3, p = 0,60: nahe am günstigsten p; die Spiegelung senkt die Varianz um etwa ein Viertel.",
    "Seltenes Ereignis": "Gitter 3×4, p = 0,99: der Ausfall ist selten, die Spiegelung bringt praktisch nichts.",
    "Nicht monotones Ziel": "Gitter 2×3, p = 0,60: das Ziel „genau zwei Teile“ ist nicht monoton, die Spiegelung kann dort sogar schaden.",
}
