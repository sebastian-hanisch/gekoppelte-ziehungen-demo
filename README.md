# Gekoppelte Ziehungen: antithetische Paare und gemeinsame Zufallszahlen – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-gekoppelte-ziehungen-demo.streamlit.app/)** -->

Zweites Stück der Linie **„Monte-Carlo-Schätzung und Varianzreduktion“** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Linie wird erst vollständig veröffentlicht). Nachfolger von [monte-carlo-schaetzer-demo](https://github.com/sebastian-hanisch/monte-carlo-schaetzer-demo): Dort stand der Schätzer mit seinem Fehler, hier wird bei gleichem Aufwand **die Varianz gesenkt**, indem dieselben Zufallszahlen **gekoppelt** werden. Zwei Kopplungen mit derselben Währung, der **Korrelation**: Ein **gespiegeltes Paar** (u und 1−u) macht sie *innerhalb* einer Schätzung negativ, **gemeinsame Zufallszahlen** machen sie *zwischen* zwei Schätzungen positiv.

Vehikel ist wie in Stück 1 die **Zuverlässigkeit eines Gitternetzes**: Jede Kante hält unabhängig mit Wahrscheinlichkeit p; Ziel ist, dass das Netz zusammenhängt (monoton in den haltenden Kanten) oder in genau zwei Teile zerfällt (nicht monoton). Bis 17 Kanten ist die Wahrheit **exakt** aufgezählt, und für bis zu 12 Kanten auch die exakte Korrelation eines Paares.

## Warum dieses Problem

„Spiegele die Zufallszahlen, dann sinkt die Varianz“ ist ein beliebtes Rezept, und „nimm für den Vergleich dieselben Zufallszahlen“ ein zweites. Beide stimmen nicht pauschal: Die Varianz einer Summe ist Var A + Var B + 2 Cov, die einer Differenz Var A + Var B − 2 Cov; der Gewinn ist also die **Korrelation**, und die hängt an der **Funktion** und am **Abstand der Varianten**. Dieselbe Spiegelung senkt die Varianz von u² auf ein Achtel, die eines Ja-Nein-Ziels je nach Netz nur um 18 bis 34 %, bei seltenen Ausfällen gar nicht, und verdoppelt sie bei einer um ½ symmetrischen Funktion. Gemeinsame Zufallszahlen machen die Differenz zweier Varianten um den Faktor ≈ 1/ΔR ruhiger, aber nur bei richtiger Zuordnung der Zufallszahlen.

## Befunde und Korrekturen gegenüber dem Plan

Vor dem Bau wurden die Hypothesen der Gliederung an Messreihen geprüft (Vormessung im Website-Repo, `docs/planung/monte-carlo-planung/vormessung/`). Was sich geändert hat:

- **„Gewinn bei monotonen Größen, Verlust bei nicht monotonen“ stimmt nur im Kern.** Für ein monotones Ziel ist ρ ≤ 0 garantiert, aber der Gewinn ist für ein Ja-Nein-Ziel klein und **verschwindet bei seltenen Ereignissen**. Ein echter Verlust (Faktor 2) entsteht erst bei einer um ½ symmetrischen Funktion; das nicht monotone Netzziel „genau zwei Teile“ schadet nur leicht (Faktor 1,13 bei 2×3, p = 0,6) und hilft bei p = 0,5 sogar kräftig.
- **Der Gewinn hängt am Netz.** Der Plan nannte „höchstens 19 %“; das gilt für 3×4 (und 4×4). Bei p = 0,6 ist der Faktor 0,656 für 2×3, 0,746 für 3×3 und 0,813 für 3×4: kleinere Netze gewinnen mehr.
- **Fehler in der Gliederung berichtigt:** Dort stand, `output-analysis-demo` (Warteschlangen 2) nutze gemeinsame Zufallszahlen nur als Hinweis. Die Demo hat ein gemessenes Experiment dazu (Faktor 63 bis 1,7 je nach Auslastung). Dieses Stück wiederholt es nicht, sondern zeigt, was dort fehlt: die **geschlossene Formel** Var(X_B − X_A) = ΔR(1−ΔR) für monoton gekoppelte Varianten, den **Entwurfsvergleich** und die **Synchronisationsfalle**.
- **Die Falle liegt im Layout aus Stück 1:** Dort steht die Zufallszahl einer Kante an der Stelle `i·m + e` des Stroms, hängt also von der Kantenzahl m ab. Bei einer Zusatzkante verschiebt sich alles, die „gemeinsamen“ Zahlen sind unabhängig. Dieses Stück liest den Strom mit **fester Schrittweite**.
- **Beim Bau gefunden:** Der Entwurfsvergleich braucht die Aufzählung des Netzes **mit** Zusatzkante, beim 3×4-Gitter also 18 Kanten; die Live-Grenze der Aufzählung liegt deshalb für diesen Vergleich eine Kante höher (18, 262 144 Kantenmengen).

## Modell

- **Netz.** Gitter 2×3, 3×3, 2×5 und 3×4 (7, 12, 13, 17 Kanten); Kante e hält in Ziehung i, wenn ihre Zufallszahl u_(i,e) < p. Ziele: *zusammenhängend* (genau eine Komponente) und *genau zwei Teile* (genau zwei Komponenten).
- **Zufallsstrom.** SplitMix64, zählerbasiert: Ausgabe i hängt nur von Strom und i ab. Die Zahl der Kante e in Ziehung i ist Nr. i·s + e (s = Schrittweite).
- **Gespiegeltes Paar.** Das zweite Netz benutzt 1 − u: Kante e hält dort bei u_e > 1 − p. Verglichen werden **N unabhängige Auswertungen** mit **N/2 gespiegelten Paaren** (ebenfalls N Auswertungen); je 200 Wiederholungen mit festen Strömen. Die Varianz des Paar-Schätzers ist (1 + ρ)-mal so groß wie die der unabhängigen, ρ = Corr(X(u), X(1−u)).
- **Exakte Korrelation.** Je Kante gibt es bei p > ½ drei Fälle: hält in beiden Netzen (Wahrscheinlichkeit 2p−1), nur im ersten (1−p), nur im zweiten (1−p); bei p ≤ ½: in keinem (1−2p), nur im ersten (p), nur im zweiten (p). Aufzählung aller 3^m Kombinationen (bis 12 Kanten) liefert E X, E X' und E X X'; darüber wird ρ aus 100 000 Paaren geschätzt. Für einen Pfad mit m Kanten (p > ½) ist ρ = ((2p−1)^m − p^(2m)) / (p^m (1−p^m)).
- **Gemeinsame Zufallszahlen.** Zwei Varianten (p und p + δ oder dasselbe Netz mit einer Zusatzkante am Ende der Kantenliste) bekommen dieselben u. Ist B monoton gekoppelt (hält mindestens, wo A hält), liegt D = X_B − X_A in {0, 1}; dann ist E D = ΔR und **Var D = ΔR(1−ΔR)**, unabhängig wäre sie R_A(1−R_A) + R_B(1−R_B).
- **Skalare.** f(u) = u² (monoton, glatt) und f(u) = (2u−1)² (symmetrisch: f(1−u) = f(u)).
- Alles ist **erfunden**; Netze und Funktionen stehen nicht für Messdaten. Alle Formeln im Expander „📐 Mathematische Formulierung“.

## Methodik

Fünf Experimente auf dem gewählten Netz, jedes mit live berechnetem Verdikt: (1) Streuung gegen N, (2) Korrelation gegen p, (3) welche Funktion, (4) gemeinsame Zufallszahlen (Gewinn gegen δ und Entwurfsvergleich), (5) die Synchronisationsfalle. Dazu eine Spielwiese für ein eigenes Netz (bis 17 Kanten) mit optionaler Zusatzkante. Die Experimente laufen auf Knopfdruck (je einige Sekunden) und werden zwischengespeichert.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`; Ströme sind fest (nie ein einzelner Seed). Das Rauschen der gemessenen Varianzverhältnisse beträgt bei 200 Wiederholungen rund 10 %: Abweichungen dieser Größe sind kein Befund; deshalb steht daneben das aus der Korrelation vorhergesagte Verhältnis 1 + ρ.

| Frage | Befund | Test |
|---|---|---|
| **Standardfall** (3×4, p = 0,70, N = 1000, „zusammenhängend“, Strom 35) | Wahrheit 0,45602; Streuung 0,01569 (unabhängig) gegen 0,01436 (gespiegelt); Varianzverhältnis **0,84 gemessen, 0,82 vorhergesagt** (ρ = −0,178 aus den Paaren); die Spiegelung senkt die Varianz um 18 % | `test_preset_standardfall` |
| **Kräftiger Gewinn** (3×3, p = 0,60) | ρ = **−0,2545 exakt**, vorhergesagt 0,7455, gemessen 0,68; Wahrheit 0,26791; um 25 % | `test_preset_kraeftiger_gewinn_exakt` |
| **Seltenes Ereignis** (3×4, p = 0,99, N = 2000) | Wahrheit 0,99958, ρ ≈ −0,0004, vorhergesagt **0,9996**, gemessen 1,05 (Rauschen): praktisch nichts | `test_preset_seltenes_ereignis` |
| **Nicht monotones Ziel** („genau zwei Teile“, 2×3, p = 0,60) | ρ = **+0,1312 exakt**, vorhergesagt 1,1312, gemessen 1,18; Wahrheit 0,34836; die Spiegelung **erhöht** die Varianz um 13 % | `test_preset_nicht_monotones_ziel_exakt` |
| Ändert die Spiegelung die Rate? (3×4, p = 0,70, N = 100 / 300 / 1000 / 3000) | Varianzverhältnis 0,57 / 0,74 / 0,84 / 0,75 (Rauschen etwa 14 %); ρ zwischen −0,21 und −0,17 bei allen N; Steigung der Streuung −0,50 (unabhängig) und −0,46 (gespiegelt), Soll −0,50: die **Konstante** ändert sich, nicht die Rate | `test_experiment_1_verhaeltnisse_und_steigung` |
| Gewinn gegen p (3×4, aus je 100 000 Paaren), p = 0,5 / 0,6 / 0,7 / 0,8 / 0,9 / 0,95 / 0,99 | Varianzfaktor **0,936 / 0,813 / 0,814 / 0,916 / 0,985 / 0,993 / 0,9996**; am kleinsten bei p = 0,6 | `test_experiment_2_kurve_fuer_3x4_aus_paaren` |
| dasselbe, 3×3 (exakt aufgezählt) | 0,882 / 0,746 / 0,789 / 0,904 / 0,981 / 0,996 / 0,9999 | `test_experiment_2_kurve_fuer_3x3_exakt` |
| Gewinn hängt am Netz (p = 0,6, exakt bzw. aus Paaren) | 2×3: **0,656**, 3×3: **0,746**, 3×4: 0,813 | `test_gewinn_haengt_am_netz_bei_p_gleich_0_6` |
| Pfad, geschlossene Formel | m = 3, p = 0,6: ρ = −0,2283; m = 5, p = 0,8: −0,1344; m = 1: ρ = −(1−p)/p = −0,25 bei p = 0,8; stimmt mit der Aufzählung überein | `test_pfad_geschlossene_formel_zahlen` |
| Welche Funktion? (3×4, p = 0,70) | Varianzfaktor: „zusammenhängend“ **0,819**, „genau zwei Teile“ **1,015**, u² **0,125** (ρ = −7/8), (2u−1)² **2,000** (ρ = +1) | `test_experiment_3_funktionen_am_standardfall`, `test_skalare_exakt` |
| dasselbe, 2×3, p = 0,60 | „zusammenhängend“ 0,656, „genau zwei Teile“ **1,131** (exakt) | `test_experiment_3_funktionen_beim_nicht_monotonen_preset_exakt` |
| „Genau zwei Teile“, 3×3 (exakt) | ρ = +0,0268 bei p = 0,6 und +0,0024 bei p = 0,7: nur leicht positiv | `test_exakte_paarkorrelation_bei_zwei_teilen_3x3_ist_klein_positiv` |
| Gemeinsame Zufallszahlen, p gegen p + δ (3×4, p = 0,70, je 100 000 Ziehungen), δ = 0,2 / 0,1 / 0,05 / 0,02 / 0,01 | Gewinnfaktor nach der Formel ΔR(1−ΔR): **1,2 / 2,12 / 3,84 / 8,92 / 17,38**; gemessen 1,2 / 2,11 / 3,82 / 8,79 / 17,0; Formel und Messung der Varianz weichen höchstens 3 % ab | `test_experiment_4_gewinn_gegen_delta_und_entwurfsvergleich` |
| Entwurfsvergleich: Kante 0–5 ins 3×4-Gitter (p = 0,70) | R = 0,4560 → 0,5079, ΔR = 0,0519; Gewinnfaktor **10,1**; Korrelation der beiden Ziele 0,90 | dasselbe |
| Synchronisationsfalle (dieselbe Zusatzkante) | feste Zuordnung: Korrelation **0,903**, Varianz der Differenz 0,0485; Zuordnung nach Lage im Strom: Korrelation **−0,002**, Varianz 0,4988 (unabhängig 0,4980), also **10,3-mal** so groß; der Mittelwert bleibt unverzerrt (0,0518 gegen 0,0511) | `test_experiment_5_synchronisationsfalle` |
| Spielwiese, Standardnetz (0-1, 1-2, 2-3, 3-0, 0-2) | Wahrheit 0,77518, ρ = −0,1971 (exakt), Faktor 0,803; Zusatzkante 1–3: R 0,7752 → 0,8925, ΔR = 0,1173, Gewinnfaktor 2,61 | `test_spielwiese_standardnetz` |

## Ehrliche Grenzen

- **Nur an erfundenen Gittern und Funktionen geprüft.** Die Wahrheit ist exakt, ob die Aussagen auf reale Simulationen übertragbar sind, zeigt die Demo nicht.
- **Der Gewinn für ein Ja-Nein-Ziel ist klein.** Das ist ein Befund, kein Mangel: Die Spiegelung ist für glatte Größen stark (u²), für Schwellenfunktionen schwach. Für das Ja-Nein-Ziel gibt es bessere Wege (bedingtes Monte Carlo, Stück 6; Importance Sampling bei seltenen Ereignissen, Stück 7).
- **Die exakte Korrelation gibt es nur bis 12 Kanten** (3^12 = 531 441 Kategorienfolgen); für das 2×5- und das 3×4-Gitter wird ρ aus 100 000 Paaren geschätzt (Rauten im Experiment 2). Das Varianzverhältnis aus 200 Wiederholungen rauscht um rund 10 %, beim kleinsten N (100) um etwa 14 %.
- **Die Formel ΔR(1−ΔR) gilt nur für monoton gekoppelte Varianten** (p und p + δ, eine Kante mehr). Für Vergleiche, in denen die eine Variante nicht überall besser ist, gibt es sie nicht.
- **Das Ziel „genau zwei Teile“ ist nur ein Beispiel** eines nicht monotonen Ziels, mit kleinem Verlust. Der große Verlust (Faktor 2) steckt in der symmetrischen Skalarfunktion.
- **Experimente 4 und 5 nutzen das Ziel „zusammenhängend“**, da dort die Kopplung monoton ist.
- **Zeiten werden nicht gemessen**, nur die Zahl der Auswertungen: Ein Paar kostet zwei Auswertungen und einen Satz Zufallszahlen; ob das Erzeugen der Zufallszahlen oder die Auswertung teurer ist, hängt vom Problem ab.
- **Literaturangaben im Mathematik-Expander** (Hammersley und Morton 1956; Law und Kelton; Glasserman 2003) stammen aus dem Gedächtnis und sind nicht eingesehen. Die Websuche zum Stand der Technik im Plan stützt sich auf Suchauszüge, nicht auf gelesene Volltexte.
- **Mit echtem Mausklick** sind nur die Knöpfe der Experimente im Browser angesehen; AppTest deckt Regler, Presets, Permalink, PDF und Spielwiese ab.

## Verifikation

- **Kern** (`tests/test_algorithm.py`): Zufallsstrom (Startwert 0xE220A8397B1DCDAF, vektorisiert gleich Ganzzahl-Referenz), Stromzuordnung Nr. i·Schrittweite + e, Komponenten und Ziele auf Mini-Instanzen (Dreieck, Kreis, Pfad, K4), Aufzählung beider Ziele, exakte Paarmomente von Hand (Dreieck bei p = 0,6: E X X' = 0,296 aus den Kategorien), Pfadformel, unabhängige Neuberechnung des gespiegelten Schätzers, Bernoulli-Varianz der gekoppelten Differenz.
- **Orakel** (`tests/test_oracle_gkz.py`): Komponenten gegen `networkx.number_connected_components`; Aufzählung gegen Matrix-Baum-Satz und gegen Bruchrechnung mit Union-Find (beide Ziele); exakte Paarmomente gegen eine eigene Bruchrechnung über alle Kategorienfolgen (itertools.product, vier Werte von p, beide Ziele) und gegen eine Simulation mit 150 000 Paaren; Var(X_B − X_A) = ΔR(1−ΔR) gegen die exakte gekoppelte Verteilung in Bruchrechnung.
- **Eigenschaften** (`tests/test_properties.py`): ρ ≤ 0 für das monotone Ziel in jeder Zelle eines p-Rasters (exakt, 2×3 und 3×3, auch p ≤ ½); beide Schätzer erwartungstreu; gepoolte Korrelation gleich exakter; Differenz in {0, 1} bei gemeinsamen Zahlen; Zusatzkante mit festen Zufallszahlen nie schlechter; falsche Zuordnung unverzerrt, aber ungekoppelt; Gewinnfaktor wächst mit kleinerem δ.
- **Randfälle** (`tests/test_edge_cases.py`): Regler an den Grenzen, beide Ziele auf allen Gittern, nie eintretendes Ziel (ρ undefiniert), ungerade Auswertungszahl, Eingabefehler der Spielwiese und der Zusatzkante, Worst-Case am Regler-Maximum.
- **Verdikte** (`tests/test_verdicts.py`): jede Schwelle (0,95 und 1,05) und jede genannte Zahl an von Hand gesetzten Werten, deutsches Zahlenformat.
- **Presets, Layout, Oberfläche:** Presets vollständig, Werte innerhalb der Grenzen und Optionen; Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben; AppTest für jeden Button, Regler an Minimum und Maximum, die fünf Experimente, die Spielwiese, den PDF-Export und den Permalink.
- **Laufzeit-Importe, Footer, Claims.**

Zahl der Tests: siehe `pytest --collect-only -q tests` (Stand beim Bau: 138 Tests).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `gkz_algorithm.py` | Kern: Zufallsstrom, Komponenten, Aufzählung, Paare, Schätzer, Kopplung, Skalare |
| `gkz_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `gkz_evaluation.py` | Analyse, fünf Experimente, Spielwiese, Verdikte, Zahlenformat |
| `gkz_layout.py` | Fester Kopf (Stylesheet) |
| `gkz_pdf_export.py` | PDF der aktuellen Schätzung (fpdf2) |
| `gkz_presets.py` | Permalink (Begrenzen, Einrasten), Presets, Zufallsstrom-Knopf |
| `gkz_scenario.py` | Netze: Gitter, Zusatzkante, eigene Kantenliste mit Eingabeprüfung |
| `gkz_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy, networkx (nur Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Mini-Instanzen, Kern, Orakel, Eigenschaften, Randfälle, Verdikte, Claims, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Kontrollvariaten, Schichtung, Latin Hypercube, bedingtes Monte Carlo:** Stücke 3 bis 6 der Linie (die Spiegelung ist die einfachste Schichtung mit zwei Teilen).
- **Importance Sampling, Splitting für seltene Ereignisse:** Stück 7 und 8 (Splitting als Verfahren steht in `splitting-demo`, Warteschlangen-Linie).
- **Gradienten schätzen mit gemeinsamen Zufallszahlen (Differenzenquotient):** Stück 11; der Fehler eines Differenzenquotienten wächst mit 1/δ, hier nur als Gewinnfaktor gegen δ sichtbar.
- **Gemeinsame Zufallszahlen für Warteschlangensysteme:** `output-analysis-demo` (Warteschlangen 2), mit eigener Messung.
- **Latin Hypercube und Quasi-Monte-Carlo als Alternative zur Spiegelung:** Stück 5 und 9.
- **Spiegelung mit mehr als zwei Teilen (Schichten):** Stück 5.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
