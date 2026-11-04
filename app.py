"""Gekoppelte Ziehungen: antithetische Paare und gemeinsame Zufallszahlen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 2 der Linie "Monte-Carlo-Schätzung und Varianzreduktion": zwei Kopplungen derselben Zufallszahlen. Die gespiegelte Ziehung (u und 1-u) macht die
Korrelation INNERHALB einer Schätzung negativ, gemeinsame Zufallszahlen machen sie ZWISCHEN zwei Schätzungen positiv. Vehikel ist die Zuverlässigkeit
eines Gitternetzes, die Wahrheit ist exakt aufgezählt.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import gkz_constants as C
import gkz_evaluation as ev
from gkz_evaluation import Settings, de
from gkz_layout import dunkles_design, kopf_css
from gkz_pdf_export import build_pdf
from gkz_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    options,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from gkz_scenario import extra_edge, named_netz, parse_edges, parse_extra_edge
from gkz_visualization import (
    build_crn,
    build_functions,
    build_n_sweep,
    build_netz,
    build_rho_curve,
    build_spread,
    build_sync,
)

st.set_page_config(page_title="Gekoppelte Ziehungen: antithetische Paare und gemeinsame Zufallszahlen – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Alle Kantenmengen aufzählen …")
def _counts(gitter):
    return ev.netz_counts(named_netz(gitter))


@st.cache_data(show_spinner="Schätzungen wiederholen …")
def _analysis(settings, counts):
    return ev.analyse(settings, counts=counts)


@st.cache_data(show_spinner="Streuung gegen N messen …")
def _n_sweep(settings):
    return ev.n_sweep(settings, named_netz(settings.gitter))


@st.cache_data(show_spinner="Korrelation gegen p messen …")
def _rho_curve(settings):
    return ev.rho_curve(settings, named_netz(settings.gitter))


@st.cache_data(show_spinner="Vier Funktionen vergleichen …")
def _functions(settings):
    return ev.function_table(settings, named_netz(settings.gitter))


@st.cache_data(show_spinner="Gemeinsame Zufallszahlen messen …")
def _crn(settings, counts):
    return ev.crn_experiment(settings, named_netz(settings.gitter), counts)


@st.cache_data(show_spinner="Zuordnung der Zufallszahlen prüfen …")
def _sync(settings):
    return ev.sync_result(settings, named_netz(settings.gitter))


@st.cache_data(show_spinner=False)
def _custom(text, extra_text, settings):
    netz = parse_edges(text)
    extra = parse_extra_edge(extra_text, netz)
    return netz, extra, ev.custom_result(settings, netz, extra)


def _gitter_label(key):
    rows, cols = C.GRIDS[key]
    return f"{rows}×{cols} ({rows * (cols - 1) + cols * (rows - 1)} Kanten)"


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in gkz_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔗 Gekoppelte Ziehungen: antithetische Paare und gemeinsame Zufallszahlen")
st.markdown(
    """
Zwei **Kopplungen** derselben Zufallszahlen, und die **Korrelation** ist ihre Währung: Die Varianz einer Summe ist Var A + Var B + 2 Cov, die einer Differenz Var A + Var B − 2 Cov.
Ein **gespiegeltes Paar** (u und 1−u) macht die Korrelation *innerhalb* einer Schätzung negativ und senkt so ihre Streuung; **gemeinsame Zufallszahlen** machen sie *zwischen* zwei Schätzungen positiv und senken die Streuung einer Differenz.
Der Gewinn hängt nicht am Verfahren, sondern an der **Funktion** und am **Abstand der Varianten**; hier ist die Wahrheit bekannt (Zuverlässigkeit eines Gitternetzes, alle Kantenmengen aufgezählt).

Weiter unten: **🔬 fünf Experimente**, **🧪 Spielwiese**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Kopplung", expanded=True):
    st.markdown(
        """
1. **Zufallszahlen:** Jede Kante e bekommt in Ziehung i eine Zahl u_e. Sie hält im Netz, wenn u_e < p.
2. **Gespiegelt:** Das zweite Netz des Paares benutzt 1 − u_e: Kante e hält dort, wenn u_e > 1 − p. Ein Paar kostet zwei Auswertungen, aber nur einen Satz Zufallszahlen.
3. **Aufwand gleich:** Verglichen werden N unabhängige Auswertungen mit N/2 gespiegelten Paaren. Die Varianz des Paar-Schätzers ist (1 + ρ)-mal so groß wie die der unabhängigen, mit ρ = Korrelation von X(u) und X(1−u).
4. **Gemeinsame Zufallszahlen:** Zwei Varianten (zum Beispiel p und p + δ, oder dasselbe Netz mit einer Zusatzkante) bekommen **dieselben** u. Für monoton gekoppelte Varianten ist die Differenz 0 oder 1 und ihre Varianz genau ΔR(1−ΔR).
5. **Warum das funktioniert, wo es funktioniert:** Der Gewinn ist −ρ (innen) beziehungsweise +ρ (zwischen den Varianten); beides hängt an der Funktion, nicht am Verfahren.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz und Ziel**")
    gitter = st.selectbox("Gitter", options("gitter_select"), format_func=_gitter_label, key="gitter_select", help="Knoten in Zeilen und Spalten, Kanten waagerecht und senkrecht. Die Wahrheit wird live aufgezählt.")
    ziel = st.selectbox("Ziel", options("ziel_select"), format_func=C.TARGETS.get, key="ziel_select", help="Was gezählt wird: zusammenhängend (monoton in den haltenden Kanten) oder genau zwei Teile (nicht monoton).")
    p = st.slider("Haltewahrscheinlichkeit p je Kante", *bounds("p_slider"), key="p_slider", step=0.01, help="Jede Kante hält unabhängig mit dieser Wahrscheinlichkeit; bei p nahe 1 sind Ausfälle selten.")
    st.markdown("**Schätzung**")
    n = st.select_slider("Auswertungen N", options("n_select"), key="n_select", help="Ausgewertete Netze je Schätzung. Das gespiegelte Verfahren zieht N/2 Zufallsvektoren und wertet jeden zweimal aus.")
    seed = st.number_input("Zufallsstrom", *bounds("seed_input"), key="seed_input", step=1, help="Welcher feste Strom von Zufallszahlen benutzt wird (SplitMix64).")
    st.button("🎲 Neuen Strom wählen", width="stretch", on_click=randomize_seed)

sync_query_params({"gitter_select": gitter, "p_slider": float(p), "n_select": int(n), "ziel_select": ziel, "seed_input": int(seed)})

settings = Settings(str(gitter), float(p), int(n), str(ziel), int(seed))
counts = _counts(settings.gitter)
a = _analysis(settings, counts)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Wahrheit", de(a.truth, 4), help=f"Exakte Wahrscheinlichkeit des Ziels „{C.TARGETS[settings.ziel]}“ im {a.netz.name} bei p = {de(settings.p, 2)} (Aufzählung aller Kantenmengen).")
m2.metric("Streuung (unabh.)", de(a.sd_plain, 5), help=f"Standardabweichung von {C.REPS} Schätzungen aus je {de(settings.n, 0)} unabhängigen Auswertungen.")
m3.metric("Streuung (Paare)", de(a.sd_anti, 5), help=f"Standardabweichung von {C.REPS} Schätzungen aus je {de(settings.n // 2, 0)} gespiegelten Paaren (ebenfalls {de(settings.n, 0)} Auswertungen).")
m4.metric("Varianzverhältnis", de(a.var_ratio, 2), help="Varianz gespiegelt geteilt durch Varianz unabhängig; unter 1 ist ein Gewinn. Das Rauschen bei 200 Wiederholungen beträgt rund 14 bis 15 %.")
rho_text = f"ρ = {de(a.rho_exact, 3)} (exakt aufgezählt)" if a.rho_exact is not None else f"ρ = {de(a.rho, 3)} (aus den gezogenen Paaren)"
st.caption(f"Vorhergesagtes Varianzverhältnis 1 + ρ = {de(a.predicted_ratio, 2)}; {rho_text}.")
st.plotly_chart(build_spread(a), width="stretch", key="main_chart")
st.caption("Kästen: die Verteilung der Schätzungen über die Wiederholungen (Linie: Median, Raute und Balken: Mittel und Streuung). Gestrichelt: die Wahrheit.")

if st.button("📄 PDF der Schätzung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="gekoppelte-ziehungen.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Lohnt sich die Spiegelung, und wovon hängt es ab?")
kind, text = ev.verdict_spread(a, settings.n)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Ändert die Spiegelung die Rate oder die Konstante?")
st.caption(f"Streuung der Schätzung bei N = {', '.join(str(x) for x in C.SWEEP_N)} Auswertungen, je {C.SWEEP_REPS} Wiederholungen mit festen Strömen, gewähltes Netz, Ziel und p. Kostet einige Sekunden.")
if st.button("▶ Messung starten", key="run_sweep"):
    st.session_state["sweep_done"] = True
if st.session_state.get("sweep_done"):
    sweep = _n_sweep(settings)
    st.plotly_chart(build_n_sweep(sweep), width="stretch", key="sweep_chart")
    kind, text = ev.verdict_n_sweep(sweep)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 2: Wie hängt der Gewinn an p?")
st.caption(
    "Varianzfaktor 1 + ρ gegen die Haltewahrscheinlichkeit. Bis 12 Kanten ist ρ **exakt** aufgezählt (3^m Kategorien je Kante: hält im ersten, im zweiten oder in beiden Netzen), darüber aus "
    f"{C.RHO_PAIRS:,} Paaren je p geschätzt (Rauten). Kostet einige Sekunden.".replace(",", ".")
)
if st.button("▶ Messung starten", key="run_rho"):
    st.session_state["rho_done"] = True
if st.session_state.get("rho_done"):
    rho_exp = _rho_curve(settings)
    st.plotly_chart(build_rho_curve(rho_exp, settings.p), width="stretch", key="rho_chart")
    kind, text = ev.verdict_rho_curve(rho_exp, settings.ziel)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 3: Welche Funktion wird gespiegelt?")
st.caption("Dieselben Zufallszahlen, vier Funktionen: zwei Netzziele und zwei Skalare, an denen sich die Korrelation von Hand nachrechnen lässt. Kostet einige Sekunden.")
if st.button("▶ Messung starten", key="run_func"):
    st.session_state["func_done"] = True
if st.session_state.get("func_done"):
    func = _functions(settings)
    st.plotly_chart(build_functions(func), width="stretch", key="func_chart")
    kind, text = ev.verdict_functions(func)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 4: Gemeinsame Zufallszahlen beim Vergleich zweier Varianten")
ex = extra_edge(settings.gitter)
st.caption(
    f"Differenz der Zuverlässigkeit bei p und p + δ (gleiche Zufallszahlen gegen unabhängige), dazu der Entwurfsvergleich „Kante {ex[0]}–{ex[1]} dazu“; "
    f"je {C.CRN_DRAWS:,} Ziehungen. Bei monotoner Kopplung ist die Differenz 0 oder 1, ihre Varianz also genau ΔR(1−ΔR). Kostet einige Sekunden.".replace(",", ".")
)
if st.button("▶ Messung starten", key="run_crn"):
    st.session_state["crn_done"] = True
if st.session_state.get("crn_done"):
    crn = _crn(settings, counts)
    st.plotly_chart(build_crn(crn), width="stretch", key="crn_chart")
    kind, text = ev.verdict_crn(crn)
    getattr(st, kind)(text)
    d = crn["design"]
    st.markdown(
        f"**Entwurfsvergleich** (Zusatzkante {d['edge'][0]}–{d['edge'][1]}, p = {de(settings.p, 2)}): R ohne die Kante {de(d['R_a'], 4)}, mit ihr {de(d['R_b'], 4)}, ΔR = {de(d['dR'], 4)}; "
        f"Varianz der Differenz gekoppelt {de(d['var_crn_meas'], 4)} gemessen ({de(d['var_crn'], 4)} nach der Formel), unabhängig {de(d['var_ind'], 4)}; Korrelation der beiden Ziele {de(d['corr'], 3)}."
    )
    st.plotly_chart(build_netz(named_netz(settings.gitter), tuple(d["edge"])), width="stretch", key="design_net")

st.markdown("---")
st.subheader("🔬 Experiment 5: Wenn die Zuordnung der Zufallszahlen nicht stimmt")
st.caption(
    "Gleiche Zufallszahlen setzen voraus, dass die Zahl einer Kante in beiden Varianten dieselbe ist. Liest jede Variante den Strom mit ihrer eigenen Kantenzahl als Schrittweite, "
    f"verschiebt sich bei einer Zusatzkante alles: die „gemeinsamen“ Zahlen sind unabhängig. {C.CRN_DRAWS:,} Ziehungen. Kostet einige Sekunden.".replace(",", ".")
)
if st.button("▶ Messung starten", key="run_sync"):
    st.session_state["sync_done"] = True
if st.session_state.get("sync_done"):
    sync = _sync(settings)
    st.plotly_chart(build_sync(sync), width="stretch", key="sync_chart")
    kind, text = ev.verdict_sync(sync)
    getattr(st, kind)(text)

st.markdown("---")

# --- Spielwiese -----------------------------------------------------------------------------------------------------------------------------

with st.expander("🧪 Spielwiese: ein eigenes Netz und eine eigene Zusatzkante"):
    st.caption(
        f"Eine Kante je Zeile, zwei ganze Zahlen (»0-1«). Höchstens {C.CUSTOM_EDGE_LIMIT} Kanten und {C.CUSTOM_NODE_LIMIT} Knoten; Ziel, p und Strom stammen aus der Seitenleiste. "
        "Die Zusatzkante ist optional (Knotennummern wie im Bild, bei eigenen Netzen von 0 an in der Reihenfolge der Knotennamen)."
    )
    text_in = st.text_area("Kanten", "0-1\n1-2\n2-3\n3-0\n0-2", height=140, key="eigene_kanten")
    extra_in = st.text_input("Zusatzkante (optional)", "1-3", key="eigene_zusatzkante")
    try:
        cnetz, cextra, cres = _custom(text_in, extra_in, settings)
    except ValueError as err:
        st.error(str(err))
    else:
        s1, s2, s3 = st.columns(3)
        s1.metric("Wahrheit", de(cres["truth"], 4))
        s2.metric("Korrelation ρ", "nicht definiert" if cres["rho"] != cres["rho"] else de(cres["rho"], 3), help="Exakt aufgezählt bis 12 Kanten, sonst aus Paaren geschätzt." if not cres["exakt"] else "Exakt aufgezählt.")
        s3.metric("Varianzfaktor 1 + ρ", "–" if cres["rho"] != cres["rho"] else de(cres["factor"], 3))
        st.plotly_chart(build_netz(cnetz, cextra), width="stretch", key="custom_net")
        if cres["rho"] != cres["rho"]:
            st.info("Das Ziel tritt bei diesen Einstellungen nie oder immer ein: es gibt keine Streuung, die sich senken ließe.")
        if "design" in cres:
            d = cres["design"]
            st.success(
                f"Entwurfsvergleich mit der Zusatzkante {d['edge'][0]}–{d['edge'][1]}: R {de(d['R_a'], 4)} → {de(d['R_b'], 4)}, ΔR = {de(d['dR'], 4)}; "
                f"gekoppelt ist die Differenz {de(d['factor'], 1)}-mal ruhiger als bei unabhängigen Ziehungen (Varianz {de(d['var_crn'], 4)} gegen {de(d['var_ind'], 4)}, gemessen {de(d['var_crn_meas'], 4)})."
            )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Funktion ist monoton in den Zufallszahlen** | Nur dann ist ρ ≤ 0 garantiert. Bei symmetrischen Funktionen verdoppelt die Spiegelung die Varianz (Experiment 3); beim Netzziel „genau zwei Teile“ kann sie leicht schaden. | – |
| **Das Ereignis ist nicht selten** | Bei seltenen Ausfällen verschwindet der Gewinn der Spiegelung (Experiment 2). | Stück 7 (Importance Sampling) |
| **Ein Ja-Nein-Ziel lässt sich glätten** | Der Gewinn für ein Ja-Nein-Ziel ist klein; für eine glatte Größe ist er viel größer (Experiment 3, u²). | Stück 6 (bedingtes Monte Carlo) |
| **Gleiche Zufallszahlen bedeuten gleiche Zuordnung** | Bei verschobenem Strom sind die „gemeinsamen“ Zahlen unabhängig, der Mittelwert bleibt unverzerrt, die Streuung wird so groß wie ohne Kopplung (im Standardfall zehnmal so groß, Experiment 5). | – |
| **Die Varianten sind monoton gekoppelt** | Nur dann gilt Var(X_B − X_A) = ΔR(1−ΔR) und der Gewinn wächst mit 1/ΔR. Für nicht monotone Vergleiche gibt es diese Formel nicht. | Stück 11 (Gradienten schätzen) |
| **Bessere Verfahren der Aufteilung** | Spiegelung ist die einfachste Schichtung mit zwei Teilen. | Stück 5 (Schichtung, Latin Hypercube) |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.** Alle Netze, Ausfälle und Funktionen sind **erfunden**; die Wahrheit ist die exakte Aufzählung oder eine geschlossene Formel. Kanten fallen unabhängig und mit gleicher Wahrscheinlichkeit aus. Die Zufallszahlen kommen aus festen Strömen (SplitMix64), nie aus einem einzelnen Seed.

**Grenzen.** Aufgezählt wird live bis 17 Kanten; die exakte Korrelation eines Paares (3^m Kategorien) nur bis 12 Kanten, darüber schätzt die App sie aus 100 000 Paaren (Rauten im Experiment 2). Die Experimente 1 bis 5 laufen auf dem gewählten Netz; Experiment 4 und 5 nutzen das Ziel „zusammenhängend“, da die Formel ΔR(1−ΔR) nur für monotone Kopplungen gilt. Zeiten werden nicht gemessen, nur die Zahl der Auswertungen.

**Bekannte Probleme und Unsicherheiten.** Das Varianzverhältnis aus 200 Wiederholungen rauscht um rund 14 bis 15 %; deshalb zeigt die App daneben das aus der Korrelation vorhergesagte Verhältnis 1 + ρ. Die Verhältnisse in Experiment 1 rauschen ebenso. Die Literaturangaben im Mathematik-Expander stammen aus dem Gedächtnis und sind nicht eingesehen. Die Oberfläche ist mit AppTest geprüft und im Browser angesehen, nicht mit echtem Mausklick auf jedem Gerät.
        """
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Ein Netz mit $m$ Kanten; Kante $e$ hält bei $u_e < p$ mit $u_e$ gleichverteilt. Gesucht ist $\mu = \mathbb E\,X(u)$ für ein Ziel $X \in \{0,1\}$, zum Beispiel „das Netz hängt zusammen“.

**Gespiegeltes Paar.** $X' = X(1-u)$ hat dieselbe Verteilung wie $X$. Der Paar-Schätzer $\tfrac12 (X + X')$ ist erwartungstreu mit
$\operatorname{Var} = \tfrac12 \operatorname{Var} X\,(1 + \rho)$, $\rho = \operatorname{Corr}(X, X')$; zwei unabhängige Auswertungen haben $\tfrac12 \operatorname{Var} X$. Das Verhältnis ist $1+\rho$.

**Garantie.** Ist $X$ in jeder Koordinate von $u$ monoton (hier: wachsend in den haltenden Kanten, also fallend in $u$), gilt $\operatorname{Cov}(X, X') \le 0$, also $1+\rho \le 1$. Für nicht monotone $X$ gibt es keine Garantie: ist $X(u) = X(1-u)$, gilt $\rho = 1$ und das Verhältnis ist $2$.

**Exakte Korrelation.** Je Kante gibt es für $p > \tfrac12$ drei Fälle: hält in beiden Netzen (Wahrscheinlichkeit $2p-1$), nur im ersten ($1-p$), nur im zweiten ($1-p$); für $p \le \tfrac12$: in keinem ($1-2p$), nur im ersten ($p$), nur im zweiten ($p$). Aufzählung der $3^m$ Kombinationen liefert $\mathbb E X$, $\mathbb E X'$, $\mathbb E XX'$. Für einen Pfad mit $m$ Kanten ($p > \tfrac12$): $\rho = \dfrac{(2p-1)^m - p^{2m}}{p^m (1-p^m)}$.

**Skalare Beispiele.** $f(u) = u^2$: $\operatorname{Cov}(u^2, (1-u)^2) = \tfrac1{30} - \tfrac19$, $\operatorname{Var}(u^2) = \tfrac15 - \tfrac19$, also $\rho = -\tfrac78$ und Verhältnis $\tfrac18$. $f(u) = (2u-1)^2$: $f(1-u) = f(u)$, also $\rho = 1$.

**Gemeinsame Zufallszahlen.** Zwei Varianten $A, B$ mit denselben $u$; $B$ hält mindestens dort, wo $A$ hält ($p_B \ge p_A$ oder eine Kante mehr). Dann ist $D = X_B - X_A \in \{0,1\}$, $\mathbb E D = \Delta R = R_B - R_A$ und $\operatorname{Var} D = \Delta R\,(1 - \Delta R)$. Bei unabhängigen Zufallszahlen ist $\operatorname{Var} D = R_A(1-R_A) + R_B(1-R_B)$. Der Gewinnfaktor ist das Verhältnis beider.

**Synchronisation.** Die Zufallszahl der Kante $e$ in Ziehung $i$ ist die Nummer $i\,s + e$ des Stroms mit fester Schrittweite $s$. Verwenden $A$ und $B$ verschiedene Schrittweiten (je ihre eigene Kantenzahl), sind ihre Zahlen unabhängig; $\mathbb E D$ bleibt $\Delta R$, aber $\operatorname{Var} D$ ist die unabhängige.

**Literatur (aus dem Gedächtnis, nicht eingesehen).** Hammersley und Morton 1956 (antithetische Variablen); Law und Kelton, *Simulation Modeling and Analysis*, und Glasserman, *Monte Carlo Methods in Financial Engineering* 2003 (gemeinsame Zufallszahlen, Synchronisation).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
