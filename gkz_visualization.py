"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import gkz_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_spread(a):
    """Verteilung der Schätzungen: N unabhängige Auswertungen gegen N/2 gespiegelte Paare, mit der Wahrheit als Linie."""
    fig = go.Figure()
    fig.add_trace(go.Box(y=list(a.plain), name="unabhängig", marker_color=C.COLOR_PLAIN, boxmean="sd", boxpoints="outliers"))
    fig.add_trace(go.Box(y=list(a.anti), name="gespiegelte Paare", marker_color=C.COLOR_ANTI, boxmean="sd", boxpoints="outliers"))
    fig.add_hline(y=a.truth, line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="Wahrheit", annotation_position="top right")
    fig.update_yaxes(title_text="Schätzung")
    fig = _base(fig, 330)
    fig.update_layout(showlegend=False)
    return fig


def build_n_sweep(exp):
    rows = exp["rows"]
    fig = go.Figure()
    ns = [r["N"] for r in rows]
    fig.add_trace(go.Scatter(x=ns, y=[r["sd_plain"] for r in rows], mode="markers+lines", name="unabhängig", line=dict(color=C.COLOR_PLAIN)))
    fig.add_trace(go.Scatter(x=ns, y=[r["sd_anti"] for r in rows], mode="markers+lines", name="gespiegelte Paare", line=dict(color=C.COLOR_ANTI)))
    fig.update_xaxes(title_text="Auswertungen N", type="log")
    fig.update_yaxes(title_text="Streuung der Schätzung", type="log")
    return _base(fig, 320)


def build_rho_curve(exp, p_now):
    rows = exp["rows"]
    fig = go.Figure()
    ps = [r["p"] for r in rows]
    fig.add_trace(go.Scatter(x=ps, y=[r["factor"] for r in rows], mode="markers+lines", name="Varianzfaktor 1 + ρ", line=dict(color=C.COLOR_ANTI),
                             marker=dict(symbol=["circle" if r["exakt"] else "diamond" for r in rows], size=9)))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="unabhängig", annotation_position="top left")
    fig.add_vline(x=p_now, line=dict(color=C.COLOR_OTHER, dash="dot"), annotation_text="Regler p", annotation_position="bottom right")
    fig.update_xaxes(title_text="Haltewahrscheinlichkeit p")
    fig.update_yaxes(title_text="Varianz gespiegelt / unabhängig")
    return _base(fig, 320)


def build_functions(exp):
    rows = exp["rows"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[r["name"] for r in rows], x=[r["factor"] for r in rows], orientation="h",
        marker_color=[C.COLOR_ANTI if r["factor"] <= 1.0 + 1e-9 else C.COLOR_BAD for r in rows], text=[f"{r['factor']:.3f}".replace(".", ",") for r in rows], textposition="outside",
    ))
    fig.add_vline(x=1.0, line=dict(color=C.COLOR_TRUTH, dash="dash"))
    fig.update_xaxes(title_text="Varianzfaktor (unter 1 ist ein Gewinn)", range=[0, max(2.3, max(r["factor"] for r in rows) * 1.15)])
    fig.update_yaxes(autorange="reversed")
    fig = _base(fig, 300)
    fig.update_layout(showlegend=False, margin=dict(l=10, r=40, t=20, b=10))
    return fig


def build_crn(exp):
    rows = exp["rows"]
    fig = go.Figure()
    ds = [r["delta"] for r in rows]
    fig.add_trace(go.Scatter(x=ds, y=[r["factor"] for r in rows], mode="lines", name="Formel ΔR(1−ΔR)", line=dict(color=C.COLOR_PLAIN, dash="dash")))
    fig.add_trace(go.Scatter(x=ds, y=[r["factor_meas"] for r in rows], mode="markers", name="gemessen", marker=dict(size=10, color=C.COLOR_ANTI)))
    fig.update_xaxes(title_text="Abstand δ der beiden Haltewahrscheinlichkeiten", type="log", tickvals=ds, ticktext=[str(d).replace(".", ",") for d in ds])
    fig.update_yaxes(title_text="Gewinnfaktor (Var unabhängig / Var gekoppelt)", type="log")
    return _base(fig, 320)


def build_sync(exp):
    fig = go.Figure()
    labels = ["feste Zuordnung", "Zuordnung nach Lage im Strom"]
    fig.add_trace(go.Bar(x=labels, y=[exp["fest"]["var"], exp["lage"]["var"]], marker_color=[C.COLOR_ANTI, C.COLOR_BAD],
                         text=[f"{exp['fest']['var']:.4f}".replace(".", ","), f"{exp['lage']['var']:.4f}".replace(".", ",")], textposition="outside"))
    fig.add_hline(y=exp["var_ind"], line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="unabhängig", annotation_position="top left")
    fig.update_yaxes(title_text="Varianz der Differenz X_B − X_A", range=[0, exp["var_ind"] * 1.25])
    fig = _base(fig, 300)
    fig.update_layout(showlegend=False)
    return fig


def build_netz(netz, extra=None):
    """Das Netz; eine Zusatzkante (Entwurfsvergleich) gestrichelt."""
    fig = go.Figure()
    xs, ys = [c[0] for c in netz.coords], [c[1] for c in netz.coords]
    ex, ey = [], []
    for u, v in netz.edges:
        ex += [xs[u], xs[v], None]
        ey += [ys[u], ys[v], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#666", width=3), name="Kante", hoverinfo="skip"))
    if extra is not None:
        a, b = extra
        fig.add_trace(go.Scatter(x=[xs[a], xs[b]], y=[ys[a], ys[b]], mode="lines", line=dict(color=C.COLOR_ANTI, width=4, dash="dot"), name="Zusatzkante", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=22, color="white", line=dict(color="#222", width=2)), showlegend=False, hoverinfo="skip"))
    for i, (x, y) in enumerate(zip(xs, ys)):
        fig.add_annotation(x=x, y=y, text=str(i), showarrow=False, font=dict(size=10, color="#222"))
    pad = 0.5
    fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad])
    fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad])
    fig = _base(fig, 300)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=0.0, x=0), margin=dict(l=10, r=10, t=20, b=40))
    return fig
