import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display

# --- pressupoe que ja existem: painel, backtest_df, mob_maximo, H_EMPIRICO ---

LARGURA = 1400
safras_ordenadas = sorted(painel["safra"].unique())
meses_unicos = sorted(painel["mes_calendario"].unique())

def bolha_para_intervalo(safra_ini, safra_fim):
    subset = painel[(painel["safra"] >= safra_ini) & (painel["safra"] <= safra_fim)]
    valor_para_tamanho = subset["taxa_entrada"].values  # tamanho agora acompanha a taxa, nao a populacao
    raio_min_px, raio_max_px = 4, 18
    valor_min, valor_max = valor_para_tamanho.min(), valor_para_tamanho.max()
    if valor_max == valor_min:
        tamanhos = np.full(len(valor_para_tamanho), (raio_min_px + raio_max_px) / 2)
    else:
        t = (np.sqrt(valor_para_tamanho) - np.sqrt(valor_min)) / (np.sqrt(valor_max) - np.sqrt(valor_min))
        tamanhos = raio_min_px + t * (raio_max_px - raio_min_px)
    taxas = subset["taxa_entrada"].values
    safras_no_intervalo = sorted(subset["safra"].unique())
    indice_local = {s: i for i, s in enumerate(safras_no_intervalo)}
    y_num = subset["safra"].map(indice_local).values
    return dict(
        x=subset["mob"].values, y=y_num, tamanhos=tamanhos * 2, taxas=taxas,
        cmin=float(np.min(taxas)) if len(taxas) else 0.0,
        cmax=float(np.quantile(taxas, 0.98)) if len(taxas) else 1.0,
        mob_min=float(subset["mob"].min() - 0.5) if len(subset) else 0.5,
        mob_max=float(subset["mob"].max() + 0.5) if len(subset) else mob_maximo + 0.5,
        n_safras=len(safras_no_intervalo),
        rotulos_y=[s.strftime("%Y-%m") for s in safras_no_intervalo],
        customdata=[str(s.date()) for s in subset["safra"]],
    )

# --- figura base: 3 traces, SEM frames (FigureWidget nao suporta frames) ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Bolhas: safra x MOB -- tamanho e cor pela taxa, no intervalo selecionado",
                     f"Teste de Page: C+ (piora) e C- (melhora), limite h={H_EMPIRICO:.1f}"),
    column_widths=[0.5, 0.5],
)

j0 = bolha_para_intervalo(safras_ordenadas[0], safras_ordenadas[min(5, len(safras_ordenadas) - 1)])
fig.add_trace(go.Scatter(
    x=j0["x"], y=j0["y"], mode="markers",
    marker=dict(size=j0["tamanhos"], color=j0["taxas"], colorscale="Reds",
                cmin=j0["cmin"], cmax=j0["cmax"], showscale=True,
                colorbar=dict(title="taxa", x=0.46)),
    customdata=j0["customdata"],
    hovertemplate="Safra %{customdata}<br>MOB %{x}<br>Taxa %{marker.color:.2%}<extra></extra>",
), row=1, col=1)
fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name="C+ (piora sustentada)",
                          line=dict(color="#B23A48", width=2)), row=1, col=2)
fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name="C- (melhora sustentada)",
                          line=dict(color="#2A6F97", width=2)), row=1, col=2)
fig.add_hline(y=H_EMPIRICO, line_dash="dash", line_color="black",
              annotation_text=f"h = {H_EMPIRICO:.1f}", row=1, col=2)
fig.update_layout(
    width=LARGURA, height=650,
    xaxis=dict(title="MOB (idade do contrato)", range=[j0["mob_min"], j0["mob_max"]]),
    yaxis=dict(title="Safra", range=[-0.5, j0["n_safras"] - 0.5],
               tickvals=list(range(j0["n_safras"])), ticktext=j0["rotulos_y"]),
    xaxis2=dict(title="Fotografia (mes de referencia)", range=[meses_unicos[0], meses_unicos[-1]]),
    yaxis2_title="Estatistica de Page",
    template="plotly_white", margin=dict(t=70, b=50),
)

fig_widget = go.FigureWidget(fig)

# --- controle 1: intervalo de safras (duas alcas de verdade) ---
slider_safra = widgets.SelectionRangeSlider(
    options=[(s.strftime("%Y-%m"), s) for s in safras_ordenadas],
    index=(0, min(5, len(safras_ordenadas) - 1)),
    description="Safras:", layout=widgets.Layout(width="700px"),
    style={"description_width": "initial"},
)

def ao_mudar_intervalo(change):
    safra_ini, safra_fim = change["new"]
    j = bolha_para_intervalo(safra_ini, safra_fim)
    with fig_widget.batch_update():
        fig_widget.data[0].x = j["x"]
        fig_widget.data[0].y = j["y"]
        fig_widget.data[0].marker.size = j["tamanhos"]
        fig_widget.data[0].marker.color = j["taxas"]
        fig_widget.data[0].marker.cmin = j["cmin"]
        fig_widget.data[0].marker.cmax = j["cmax"]
        fig_widget.data[0].customdata = j["customdata"]
        fig_widget.layout.xaxis.range = [j["mob_min"], j["mob_max"]]
        fig_widget.layout.yaxis.range = [-0.5, j["n_safras"] - 0.5]
        fig_widget.layout.yaxis.tickvals = list(range(j["n_safras"]))
        fig_widget.layout.yaxis.ticktext = j["rotulos_y"]

slider_safra.observe(ao_mudar_intervalo, names="value")

# --- controle 2: fotografia (valor unico -- substitui o slider nativo de frames do Plotly,
#     que nao pode conviver com FigureWidget) ---
slider_fotografia = widgets.SelectionSlider(
    options=[(m.strftime("%Y-%m"), m) for m in meses_unicos],
    value=meses_unicos[-1],
    description="Fotografia:", layout=widgets.Layout(width="700px"),
    style={"description_width": "initial"},
)

def ao_mudar_fotografia(change):
    mes_foto = change["new"]
    sub = backtest_df[backtest_df["mes_calendario"] <= mes_foto]
    with fig_widget.batch_update():
        fig_widget.data[1].x = sub["mes_calendario"]
        fig_widget.data[1].y = sub["c_mais"]
        fig_widget.data[2].x = sub["mes_calendario"]
        fig_widget.data[2].y = sub["c_menos"]

slider_fotografia.observe(ao_mudar_fotografia, names="value")
ao_mudar_fotografia({"new": meses_unicos[-1]})  # popula o Page chart de saida

display(widgets.VBox([slider_safra, slider_fotografia]), fig_widget)