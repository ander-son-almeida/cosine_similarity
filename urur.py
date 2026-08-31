import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyspark.sql.functions as F

# --- 1. ECL por celula (safra x MOB x mes), soma direta, sem nenhum modelo ---
df_ecl_por_celula = (
    df_painel_confiavel
    .withColumn("safra_derivada", F.add_months(F.col("ref_month"), -(F.col("months_on_book") - 1)))
    .groupBy(F.col("safra_derivada").alias("safra"),
             F.col("months_on_book").alias("mob"),
             F.col("ref_month").alias("mes_calendario"))
    .agg(F.sum("ecl_total").alias("ecl_total_celula"))
)
ecl_pd = df_ecl_por_celula.toPandas()
ecl_pd["safra"] = pd.to_datetime(ecl_pd["safra"])
ecl_pd["mes_calendario"] = pd.to_datetime(ecl_pd["mes_calendario"])
painel = painel.merge(ecl_pd, on=["safra", "mob", "mes_calendario"], how="left")
painel["ecl_por_contrato"] = painel["ecl_total_celula"] / painel["populacao_risco"]

# --- 2. Painel ECL x Cobertura por safra -- mesmo estilo do seu grafico de volume ---
ecl_por_safra = painel.groupby("safra")["ecl_total_celula"].sum().reset_index()
# junte com a tabela de volume/valor financiado que voce ja tem (a mesma que alimenta o
# grafico "N. de contratos" / "Valor financiado") -- ajuste o nome real da coluna abaixo
ecl_por_safra = ecl_por_safra.merge(valor_financiado_por_safra, on="safra", how="left")
ecl_por_safra["cobertura_pct"] = ecl_por_safra["ecl_total_celula"] / ecl_por_safra["NOME_COLUNA_VALOR_FINANCIADO"] * 100

fig_ecl_safra = go.Figure()
fig_ecl_safra.add_trace(go.Bar(
    y=[s.strftime("%Y-%m") for s in ecl_por_safra["safra"]], x=ecl_por_safra["ecl_total_celula"],
    orientation="h", name="ECL total", marker_color="#D85A30",
))
fig_ecl_safra.add_trace(go.Scatter(
    y=[s.strftime("%Y-%m") for s in ecl_por_safra["safra"]], x=ecl_por_safra["cobertura_pct"],
    mode="lines+markers", name="Cobertura (ECL / valor financiado, %)", xaxis="x2",
))
fig_ecl_safra.update_layout(
    xaxis=dict(title="ECL total (R$)"),
    xaxis2=dict(title="Cobertura (%)", overlaying="x", side="top"),
    yaxis=dict(title="Safra"),
    title="ECL por safra -- total e cobertura sobre o valor financiado",
    template="plotly_white", width=1400, height=500,
)
fig_ecl_safra.show()

# --- 3. O mesmo mapa de bolhas que voce ja tem, so trocando a variavel de cor para ECL ---
# reaproveita "bolha_para_intervalo" -- adicione o parametro variavel_cor na funcao existente:
def bolha_para_intervalo(safra_ini, safra_fim, variavel_cor="taxa_entrada"):
    subset = painel[(painel["safra"] >= safra_ini) & (painel["safra"] <= safra_fim)]
    pop = subset["populacao_risco"].values
    raio_min_px, raio_max_px = 4, 18
    pop_min, pop_max = pop.min(), pop.max()
    if pop_max == pop_min:
        tamanhos = np.full(len(pop), (raio_min_px + raio_max_px) / 2)
    else:
        t = (np.sqrt(pop) - np.sqrt(pop_min)) / (np.sqrt(pop_max) - np.sqrt(pop_min))
        tamanhos = raio_min_px + t * (raio_max_px - raio_min_px)
    cores = subset[variavel_cor].values  # ANTES: sempre taxa_entrada -- AGORA: parametrizado
    safras_no_intervalo = sorted(subset["safra"].unique())
    indice_local = {s: i for i, s in enumerate(safras_no_intervalo)}
    y_num = subset["safra"].map(indice_local).values
    return dict(
        x=subset["mob"].values, y=y_num, tamanhos=tamanhos * 2, cores=cores,
        cmin=float(np.min(cores)) if len(cores) else 0.0,
        cmax=float(np.quantile(cores, 0.98)) if len(cores) else 1.0,
        mob_min=float(subset["mob"].min() - 0.5) if len(subset) else 0.5,
        mob_max=float(subset["mob"].max() + 0.5) if len(subset) else mob_maximo + 0.5,
        n_safras=len(safras_no_intervalo),
        rotulos_y=[s.strftime("%Y-%m") for s in safras_no_intervalo],
        customdata=[str(s.date()) for s in subset["safra"]],
    )

# chame com variavel_cor="ecl_por_contrato" no lugar de "taxa_entrada" pra ver o mesmo
# mapa (mesmas posicoes de bolha) colorido por ECL em vez de taxa de entrada