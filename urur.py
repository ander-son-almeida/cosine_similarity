import pandas as pd
import pyspark.sql.functions as F
import plotly.graph_objects as go

# --- AJUSTAR: nome real da coluna de valor financiado no seu df_painel_confiavel ---
NOME_COLUNA_VALOR_FINANCIADO = "vlr_financiado"

# --- 1. Valor financiado por safra -- e ESTATICO por contrato (nao muda mes a mes),
#     por isso dropDuplicates ANTES de somar, senao conta o mesmo contrato varias vezes ---
valor_financiado_por_safra = (
    df_painel_confiavel
    .dropDuplicates(["contract_id"])
    .withColumn("safra_derivada", F.add_months(F.col("ref_month"), -(F.col("months_on_book") - 1)))
    .groupBy(F.col("safra_derivada").alias("safra"))
    .agg(F.sum(NOME_COLUNA_VALOR_FINANCIADO).alias("valor_financiado"))
    .toPandas()
)
valor_financiado_por_safra["safra"] = pd.to_datetime(valor_financiado_por_safra["safra"])

# --- 2. ECL por safra -- e ESTOQUE (saldo daquele mes), nao pode somar atraves do
#     tempo (contaria o mesmo contrato em varias fotografias). Agrega por safra+mes
#     primeiro, depois pega so a fotografia MAIS RECENTE como retrato "atual" ---
ecl_por_safra_e_mes = (
    df_painel_confiavel
    .withColumn("safra_derivada", F.add_months(F.col("ref_month"), -(F.col("months_on_book") - 1)))
    .groupBy(F.col("safra_derivada").alias("safra"), F.col("ref_month").alias("mes_calendario"))
    .agg(F.sum("ecl_total").alias("ecl_total"))
    .toPandas()
)
ecl_por_safra_e_mes["safra"] = pd.to_datetime(ecl_por_safra_e_mes["safra"])
ecl_por_safra_e_mes["mes_calendario"] = pd.to_datetime(ecl_por_safra_e_mes["mes_calendario"])

ultima_fotografia = ecl_por_safra_e_mes["mes_calendario"].max()
ecl_por_safra = ecl_por_safra_e_mes[
    ecl_por_safra_e_mes["mes_calendario"] == ultima_fotografia
][["safra", "ecl_total"]]

# --- 3. Junta os dois e calcula cobertura ---
resumo_por_safra = ecl_por_safra.merge(valor_financiado_por_safra, on="safra", how="outer").sort_values("safra")
resumo_por_safra["cobertura_pct"] = resumo_por_safra["ecl_total"] / resumo_por_safra["valor_financiado"] * 100

# --- 4. Grafico: ECL total (barra) + cobertura (linha), por safra ---
fig_ecl_safra = go.Figure()
fig_ecl_safra.add_trace(go.Bar(
    y=[s.strftime("%Y-%m") for s in resumo_por_safra["safra"]], x=resumo_por_safra["ecl_total"],
    orientation="h", name="ECL total (fotografia mais recente)", marker_color="#D85A30",
))
fig_ecl_safra.add_trace(go.Scatter(
    y=[s.strftime("%Y-%m") for s in resumo_por_safra["safra"]], x=resumo_por_safra["cobertura_pct"],
    mode="lines+markers", name="Cobertura (ECL / valor financiado, %)", xaxis="x2",
    line=dict(color="#2A6F97"),
))
fig_ecl_safra.update_layout(
    xaxis=dict(title="ECL total (R$)"),
    xaxis2=dict(title="Cobertura (%)", overlaying="x", side="top"),
    yaxis=dict(title="Safra"),
    title=f"ECL por safra (fotografia {ultima_fotografia.strftime('%Y-%m')}) e cobertura sobre o valor financiado",
    template="plotly_white", width=1400, height=550,
)
fig_ecl_safra.show()