import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import pyspark.sql.functions as F

# --- 1. Agregar ecl_total do df_painel_confiavel (ciclo_credito) no mesmo grão do carteira_diagnostico ---
# ATENCAO: confirme os nomes reais de coluna abaixo contra o df_painel_confiavel atual --
# months_on_book, ref_month, ecl_total sao os nomes registrados na v8; se algo mudou desde
# entao, ajuste aqui antes de rodar.

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

# --- 2. Juntar ao painel que ja existe no carteira_diagnostico ---
painel = painel.merge(ecl_pd, on=["safra", "mob", "mes_calendario"], how="left")

faltando = painel["ecl_total_celula"].isna().sum()
print(f"Celulas sem ECL apos o merge: {faltando}")  # mesma checagem que ja fazemos pra macro -- nao ignorar se > 0

# --- 3. Modelo de referencia para ECL -- Gamma, nao Poisson, porque agora o alvo e dinheiro
#     (continuo, positivo, assimetrico), nao mais contagem de eventos ---
FORMULA_ECL = "ecl_total_celula ~ C(mob_bin) + veio_truncado" + (" + macro_reportado" if USA_MACRO else "")

sub_valido = painel[painel["ecl_total_celula"] > 0].dropna(subset=["ecl_total_celula", "populacao_risco"])
modelo_ecl = smf.glm(
    formula=FORMULA_ECL, data=sub_valido,
    family=sm.families.Gamma(link=sm.families.links.Log()),
    offset=np.log(sub_valido["populacao_risco"]),
).fit()

painel["ecl_esperado"] = modelo_ecl.predict(painel, offset=np.log(painel["populacao_risco"]))
painel["ecl_excedente"] = painel["ecl_total_celula"] - painel["ecl_esperado"]

# --- 4. Grafico: ECL esperado x excedente, por safra ---
provisao_por_safra = painel.groupby("safra").agg(
    ecl_esperado=("ecl_esperado", "sum"),
    ecl_excedente=("ecl_excedente", "sum"),
).reset_index()

fig_provisao = go.Figure()
fig_provisao.add_trace(go.Bar(
    x=[s.strftime("%Y-%m") for s in provisao_por_safra["safra"]],
    y=provisao_por_safra["ecl_esperado"], name="ECL esperado (idade + época)",
    marker_color="#898781",
))
fig_provisao.add_trace(go.Bar(
    x=[s.strftime("%Y-%m") for s in provisao_por_safra["safra"]],
    y=provisao_por_safra["ecl_excedente"], name="ECL excedente (efeito de safra não explicado)",
    marker_color="#D85A30",
))
fig_provisao.update_layout(
    barmode="stack", title="ECL por safra -- esperado x excedente (ecl_total real, sem proxy de LGD)",
    xaxis_title="Safra", yaxis_title="R$", template="plotly_white", width=1400, height=450,
)
fig_provisao.show()