mob_fixo = 6
ref_month_antes = "2025-12-01"
ref_month_depois = "2026-06-01"

filtrado = df_painel_confiavel.filter(
    (F.col("ref_month").isin([ref_month_antes, ref_month_depois]))
    & (F.col("months_on_book") == mob_fixo)
)
com_taxa = filtrado.join(df_originacao.select("contract_id", "taxa_originacao"), on="contract_id", how="left")

com_periodo = com_taxa.withColumn(
    "periodo", F.when(F.col("ref_month") == F.lit(ref_month_antes), "antes").otherwise("depois")
)
base_oaxaca_pd = com_periodo.toPandas()

print(base_oaxaca_pd["periodo"].value_counts())

resultado_oaxaca = decompor_oaxaca_blinder(
    base_oaxaca_pd,
    coluna_periodo="periodo", periodo_base="antes", periodo_comparacao="depois",
    coluna_outcome="default_flag",
    colunas_features=["taxa_originacao", "renda"],
)
fig = plot_decomposicao_oaxaca_blinder(resultado_oaxaca)