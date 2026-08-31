# valor financiado e estatico por contrato (repete em toda fotografia mensal dele) --
# por isso dropDuplicates antes de somar, senao conta o mesmo valor varias vezes

valor_financiado_por_safra = (
    df_painel_confiavel
    .dropDuplicates(["contract_id"])  # uma linha por contrato, nao uma por mes
    .withColumn("safra_derivada", F.add_months(F.col("ref_month"), -(F.col("months_on_book") - 1)))
    .groupBy(F.col("safra_derivada").alias("safra"))
    .agg(F.sum("NOME_COLUNA_VALOR_FINANCIADO").alias("valor_financiado"))
    .toPandas()
)
valor_financiado_por_safra["safra"] = pd.to_datetime(valor_financiado_por_safra["safra"])