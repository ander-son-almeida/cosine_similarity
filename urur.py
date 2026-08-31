# 1) ECL por safra, preservando o mes -- sem somar atraves do tempo ainda
ecl_por_safra_e_mes = (
    df_painel_confiavel
    .withColumn("safra_derivada", F.add_months(F.col("ref_month"), -(F.col("months_on_book") - 1)))
    .groupBy(F.col("safra_derivada").alias("safra"), F.col("ref_month").alias("mes_calendario"))
    .agg(F.sum("ecl_total").alias("ecl_total"))
    .toPandas()
)
ecl_por_safra_e_mes["safra"] = pd.to_datetime(ecl_por_safra_e_mes["safra"])
ecl_por_safra_e_mes["mes_calendario"] = pd.to_datetime(ecl_por_safra_e_mes["mes_calendario"])

# 2) "ECL por safra" como retrato ATUAL -- filtra so a fotografia mais recente,
#    nao soma todas juntas (mesmo cuidado que ja tivemos com valor_financiado)
ultima_fotografia = ecl_por_safra_e_mes["mes_calendario"].max()
ecl_por_safra = ecl_por_safra_e_mes[ecl_por_safra_e_mes["mes_calendario"] == ultima_fotografia]