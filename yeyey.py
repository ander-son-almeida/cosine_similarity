df_filtrado_colunas = (spark.table("pr_explo.d8018s078_modelagem_credito_massificado.16_base_LGD_cheque_conta_enri...")  # sua tabela
    .select(["LGD", "ccpf_cnpj_gerc", "safra", "amostra"] + features_lista)
    # 1. safra em data
    .withColumn("safra", F.to_date("safra", "yyyyMM"))
    # 2. fora o período recente
    .filter(F.lower(F.col("amostra")) != "rec")
    # 3. só safras com 24 meses de cobrança até mai/26
    .filter(F.col("safra") <= F.lit("2024-05-01"))
    # 4. DES e OOT dentro das safras maduras
    .withColumn("amostra", F.when(F.col("safra") <= F.lit("2024-02-01"), F.lit("des"))
                            .otherwise(F.lit("oot")))
    # 5. LGD apurada e entre 0 e 1
    .filter(F.col("LGD").isNotNull())
    .withColumn("LGD", F.least(F.greatest(F.col("LGD"), F.lit(0.0)), F.lit(1.0)))
    # 6. uma linha por contrato e safra
    .dropDuplicates(["safra", "ccpf_cnpj_gerc"])
)

# conferência: deve mostrar des de out/23 a fev/24 e oot de mar/24 a mai/24
df_filtrado_colunas.groupBy("amostra").agg(
    F.count("*").alias("contratos"), F.min("safra").alias("primeira"),
    F.max("safra").alias("ultima"), F.avg("LGD").alias("lgd_media")).display()
