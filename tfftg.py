def elasticidade_ingenua_taxa_x_default(df: pd.DataFrame, coluna_taxa: str, controles: List[str],
                                          coluna_outcome: str = "default_flag") -> dict:
    """
    Regressao logistica simples de default em funcao da taxa, controlando
    por outras variaveis. Chamada de "ingenua" DE PROPOSITO -- mistura
    efeito causal com selecao de risco, so deve ser reportada ao lado do
    RDD/estudo de evento.

    NOVO: padroniza taxa + controles antes de ajustar (mesma correcao do
    Hazard) -- sem isso, coeficientes de variaveis em escalas bem
    diferentes (taxa em pontos, renda em reais) ficam dificeis de
    comparar visualmente, mesmo quando os dois sao estatisticamente
    validos (foi exatamente o caso aqui: renda tinha IC bem apertado,
    -0.000017 a -0.000016, so parecia "zero" pela escala).
    """
    colunas_x = [coluna_taxa] + list(controles)
    n_antes = len(df)
    dados = df.dropna(subset=colunas_x + [coluna_outcome])
    n_descartado = n_antes - len(dados)

    medias = dados[colunas_x].mean()
    desvios = dados[colunas_x].std().replace(0, 1.0)
    dados_padronizados = dados.copy()
    dados_padronizados[colunas_x] = (dados[colunas_x] - medias) / desvios

    X = sm.add_constant(dados_padronizados[colunas_x].astype(float))
    y = dados_padronizados[coluna_outcome].astype(float)
    modelo = sm.Logit(y, X).fit(disp=0)

    return {"modelo": modelo, "n_descartado_nan": n_descartado,
            "pct_descartado_nan": (n_descartado / n_antes) if n_antes else 0.0,
            "padronizacao": {"medias": medias, "desvios": desvios, "colunas": colunas_x}}