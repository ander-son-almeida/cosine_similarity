def ajustar_modelo_hazard_discreto(df_painel: pd.DataFrame, covariaveis: List[str],
                                      mob_maximo: Optional[int] = None,
                                      fracao_amostra: Optional[float] = None, seed: int = 42) -> dict:
    df = df_painel.copy()
    if mob_maximo is not None:
        df = df.loc[df["months_on_book"] <= mob_maximo]
    df["mob_ao_quadrado"] = df["months_on_book"] ** 2
    colunas_x = ["months_on_book", "mob_ao_quadrado"] + covariaveis

    n_antes = len(df)
    df = df.dropna(subset=colunas_x + ["default_flag"])
    n_descartado = n_antes - len(df)

    colunas_sem_variancia = [c for c in colunas_x if df[c].nunique(dropna=True) <= 1]
    colunas_x = [c for c in colunas_x if c not in colunas_sem_variancia]

    # NOVO: amostra pra reduzir custo de ajuste em base muito grande --
    # roda DEPOIS do dropna/variancia-zero, pra amostrar so o que de fato
    # vai entrar no modelo.
    n_antes_amostra = len(df)
    if fracao_amostra is not None:
        df = df.sample(frac=fracao_amostra, random_state=seed)

    # blindagem: months_on_book e mob_ao_quadrado NUNCA entram na
    # padronizacao, mesmo que por engano acabem aparecendo em `covariaveis`
    colunas_a_padronizar = [c for c in covariaveis if c in colunas_x and c not in ("months_on_book", "mob_ao_quadrado")]
    medias = df[colunas_a_padronizar].mean()
    desvios = df[colunas_a_padronizar].std().replace(0, 1.0)
    df[colunas_a_padronizar] = (df[colunas_a_padronizar] - medias) / desvios

    X = sm.add_constant(df[colunas_x].astype(float))
    y = df["default_flag"].astype(float)
    modelo = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit()

    return {"modelo": modelo, "n_descartado_nan": n_descartado,
            "pct_descartado_nan": (n_descartado / n_antes) if n_antes else 0.0,
            "colunas_removidas_variancia_zero": colunas_sem_variancia,
            "n_usado_no_ajuste": len(df), "n_antes_da_amostra": n_antes_amostra,
            "padronizacao": {"medias": medias, "desvios": desvios, "colunas": colunas_a_padronizar}}


def plot_hazard_de_base(resultado_hazard: dict, df_painel: pd.DataFrame, covariaveis: List[str]) -> plt.Figure:
    modelo_hazard = resultado_hazard["modelo"]
    padronizacao = resultado_hazard["padronizacao"]
    colunas_modelo = modelo_hazard.model.exog_names

    df = df_painel.copy()
    df["mob_ao_quadrado"] = df["months_on_book"] ** 2
    for c in padronizacao["colunas"]:
        if c in ("months_on_book", "mob_ao_quadrado"):
            continue  # blindagem, mesma logica do ajuste
        df[c] = (df[c] - padronizacao["medias"][c]) / padronizacao["desvios"][c]
    df["const"] = 1.0

    df["hazard_previsto"] = modelo_hazard.predict(df[colunas_modelo])
    curva = df.groupby("months_on_book")["hazard_previsto"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(curva["months_on_book"], curva["hazard_previsto"], marker="o", color="#4C72B0", linewidth=2)
    ax.set_xlabel("Months on Book (meses desde a originacao)")
    ax.set_ylabel("Hazard previsto (probabilidade de default neste mes)")
    ax.set_title("Hazard de base ao longo da vida do contrato")
    fig.tight_layout()
    return fig
