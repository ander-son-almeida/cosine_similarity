def plot_hazard_de_base(resultado_hazard: dict, df_painel: pd.DataFrame, covariaveis: List[str]) -> plt.Figure:
    modelo_hazard = resultado_hazard["modelo"]
    padronizacao = resultado_hazard["padronizacao"]
    colunas_modelo = modelo_hazard.model.exog_names

    df = df_painel.copy()
    df["mob_ao_quadrado"] = df["months_on_book"] ** 2
    for c in padronizacao["colunas"]:
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