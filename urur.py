# Como interpretar o gráfico de bolhas (safra × MOB)

**O que cada elemento representa**

Cada bolha é uma combinação específica de safra (eixo vertical) e MOB — idade do contrato em meses (eixo horizontal). A posição da bolha nunca muda; o que muda com os controles é o que fica visível e como a cor é calibrada.

**Cor e tamanho**: neste gráfico, os dois codificam a mesma informação — a taxa de entrada em atraso grave (90+) daquela célula, dentro do intervalo de safras selecionado no momento. Bolha maior e mais escura = taxa mais alta. A escala de cor é recalculada a cada vez que o intervalo de safras muda, então a mesma taxa pode parecer mais ou menos "grave" dependendo do que mais está sendo comparado — é contraste relativo à seleção atual, não absoluto contra a carteira inteira.

**Os três padrões espaciais, e o que cada um aponta:**

- **Linha (bolhas na mesma altura, atravessando o eixo horizontal)** — a mesma safra, em idades diferentes. Se essa linha inteira aparece consistentemente maior/mais escura que as linhas vizinhas, é sinal de problema de subscrição daquela leva específica — algo na forma como aqueles contratos foram aprovados, não algo que aconteceu depois. Aponta de volta para a política de crédito vigente no mês de originação daquela safra.

- **Diagonal (bolhas que, somando safra + MOB, caem no mesmo mês calendário)** — safras diferentes, cada uma na sua própria idade, mas todas vivendo o mesmo momento. Se essa diagonal aparece mais escura, é sinal de choque de época — algo bateu em todo mundo que estava com contrato ativo naquele mês, independente de quando cada um nasceu. Aponta para um evento externo (macro, operacional) datado.

- **Coluna (mesma idade, safras diferentes)** — se uma faixa inteira de MOB aparece sistematicamente maior/mais escura em todas as safras, isso normalmente não é anomalia: é o formato natural da curva de risco por idade, que sobe até certo ponto da vida do contrato e depois cai. Só vira sinal de alerta se uma coluna específica estiver destoando do formato esperado para aquela idade — não pelo tamanho absoluto, mas por fugir do padrão que as outras colunas também deveriam seguir.

**Regiões (aglomerados de bolhas vizinhas, não uma célula isolada)**: uma única bolha grande e escura, sozinha, pode ser ruído — população pequena, evento raro isolado. Um aglomerado — várias células vizinhas, tanto em safra quanto em MOB, todas destacadas ao mesmo tempo — é evidência mais forte, porque é bem menos provável que aconteça só por acaso. Ao ler o gráfico, dê mais peso a regiões do que a pontos isolados.

**Os dois controles interativos:**

- **Intervalo de safras**: estreita ou amplia quais safras entram na comparação, recalculando a escala de cor para esse subconjunto. Útil para investigar uma janela específica sem a cor ser "diluída" pelo resto da carteira.
- **Fotografia**: controla até qual mês o painel de resíduo acumulado (teste de Page, ao lado) está revelado — não filtra o gráfico de bolhas, só o painel de monitoramento.

**Limite deste gráfico**: como tamanho e cor aqui codificam a mesma coisa (taxa), ele não mais avisa visualmente quando uma célula tem pouco contrato por trás — isso é uma troca deliberada feita nesta versão. Célula com poucos contratos ainda pode aparecer grande e escura por puro ruído de amostra pequena; vale checar `populacao_risco` da célula antes de tratar qualquer bolha isolada como sinal confiável.