Vejo que seu roteiro focou em literatura em português, quero também em inglês. Eu quero que você mostre extremamente um roteiro com base nessa literatura me ajudar o que testar para encontrar o problema. Se puder amparar cada teste com a literatura ótimo. Alguns não precisam. Mantenha o padrão de olhar crítico, técnico

Perfeito — abaixo vai um **roteiro crítico, técnico e orientado a teste**, ancorado na literatura em inglês e pensado para achar a origem do problema, não só prever default. A ideia central que emerge das revisões é: **inadimplência é um fenômeno multicausal, temporal e de governança**, então você precisa testar camadas diferentes — cliente, produto, processo, modelo e macro — em vez de procurar uma única causa [1][2][3][4].

## Como pensar o problema

A literatura em inglês mostra uma mudança clara: de modelos puramente estatísticos para abordagens que combinam dados tradicionais, variáveis comportamentais, dados alternativos, monitoramento contínuo e early warning systems [2][4][3]. Também aparece repetidamente a limitação de tratar default como um evento isolado, quando na prática ele é precedido por uma trajetória de deterioração com sinais precoces observáveis [2][3].  

Então o seu roteiro deve responder quatro perguntas:

1. O risco piorou porque o cliente piorou?
2. O risco piorou porque o produto ou política empurrou o cliente para stress?
3. O risco piorou porque o macro apertou?
4. O risco piorou porque o modelo, o monitoramento ou a cobrança falharam?

## Estrutura de teste

### 1) Teste de composição da carteira
Objetivo: descobrir se a inadimplência subiu porque a carteira mudou de perfil ou porque o mesmo perfil passou a inadimplir mais.  
Teste: decompor a variação do default em efeito composição versus efeito dentro do segmento. Faça isso por vintage, canal, produto, score, renda, geografia e prazo. A literatura de revisão mostra que mudanças de mix de preditores e amostras ao longo do tempo alteram fortemente o comportamento do default [2][4].

Se a carteira “piorou” porque você originou mais risco, o problema está na política. Se a carteira manteve o mix e o default subiu, o problema está mais em macro, comportamento ou processo.

### 2) Teste de vintage e coorte
Objetivo: localizar a safra que começou o problema.  
Teste: construir curvas de default, atraso e cura por coorte mensal/semanal de originação. Compare cohort curves por produto, canal e política de aprovação. A literatura de credit risk management recomenda análises longitudinais justamente porque o risco muda com o tempo e com o estágio da relação de crédito [2][3].

Se o salto aparece só em algumas safras, há forte suspeita de mudança de política, canal, underwriting ou pricing. Se todas as safras pioram juntas, a chance de macro ou stress sistêmico aumenta.

### 3) Teste de trajetória pré-default
Objetivo: identificar os sinais que antecedem o evento.  
Teste: alinhar os clientes pelo mês do default e olhar 12, 6, 3 e 1 meses antes. Examine uso de limite, pagamento mínimo, revolvência, queda de saldo saudável, atrasos pequenos, recuperação parcial, concentração de consultas, renegociações e redução de liquidez. Revisões de default prediction e EWS destacam a importância de sinais comportamentais e da progressão temporal antes do evento final [2][3][5].

Esse teste costuma responder melhor que o score final quando a pergunta é “o que está mudando no comportamento?”

### 4) Teste de elasticidade ao macro
Objetivo: medir o quanto a carteira é sensível ao ambiente econômico.  
Teste: modelar inadimplência com lags de Selic, inflação, desemprego, renda, câmbio e volume de crédito. Compare elasticidades por segmento, porque a literatura brasileira e internacional encontra impacto relevante de fatores macroeconômicos, com diferenças entre curto e longo prazo [6][7][8].  

Se a inadimplência responde com defasagem a juros e inflação, você tem um problema de exposição sistêmica. Se responde fortemente só em segmentos específicos, há interação entre macro e perfil.

### 5) Teste de afetação por produto
Objetivo: descobrir se a estrutura do produto está induzindo risco.  
Teste: comparar inadimplência por taxa, prazo, parcela, carência, rotativo, limite, overlimit e mecanismo de amortização. A literatura de credit default predictors inclui fatores institucionais e financeiros como determinantes relevantes, além de variáveis de comportamento de crédito [2][4].  

Se produtos com maior flexibilidade mostram mais default, isso pode indicar uso do crédito como ponte de liquidez; se produtos com parcela mais alta pioram, o problema pode ser affordability.

### 6) Teste de affordability e stress
Objetivo: ver se o cliente consegue carregar a dívida.  
Teste: estimar comprometimento de renda, payment-to-income, debt-to-income, debt service ratio e sensibilidade a choque de juros/inflation. A literatura de behavioral and socioeconomic predictors enfatiza renda, estabilidade ocupacional, literacia financeira, impulsividade e condições de vida como fatores que afetam pagamento [2][5].  

Esse teste é essencial para distinguir inadimplência por incapacidade de pagamento versus inadimplência por má gestão do crédito.

### 7) Teste de segmentação comportamental
Objetivo: detectar clusters de risco com mecanismos diferentes.  
Teste: segmentar por comportamento de uso, não só por score: clientes de pagamento mínimo, transatores, rotativos, recorrentes, intermitentes, concentrados, superutilizadores, renegociadores e reincidentes. Revisões mostram que fatores psicológicos, situacionais e comportamentais têm papel consistente em default e delinquency [2][5].  

Se um cluster específico concentra a piora, a solução não é genérica; é intervenção específica por comportamento.

### 8) Teste de drift e calibração do modelo
Objetivo: saber se o modelo deixou de representar a realidade.  
Teste: medir PSI, CSI, estabilidade das variáveis, calibração por faixa de score, performance por período, por canal e por coorte. A revisão de 2024 sobre default prediction models destaca a evolução para modelos híbridos e o problema da adequação do método ao ambiente de dados, enquanto a revisão longitudinal de 2021 mostra a migração para novos dados e técnicas [4][2].  

Se o score separa mal as faixas ou perde calibração, o “aumento de inadimplência” pode ser em parte uma falha de monitoramento e não só do risco real.

### 9) Teste de policy change
Objetivo: verificar se uma mudança de política criou o problema.  
Teste: event study, before-after, difference-in-differences ou synthetic control para avaliar alterações em corte, limite, taxa, carência, canal, acionamento de cobrança, ou regras de renegociação. A literatura de EWS e governance insiste que o valor do modelo está na capacidade de intervir cedo e coordenar resposta operacional [3].

Isso é um dos testes mais importantes porque muita inadimplência nasce de decisão interna, não do mercado.

### 10) Teste de cobrança e cura
Objetivo: medir se o problema é de recuperação, não de originação.  
Teste: analisar roll rates, cure rates, tempo até cura, taxa de contato efetivo, promessa versus pagamento, reincidência após acordo e performance por estratégia de cobrança. A literatura de EWS destaca a necessidade de combinar modelos quantitativos com julgamento e intervenção antecipada para evitar misclassification e prevenir eventos de default [3].  

Se a cura cai e a origem não piora, o problema pode estar na operação de cobrança ou na estratégia de renegociação.

## Mapa dos testes

| Bloco | O que testar | Sinal de problema | Fonte |
|---|---|---|---|
| Carteira | Mudança de mix | A inadimplência sobe por composição | [2][4] |
| Coorte | Vintage específico | Apenas certas safras pioram | [2][3] |
| Trajetória | Sinais pré-default | Stress aparece meses antes | [2][3][5] |
| Macro | Juros, inflação, desemprego | Carteira sensível ao ciclo | [7][6][8] |
| Produto | Estrutura do crédito | Limite, rotativo, parcela explicam o default | [2][4] |
| Affordability | DTI, PTI, DSR | Cliente não carrega a dívida | [2][5] |
| Comportamento | Clusters de uso | Default concentrado em perfis específicos | [2][5] |
| Modelo | Drift, calibração | Score perde validade | [2][4] |
| Política | Mudança interna | Default reage após ação da empresa | [3] |
| Cobrança | Roll/cure/reincidência | Falha de recuperação | [3] |

## Sequência recomendada

Eu faria nesta ordem:

1. Decomposição da piora: mix versus performance.
2. Vintage e coorte.
3. Trajetória pré-default.
4. Segmentação comportamental e por produto.
5. Macro com defasagens.
6. Drift e calibração do modelo.
7. Efeito de políticas e cobrança.
8. Testes quase-causais para confirmar mecanismos.

Essa ordem evita o erro comum de começar pelo modelo e terminar sem saber onde o problema realmente nasceu. A literatura revisada em inglês é consistente em dizer que o melhor resultado vem da combinação de preditores tradicionais, variáveis alternativas, análise longitudinal e governança de early warning [2][4][3].

## Hipóteses críticas

Você pode estruturar a investigação com hipóteses como estas:

- H1: A piora da inadimplência é principalmente explicada por mudança de mix de originação.
- H2: A piora está concentrada em uma ou duas coortes específicas.
- H3: O stress é antecipado por sinais comportamentais mensuráveis antes do atraso formal.
- H4: O modelo perdeu calibração em determinados segmentos.
- H5: O macro explica mais o aumento do default do que a política interna.
- H6: A cobrança está chegando tarde demais para clientes com transição rápida para default.
- H7: O produto induz comportamento de rolagem e pagamento mínimo, elevando o risco futuro [2][5][3].

## Como transformar em estudo

Se você quiser fazer isso de forma robusta, o desenho ideal é um estudo em camadas:

- camada descritiva: onde o default cresce;
- camada explicativa: quais variáveis mudam antes;
- camada causal/quase-causal: o que realmente gera a mudança;
- camada operacional: qual ação reduz o problema.

A literatura em inglês sustenta exatamente essa visão integrada, com revisão sistemática de default prediction models, revisão longitudinal dos preditores e ênfase em early warning systems e governança [4][2][3].  

Se quiser, eu posso transformar isso agora em um **protocolo de pesquisa completo**, com: pergunta, hipótese, desenho amostral, variáveis, features, testes estatísticos, análise temporal, estrutura de relatório e um checklist de diagnóstico para usar em carteira real.

Citações:
[1] Advancing financial resilience: A systematic review of default prediction models and future directions in credit risk management - PubMed https://pubmed.ncbi.nlm.nih.gov/39553697/
[2] A Longitudinal Systematic Review of Credit Risk Assessment and Credit Default Predictors - Büşra Alma Çallı, Erman Coşkun, 2021 https://journals.sagepub.com/doi/10.1177/21582440211061333
[3] Early warning system e governance del rischio di credito. L ... - IRIS https://iris.uniroma1.it/handle/11573/1607103
[4] Advancing financial resilience: A systematic review of ... - Mendeley https://www.mendeley.com/catalogue/ce0dde20-5885-346a-97f8-19a07982db4c/
[5] BEHAVIORAL FACTORS IN LOAN DEFAULT PREDICTION A LITERATURE ... https://ajates-scholarly.com/index.php/ajates/article/view/37
[6] Cartão de crédito: prevendo a taxa de inadimplência ... https://bdtd.ucb.br:8443/jspui/handle/tede/3702
[7] Análise dos fatores macroeconômicos causadores de ... https://dspace.mackenzie.br/items/1016f078-1c75-4da6-be29-9377c12f2b6e
[8] The Fundamental Determinants of Credit Default Risk for ... https://www.imf.org/external/pubs/ft/wp/2010/wp10153.pdf
[9] Scopus - Print Document http://irep.iium.edu.my/114757/13/114757_Loan%20default%20prediction%20using%20machine%20learning%20algorithms_a%20systematic%20literature%20review%202020%20-2023.pdf
[10] Advancing financial resilience: A systematic review of ... https://www.sciencedirect.com/science/article/pii/S240584402415801X
[11] Advancing financial resilience: A systematic review of ... https://www.cell.com/heliyon/fulltext/S2405-8440(24)15801-X
[12] Early warning system e governance del rischio di credito https://journals.francoangeli.it/index.php/cgrds/article/download/12637/1229/57326
[13] Rethinking SME default prediction: a systematic literature review ... https://ideas.repec.org/r/spr/scient/v126y2021i3d10.1007_s11192-020-03856-0.html
[14] Advancing financial resilience: A systematic review of ... https://www.epistemonikos.org/en/documents/a6d028c565f11e79bf96d641f147648add90cdae
