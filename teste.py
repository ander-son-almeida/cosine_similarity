# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 22:27:29 2025

@author: Anderson Almeida
"""

import os
import PyPDF2
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Baixar recursos do NLTK (se necessário)
nltk.download('punkt')

def ler_pdf(caminho):
    with open(caminho, 'rb') as arquivo:
        leitor = PyPDF2.PdfReader(arquivo)
        texto = ''
        for pagina in leitor.pages:
            texto += pagina.extract_text()
        return texto

def processar_texto(texto):
    tokens = nltk.word_tokenize(texto.lower())
    return ' '.join(tokens)

def criar_embeddings(textos):
    vetorizador = TfidfVectorizer()
    embeddings = vetorizador.fit_transform(textos)
    return vetorizador, embeddings

def pesquisar(query, vetorizador, embeddings, textos):
    query_embedding = vetorizador.transform([query])
    similaridades = cosine_similarity(query_embedding, embeddings).flatten()
    resultados = [(similaridade, texto) for similaridade, texto in zip(similaridades, textos)]
    resultados.sort(reverse=True, key=lambda x: x[0])
    return resultados

def encontrar_trecho_relevante(query, texto, tamanho_janela=100):
    """
    Encontra o trecho do texto mais relevante para a consulta.
    """
    palavras_query = set(query.lower().split())
    palavras_texto = texto.lower().split()
    
    melhor_trecho = ''
    melhor_pontuacao = 0
    
    # Procura o trecho com mais palavras da consulta
    for i in range(0, len(palavras_texto), tamanho_janela):
        trecho = ' '.join(palavras_texto[i:i + tamanho_janela])
        palavras_trecho = set(trecho.split())
        pontuacao = len(palavras_query.intersection(palavras_trecho))
        
        if pontuacao > melhor_pontuacao:
            melhor_pontuacao = pontuacao
            melhor_trecho = trecho
    
    return melhor_trecho if melhor_trecho else texto[:tamanho_janela]  # Retorna o início se não encontrar nada

def main():
    pasta = r'C:\Users\Anderson Almeida\Desktop\teste_pdf'
    textos = []
    nomes_arquivos = []
    
    for arquivo in os.listdir(pasta):
        if arquivo.endswith('.pdf'):
            caminho = os.path.join(pasta, arquivo)
            texto = ler_pdf(caminho)
            texto_processado = processar_texto(texto)
            textos.append(texto_processado)
            nomes_arquivos.append(arquivo)

    vetorizador, embeddings = criar_embeddings(textos)

    while True:
        query = input("Digite sua pesquisa (ou 'sair' para terminar): ")
        if query.lower() == 'sair':
            break
        
        resultados = pesquisar(query, vetorizador, embeddings, textos)
        
        for i, (similaridade, texto) in enumerate(resultados[:5]):  # Mostrar os top 5 resultados
            trecho_relevante = encontrar_trecho_relevante(query, texto)
            print(f"\nArquivo: {nomes_arquivos[i]}")
            print(f"Similaridade: {similaridade:.4f}")
            print(f"Trecho relevante:\n{trecho_relevante}\n")

if __name__ == "__main__":
    main()


from pyspark.sql import SparkSession
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier, RandomForestClassifier,
    GBTClassifier, LinearSVC, NaiveBayes, MultilayerPerceptronClassifier,
    FMClassifier
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from synapse.ml.lightgbm import LightGBMClassifier
from pyspark.ml.classification import OneVsRest
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import optuna

# Configurar o SparkSession para usar o SynapseML
spark = SparkSession.builder \
    .appName("AllModelsComparison") \
    .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.10.0") \
    .getOrCreate()

# Exemplo de DataFrame
data = [
    (1.0, 2.0, 3.0, 0),
    (4.0, 5.0, 6.0, 1),
    (7.0, 8.0, 9.0, 0),
    (10.0, 11.0, 12.0, 1)
]
columns = ["feature1", "feature2", "feature3", "target"]  # "target" é a coluna de rótulos
df = spark.createDataFrame(data, columns)

# Lista de features (nomes das colunas)
feature_cols = ["feature1", "feature2", "feature3"]

# Nome da coluna de predição (rótulos)
label_col = "target"

# Criar um VectorAssembler para combinar as features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Dividir o DataFrame em treino e teste
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Função para calcular o KS
def calculate_ks(predictions):
    # Ordenar as previsões por probabilidade da classe positiva
    sorted_predictions = predictions.orderBy(F.desc("probability"))
    
    # Calcular a taxa acumulada de verdadeiros positivos (TPR) e falsos positivos (FPR)
    tpr = sorted_predictions.withColumn("tpr", F.sum("label").over(Window.orderBy(F.desc("probability"))))
    fpr = sorted_predictions.withColumn("fpr", F.sum(1 - F.col("label")).over(Window.orderBy(F.desc("probability"))))
    
    # Calcular o KS
    ks = tpr.withColumn("ks", F.col("tpr") - F.col("fpr")).agg(F.max("ks")).collect()[0][0]
    return ks

# Função de objetivo para otimização com Optuna
def objective(trial, model_name, train_df, test_df, label_col):
    if model_name == "LogisticRegression":
        model = LogisticRegression(
            featuresCol='features',
            labelCol=label_col,
            regParam=trial.suggest_float("regParam", 0.01, 10.0, log=True),
            elasticNetParam=trial.suggest_float("elasticNetParam", 0.0, 1.0)
        )
    elif model_name == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(
            featuresCol='features',
            labelCol=label_col,
            maxDepth=trial.suggest_int("maxDepth", 2, 10),
            minInstancesPerNode=trial.suggest_int("minInstancesPerNode", 1, 10)
        )
    elif model_name == "RandomForestClassifier":
        model = RandomForestClassifier(
            featuresCol='features',
            labelCol=label_col,
            numTrees=trial.suggest_int("numTrees", 10, 100),
            maxDepth=trial.suggest_int("maxDepth", 2, 10)
        )
    elif model_name == "GBTClassifier":
        model = GBTClassifier(
            featuresCol='features',
            labelCol=label_col,
            maxIter=trial.suggest_int("maxIter", 10, 100),
            maxDepth=trial.suggest_int("maxDepth", 2, 10)
        )
    elif model_name == "LinearSVC":
        model = LinearSVC(
            featuresCol='features',
            labelCol=label_col,
            regParam=trial.suggest_float("regParam", 0.01, 10.0, log=True),
            maxIter=trial.suggest_int("maxIter", 10, 100)
        )
    elif model_name == "NaiveBayes":
        model = NaiveBayes(
            featuresCol='features',
            labelCol=label_col,
            smoothing=trial.suggest_float("smoothing", 0.0, 10.0)
        )
    elif model_name == "MultilayerPerceptronClassifier":
        model = MultilayerPerceptronClassifier(
            featuresCol='features',
            labelCol=label_col,
            layers=[3, trial.suggest_int("hiddenLayerSize", 2, 10), 2],
            maxIter=trial.suggest_int("maxIter", 10, 100)
        )
    elif model_name == "FMClassifier":
        model = FMClassifier(
            featuresCol='features',
            labelCol=label_col,
            factorSize=trial.suggest_int("factorSize", 2, 10),
            regParam=trial.suggest_float("regParam", 0.01, 10.0, log=True)
        )
    elif model_name == "LightGBMClassifier":
        model = LightGBMClassifier(
            featuresCol='features',
            labelCol=label_col,
            numLeaves=trial.suggest_int("numLeaves", 10, 100),
            maxDepth=trial.suggest_int("maxDepth", 2, 10),
            learningRate=trial.suggest_float("learningRate", 0.01, 0.3, log=True)
        )
    elif model_name == "OneVsRest":
        base_model = LogisticRegression(
            featuresCol='features',
            labelCol=label_col,
            regParam=trial.suggest_float("regParam", 0.01, 10.0, log=True),
            elasticNetParam=trial.suggest_float("elasticNetParam", 0.0, 1.0)
        )
        model = OneVsRest(classifier=base_model)
    
    # Treinar o modelo
    fitted_model = model.fit(train_df)
    
    # Fazer previsões no conjunto de teste
    predictions = fitted_model.transform(test_df)
    
    # Avaliar a acurácia (métrica a ser otimizada)
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    
    return accuracy

# Definir todos os modelos
models = [
    "LogisticRegression",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "GBTClassifier",
    "LinearSVC",
    "NaiveBayes",
    "MultilayerPerceptronClassifier",
    "FMClassifier",
    "LightGBMClassifier",
    "OneVsRest"
]

# Criar um DataFrame para armazenar os resultados
results = []

# Métricas a serem avaliadas
metrics = ["accuracy", "weightedPrecision", "weightedRecall", "f1"]

# Loop para treinar e avaliar cada modelo
for model_name in models:
    print(f"Otimizando {model_name}...")
    
    try:
        # Otimizar hiperparâmetros com Optuna
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_name, train_df, test_df, label_col), n_trials=10)
        
        # Melhores hiperparâmetros
        best_params = study.best_params
        print(f"Melhores hiperparâmetros para {model_name}: {best_params}")
        
        # Treinar o modelo com os melhores hiperparâmetros
        if model_name == "LogisticRegression":
            model = LogisticRegression(featuresCol='features', labelCol=label_col, **best_params)
        elif model_name == "DecisionTreeClassifier":
            model = DecisionTreeClassifier(featuresCol='features', labelCol=label_col, **best_params)
        elif model_name == "RandomForestClassifier":
            model = RandomForestClassifier(featuresCol='features', labelCol=label_col, **best_params)
        elif model_name == "GBTClassifier":
            model = GBTClassifier(featuresCol='features', labelCol=label_col, **best_params)
        elif model_name == "LinearSVC":
            model = LinearSVC(featuresCol='features', labelCol=label_col, **best_params)
        elif model_name == "NaiveBayes":
            model = NaiveBayes(featuresCol='features', labelCol=label_col, **best_params)
        elif model_name == "MultilayerPerceptronClassifier":
            model = MultilayerPerceptronClassifier(featuresCol='features', labelCol=label_col, **best_params)
        elif model_name == "FMClassifier":
            model = FMClassifier(featuresCol='features', labelCol=label_col, **best_params)
        elif model_name == "LightGBMClassifier":
            model = LightGBMClassifier(featuresCol='features', labelCol=label_col, **best_params)
        elif model_name == "OneVsRest":
            base_model = LogisticRegression(featuresCol='features', labelCol=label_col, **best_params)
            model = OneVsRest(classifier=base_model)
        
        # Treinar o modelo
        fitted_model = model.fit(train_df)
        
        # Fazer previsões no conjunto de teste
        predictions = fitted_model.transform(test_df)
        
        # Avaliar o desempenho do modelo para cada métrica
        model_metrics = {"Model": model_name}
        for metric in metrics:
            evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName=metric)
            metric_value = evaluator.evaluate(predictions)
            model_metrics[metric] = metric_value
        
        # Métricas binárias (AUC e KS)
        if len(predictions.select(label_col).distinct().collect()) == 2:  # Verificar se é um problema binário
            # AUC
            auc_evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
            auc = auc_evaluator.evaluate(predictions)
            model_metrics["AUC"] = auc
            
            # KS
            ks = calculate_ks(predictions)
            model_metrics["KS"] = ks
        
        # Matriz de Confusão
        confusion_matrix = predictions.groupBy(label_col, "prediction").count().orderBy(label_col, "prediction")
        model_metrics["ConfusionMatrix"] = confusion_matrix.collect()
        
        # Salvar os resultados
        results.append(model_metrics)
        print(f"{model_name} - Métricas: {model_metrics}")
    except Exception as e:
        print(f"Erro ao treinar {model_name}: {str(e)}")

# Converter os resultados para um DataFrame Spark
results_df = spark.createDataFrame(results)

# Mostrar os resultados
results_df.show(truncate=False)

# Salvar os resultados em um arquivo (opcional)
results_df.write.mode("overwrite").csv("path/to/save/results")

# Parar a sessão Spark (opcional)
spark.stop()
    
    
    
    
    
    
    
    
    
