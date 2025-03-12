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
from synapse.ml.automl import AutoMLClassifier
from synapse.ml.core.platform import running_on_synapse

# Iniciar uma sessão Spark
spark = SparkSession.builder.appName("AutoMLExample").getOrCreate()

# Exemplo de DataFrame
data = [
    (1.0, 2.0, 3.0, 0),
    (4.0, 5.0, 6.0, 1),
    (7.0, 8.0, 9.0, 0),
    (10.0, 11.0, 12.0, 1)
]
columns = ["feature1", "feature2", "feature3", "label"]  # "label" é a coluna de rótulos
df = spark.createDataFrame(data, columns)

# Criar um VectorAssembler para combinar as features
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=["feature1", "feature2", "feature3"], outputCol="features")
df = assembler.transform(df)

# Dividir o DataFrame em treino e teste
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Configurar o AutoMLClassifier
automl = AutoMLClassifier(
    task="classification",  # Tipo de tarefa (classificação)
    labelCol="label",       # Coluna de rótulos
    featuresCol="features", # Coluna de features
    primaryMetric="accuracy", # Métrica primária para avaliação
    maxIterations=10,       # Número máximo de iterações
    timeout=300             # Tempo máximo em segundos
)

# Treinar o modelo AutoML
print("Treinando o modelo AutoML...")
fitted_automl = automl.fit(train_df)

# Fazer previsões no conjunto de teste
predictions = fitted_automl.transform(test_df)

# Avaliar o desempenho do modelo
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Acurácia do modelo AutoML: {accuracy}")

# Mostrar as previsões
predictions.select("features", "label", "prediction").show()

# Parar a sessão Spark
spark.stop()
    
    
from pyspark.sql import SparkSession
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier, RandomForestClassifier,
    GBTClassifier, LinearSVC, NaiveBayes, MultilayerPerceptronClassifier,
    FMClassifier
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from synapse.ml.lightgbm import LightGBMClassifier
from pyspark.ml.classification import OneVsRest
from pyspark.ml import Pipeline

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

# Definir todos os modelos
models = [
    ("LogisticRegression", LogisticRegression(featuresCol='features', labelCol=label_col)),
    ("DecisionTreeClassifier", DecisionTreeClassifier(featuresCol='features', labelCol=label_col)),
    ("RandomForestClassifier", RandomForestClassifier(featuresCol='features', labelCol=label_col)),
    ("GBTClassifier", GBTClassifier(featuresCol='features', labelCol=label_col)),
    ("LinearSVC", LinearSVC(featuresCol='features', labelCol=label_col)),
    ("NaiveBayes", NaiveBayes(featuresCol='features', labelCol=label_col)),
    ("MultilayerPerceptronClassifier", MultilayerPerceptronClassifier(featuresCol='features', labelCol=label_col, layers=[3, 5, 2])),
    ("FMClassifier", FMClassifier(featuresCol='features', labelCol=label_col)),
    ("LightGBMClassifier", LightGBMClassifier(featuresCol='features', labelCol=label_col, predictionCol="prediction"))
]

# Adicionar OneVsRest (usando LogisticRegression como classificador base)
ovr = ("OneVsRest", OneVsRest(classifier=LogisticRegression(featuresCol='features', labelCol=label_col)))
models.append(ovr)

# Criar um DataFrame para armazenar os resultados
results = []

# Avaliador de desempenho
evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")

# Loop para treinar e avaliar cada modelo
for model_name, model in models:
    print(f"Treinando {model_name}...")
    
    try:
        # Treinar o modelo
        fitted_model = model.fit(train_df)
        
        # Fazer previsões no conjunto de teste
        predictions = fitted_model.transform(test_df)
        
        # Avaliar o desempenho do modelo
        accuracy = evaluator.evaluate(predictions)
        
        # Salvar os resultados
        results.append((model_name, accuracy))
        print(f"{model_name} - Acurácia: {accuracy}")
    except Exception as e:
        print(f"Erro ao treinar {model_name}: {str(e)}")

# Converter os resultados para um DataFrame Spark
results_df = spark.createDataFrame(results, ["Model", "Accuracy"])

# Mostrar os resultados
results_df.show()

# Salvar os resultados em um arquivo (opcional)
results_df.write.mode("overwrite").csv("path/to/save/results")

# Parar a sessão Spark (opcional)
spark.stop()
    
    
    
    
    
    
    
    
    
    
    
    
