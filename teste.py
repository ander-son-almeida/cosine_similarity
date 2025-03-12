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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
