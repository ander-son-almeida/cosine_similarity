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

https://microsoft.github.io/copilot-camp/pages/extend-m365-copilot/01-declarative-copilot/

    
    
    
    
    
    
