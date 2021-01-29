# -*- coding: utf-8 -*-

# ETAPA 1 - IMPORTACAO E INSTALAÇAO DAS BIBLIOTECAS
import pandas as pd
import string
import spacy
import random
import seaborn as sns
import numpy as np
import re

# ETAPA 2 - CARREGAMENTO DA BASE DE DADOS
  #Link Kaggle: https://www.kaggle.com/augustop/portuguese-tweets-for-sentiment-analysis#TweetsNeutralHash.csv
  
  
# Base de treinamento

 # Negative label: 0
 # Positive label: 1
 
base_treinamento = pd.read_csv('C:\\Users\\olive\\Desktop\\Classificador_de_sentimentos\\DADOS_TWITTER\\TrainingDatasets\\Train50.csv', delimiter=';')
 
base_treinamento.shape 
base_treinamento.head()
base_treinamento.tail()
sns.countplot(base_treinamento['sentiment'], label = 'Contagem');

base_treinamento.drop(['id', 'tweet_date', 'query_used'], axis = 1, inplace=True)
base_treinamento.head()

sns.heatmap(pd.isnull(base_treinamento));

# Base de teste

base_teste = pd.read_csv('C:\\Users\\olive\\Desktop\\Classificador_de_sentimentos\\DADOS_TWITTER\\TestDatasets\\Test.csv', delimiter=';')

base_teste.head()
base_teste.shape

sns.countplot(base_teste['sentiment'], label='Contagem');

# Isolar tweet e sentimento
base_teste.drop(['id', 'tweet_date', 'query_used'], axis = 1, inplace=True)

base_teste.head()
sns.heatmap(pd.isnull(base_teste));

# ETAPA 3 - FUNÇÃO PARA PREPROCESSAMENTO DOS TEXTOS

    #Letras minúsculas
    #Nome do usuário (@)
    #URLs
    #Espaços em branco
    #Emoticons
    #Stop words
    #Lematização
    #Pontuações

pln = spacy.load('pt')
pln

base_treinamento['tweet_text'][1]

stop_words = spacy.lang.pt.stop_words.STOP_WORDS
print(stop_words)

string.punctuation

def preprocessamento(texto):
  # Letras minúsculas
  texto = texto.lower()

  # Nome do usuário
  texto = re.sub(r"@[A-Za-z0-9$-_@.&+]+", ' ', texto)

  # URLs
  texto = re.sub(r"https?://[A-Za-z0-9./]+", ' ', texto)

  # Espaços em branco
  texto = re.sub(r" +", ' ', texto)

  # Emoticons
  lista_emocoes = {':)': 'emocaopositiva',
                   ':d': 'emocaopositiva',
                   ':(': 'emocaonegativa'}
  for emocao in lista_emocoes:
    texto = texto.replace(emocao, lista_emocoes[emocao])
    
  #return text

  # Lematização
  documento = pln(texto)

  lista = []
  for token in documento:
    lista.append(token.lemma_)
  
  # Stop words e pontuações
  lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in string.punctuation]
  lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])
  
  return lista

texto_teste = '@behin_d_curtain :D Para :( mim, https://anaconda.org/anaconda/regex é precisamente o contrário :) Vem a chuva e vem a boa disposição :)'
resultado = preprocessamento(texto_teste)
resultado

# ETAPA 4 - PREPROCESSAMENTO DA BASE DE DADOS

base_treinamento.head(10)

base_treinamento['tweet_text'] = base_treinamento['tweet_text'].apply(preprocessamento)
base_treinamento.head(10)

base_teste['tweet_text'] = base_teste['tweet_text'].apply(preprocessamento)

base_teste.head(10)


# Tratamento da classe
exemplo_base_dados = [["este trabalho é agradável", {"POSITIVO": True, "NEGATIVO": False}],
                      ["este lugar continua assustador", {"POSITIVO": False, "NEGATIVO": True}]]

base_dados_treinamento_final = []
for texto, emocao in zip(base_treinamento['tweet_text'], base_treinamento['sentiment']):
  if emocao == 1:
    dic = ({'POSITIVO': True, 'NEGATIVO': False})
  elif emocao == 0:
    dic = ({'POSITIVO': False, 'NEGATIVO': True})

  base_dados_treinamento_final.append([texto, dic.copy()])
  
len(base_dados_treinamento_final)

base_dados_treinamento_final[10:15]
base_dados_treinamento_final[45000:45005]

# ETAPA 5 - CRIAÇÃO DO CLASSIFICADOR

modelo = spacy.blank('pt')
categorias = modelo.create_pipe("textcat")
categorias.add_label("POSITIVO")
categorias.add_label("NEGATIVO")
modelo.add_pipe(categorias)
historico = []

modelo.begin_training()
for epoca in range(20):
  random.shuffle(base_dados_treinamento_final)
  losses = {}
  for batch in spacy.util.minibatch(base_dados_treinamento_final, 512):
    textos = [modelo(texto) for texto, entities in batch]
    annotations = [{'cats': entities} for texto, entities in batch]
    modelo.update(textos, annotations, losses=losses)
    historico.append(losses)
  if epoca % 5 == 0:
    print(losses)
    
historico_loss = []
for i in historico:
  historico_loss.append(i.get('textcat'))
  
historico_loss = np.array(historico_loss)
historico_loss

import matplotlib.pyplot as plt
plt.plot(historico_loss)
plt.title('Progressão do erro')
plt.xlabel('Batches')
plt.ylabel('Erro')

modelo.to_disk('modelo')

# ETAPA 6 - TESTES COM UMA FRASE

modelo_carregado = spacy.load('modelo')
modelo_carregado

texto_positivo = base_teste['tweet_text'][21]
texto_positivo

previsao = modelo_carregado(texto_positivo)
previsao

previsao.cats

texto_positivo = 'eu gosto muito de você'
texto_positivo = preprocessamento(texto_positivo)


































