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