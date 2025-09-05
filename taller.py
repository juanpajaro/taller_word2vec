#Librerías y configuraciones
import re
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter
from utils2 import sigmoid, get_batches, compute_pca, get_dict
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.data.path.append('.')

with open("elquijote.txt") as f:
    data = f.read()

data = re.sub(r'[,!?;-]', '.', data) #Se remplazan signos de puntuación por puntos
data = nltk.word_tokenize(data) #Se tokeniza el texto
data = [word.lower() for word in data if word.isalpha()] #Se convierten las palabras a minúsculas y se eliminan los tokens que no son palabras
print("Number of tokens:", len(data),'\n', data[:15])
