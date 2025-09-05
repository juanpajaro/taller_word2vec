#Librerías y configuraciones
import re
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter
from utils2 import sigmoid, get_batches, compute_pca, get_dict
#nltk.download('punkt')
nltk.download('punkt_tab')
nltk.data.path.append('.')

with open("elquijote.txt") as f:
    data = f.read()

data = re.sub(r'[,!?;-]', '.', data) #Se remplazan signos de puntuación por puntos
data = nltk.word_tokenize(data) #Se tokeniza el texto
data = [word.lower() for word in data if word.isalpha()] #Se convierten las palabras a minúsculas y se eliminan los tokens que no son palabras
print("Number of tokens:", len(data),'\n', data[20:150])

# Compute the frequency distribution of the words in the dataset (vocabulary)
fdist = nltk.FreqDist(word for word in data)
print("Size of vocabulary: ",len(fdist) )
print("Most frequent tokens: ",fdist.most_common(20) ) # print the 20 most frequent words and their freq.

# get_dict creates two dictionaries, converting words to indices and viceversa.
word2Ind, Ind2word = get_dict(data)
V = len(word2Ind)
print("Size of vocabulary: ", V)

# example of word to index mapping
print("Index of the word 'dulcinea' :  ",word2Ind['dulcinea'] )
print("Word which has index 8097:  ",Ind2word[8096] )