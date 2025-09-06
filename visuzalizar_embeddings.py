import matplotlib.pyplot as plt
from taller import get_dict, cargar_datos, initialize_model, gradient_descent
from utils2 import compute_pca

data = cargar_datos()
word2Ind, Ind2word = get_dict(data)
V = len(word2Ind)
N = 300
num_iters = 150

print("Call gradient_descent")
W1, W2, b1, b2 = gradient_descent(data, word2Ind, N, V, num_iters)

words = ['virtud', 'malo', 'bueno', 'molino', 'dulcinea','sancho','caballero',
         'mortal','ingratitud','memoria']

embs = (W1.T + W2)/2.0

# given a list of words and the embeddings, it returns a matrix with all the embeddings
idx = [word2Ind[word] for word in words]
X = embs[idx, :]
print(X.shape, idx)  # X.shape:  Number of words of dimension N each

result= compute_pca(X, 2)
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.savefig("word_embeddings.png")