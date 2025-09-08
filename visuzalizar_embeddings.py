import matplotlib.pyplot as plt
from taller import cargar_datos, gradient_descent
from utils2 import compute_pca, get_dict

def embedding_vectors(N=300):

    data = cargar_datos()
    word2Ind, Ind2word = get_dict(data)
    V = len(word2Ind)    
    num_iters = 150

    print("Call gradient_descent")
    W1, W2, b1, b2 = gradient_descent(data, word2Ind, N, V, num_iters)

    words = ['virtud', 'malo', 'bueno', 'molino', 'dulcinea','sancho','caballero',
            'mortal','ingratitud','memoria']

    embs = (W1.T + W2)/2.0
    print(embs.shape)  # embs.shape:  (23424, 50)
    print(embs[:1, :])  # First 1 word embeddings

    return embs, word2Ind, words

def visualize_embeddings(embs, word2Ind, words):

    # given a list of words and the embeddings, it returns a matrix with all the embeddings
    idx = [word2Ind[word] for word in words]
    X = embs[idx, :]
    print(X.shape, idx)  # X.shape:  Number of words of dimension N each

    result= compute_pca(X, 2)
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.savefig("word_embeddings.png")
    print("Figure saved as word_embeddings.png")

if __name__ == "__main__":
    N = 50  # dimension of the embedding vector
    embs, word2Ind, words = embedding_vectors(N=N)
    visualize_embeddings(embs, word2Ind, words)