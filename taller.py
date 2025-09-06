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

def cargar_datos():
    with open("elquijote.txt") as f:
        data = f.read()    
    data = re.sub(r'[,!?;-]', '.', data) #Se remplazan signos de puntuación por puntos
    data = nltk.word_tokenize(data) #Se tokeniza el texto
    data = [word.lower() for word in data if word.isalpha()] #Se convierten las palabras a minúsculas y se eliminan los tokens que no son palabras
    return data

# get_dict creates two dictionaries, converting words to indices and viceversa.
def get_dict(data):
    word2Ind = {word: i for i, word in enumerate(set(data))}
    Ind2word = {i: word for word, i in word2Ind.items()}
    return word2Ind, Ind2word

def initialize_model(N,V, random_seed=1):
    '''
    Inputs:
        N:  dimension of hidden vector
        V:  dimension of vocabulary
        random_seed: random seed for consistent results in the unit tests
     Outputs:
        W1, W2, b1, b2: initialized weights and biases
    '''

    ### START CODE HERE (Replace instances of 'None' with your code) ###
    np.random.seed(random_seed)
    # W1 has shape (N,V)
    W1 = np.random.rand(N, V)

    # W2 has shape (V,N)
    W2 = np.random.rand(V, N)

    # b1 has shape (N,1)
    b1 = np.random.rand(N, 1)

    # b2 has shape (V,1)
    b2 = np.random.rand(V, 1)

    ### END CODE HERE ###
    return W1, W2, b1, b2

def softmax(z):
    '''
    Inputs:
        z: output scores from the hidden layer
    Outputs:
        yhat: prediction (estimate of y)
    '''
    ### START CODE HERE (Replace instances of 'None' with your own code) ###
    # Calculate yhat (softmax)
    yhat = np.exp(z)/ np.sum(np.exp(z), axis = 0)
    ### END CODE HERE ###
    return yhat

def forward_prop(x, W1, W2, b1, b2):
    '''
    Inputs:
        x:  average one hot vector for the context
        W1, W2, b1, b2:  matrices and biases to be learned
     Outputs:
        z:  output score vector
    '''

    ### START CODE HERE (Replace instances of 'None' with your own code) ###
    # Calculate h
    h = np.dot(W1, x) + b1

    # Apply the relu on h,
    # store the relu in h
    h = np.maximum(0, h)

    # Calculate z
    z = np.dot(W2, h) + b2

    ### END CODE HERE ###

    return z, h

# compute_cost: cross-entropy cost function
def compute_cost(y, yhat, batch_size):

    # cost function
    logprobs = np.multiply(np.log(yhat),y)  + np.multiply(np.log(1 - yhat), 1 - y)
    cost = - 1/batch_size * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost

def back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size):
    '''
    Inputs:
        x:  average one hot vector for the context
        yhat: prediction (estimate of y)
        y:  target vector
        h:  hidden vector (see eq. 1)
        W1, W2, b1, b2:  matrices and biases
        batch_size: batch size
     Outputs:
        grad_W1, grad_W2, grad_b1, grad_b2:  gradients of matrices and biases
    '''
    ### START CODE HERE (Replace instanes of 'None' with your code) ###
    # Compute l1 as W2^T (Yhat - Y)
    # and re-use it whenever you see W2^T (Yhat - Y) used to compute a gradient
    l1 = np.dot(W2.T, (yhat - y))

    # Apply relu to l1
    l1 = np.maximum(0, l1)

    # compute the gradient for W1
    grad_W1 = (1/ batch_size) * np.dot(l1, x.T)

    # Compute gradient of W2
    grad_W2 = (1/ batch_size) * np.dot((yhat - y), h.T)

    # compute gradient for b1
    grad_b1 = l1

    # compute gradient for b2
    grad_b2 = yhat - y
    ### END CODE HERE ####

    return grad_W1, grad_W2, grad_b1, grad_b2

def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03,
                     random_seed=282, initialize_model=initialize_model,
                     get_batches=get_batches, forward_prop=forward_prop,
                     softmax=softmax, compute_cost=compute_cost,
                     back_prop=back_prop):

    '''
    This is the gradient_descent function

      Inputs:
        data:      text
        word2Ind:  words to Indices
        N:         dimension of hidden vector
        V:         dimension of vocabulary
        num_iters: number of iterations
        random_seed: random seed to initialize the model's matrices and vectors
        initialize_model: your implementation of the function to initialize the model
        get_batches: function to get the data in batches
        forward_prop: your implementation of the function to perform forward propagation
        softmax: your implementation of the softmax function
        compute_cost: cost function (Cross entropy)
        back_prop: your implementation of the function to perform backward propagation
     Outputs:
        W1, W2, b1, b2:  updated matrices and biases after num_iters iterations

    '''
    W1, W2, b1, b2 = initialize_model(N,V, random_seed=random_seed) #W1=(N,V) and W2=(V,N)

    batch_size = 128
    #batch_size = 512
    iters = 0
    C = 2

    for x, y in get_batches(data, word2Ind, V, C, batch_size):
        ### START CODE HERE (Replace instances of 'None' with your own code) ###
        # get z and h
        z, h = forward_prop(x, W1, W2, b1, b2)

        # get yhat
        yhat = softmax(z)

        # get cost
        cost = compute_cost(y, yhat, batch_size)
        if ( (iters+1) % 10 == 0):
            print(f"iters: {iters + 1} cost: {cost:.6f}")

        # get gradients
        grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size)

        # update weights and biases
        W1 = W1 - alpha * grad_W1
        W2 = W2 - alpha * grad_W2
        b1 = b1 - alpha * grad_b1
        b2 = b2 - alpha * grad_b2

        ### END CODE HERE ###
        iters +=1
        if iters == num_iters:
            break
        if iters % 100 == 0:
            alpha *= 0.66

    return W1, W2, b1, b2

if __name__ == "__main__":

    data = cargar_datos()
    print("Number of tokens:", len(data),'\n', data[20:150])

    word2Ind, Ind2word = get_dict(data)
    V = len(word2Ind)
    print("Size of vocabulary: ", V)

    # example of word to index mapping
    dulci = word2Ind['dulcinea']
    print("Index of the word 'dulcinea' :  ",word2Ind['dulcinea'])
    print("Word which has index 21473:  ",Ind2word[dulci])

    """
    
  
    
    

    # Test the function
    tmp_C = 2
    tmp_N = 50
    tmp_batch_size = 4
    tmp_word2Ind, tmp_Ind2word = get_dict(data)
    tmp_V = len(word2Ind)

    tmp_x, tmp_y = next(get_batches(data, tmp_word2Ind, tmp_V,tmp_C, tmp_batch_size))

    print(f"tmp_x.shape {tmp_x.shape}")
    print(f"tmp_y.shape {tmp_y.shape}")

    tmp_W1, tmp_W2, tmp_b1, tmp_b2 = initialize_model(tmp_N,tmp_V)

    print(f"tmp_W1.shape {tmp_W1.shape}")
    print(f"tmp_W2.shape {tmp_W2.shape}")
    print(f"tmp_b1.shape {tmp_b1.shape}")
    print(f"tmp_b2.shape {tmp_b2.shape}")

    tmp_z, tmp_h = forward_prop(tmp_x, tmp_W1, tmp_W2, tmp_b1, tmp_b2)
    print(f"tmp_z.shape: {tmp_z.shape}")
    print(f"tmp_h.shape: {tmp_h.shape}")

    tmp_yhat = softmax(tmp_z)
    print(f"tmp_yhat.shape: {tmp_yhat.shape}")

    tmp_cost = compute_cost(tmp_y, tmp_yhat, tmp_batch_size)
    print("call compute_cost")
    print(f"tmp_cost {tmp_cost:.4f}")

    # Test the function
    tmp_C = 2
    tmp_N = 50
    tmp_batch_size = 4
    tmp_word2Ind, tmp_Ind2word = get_dict(data)
    tmp_V = len(word2Ind)

    # get a batch of data
    tmp_x, tmp_y = next(get_batches(data, tmp_word2Ind, tmp_V,tmp_C, tmp_batch_size))

    print("get a batch of data")
    print(f"tmp_x.shape {tmp_x.shape}")
    print(f"tmp_y.shape {tmp_y.shape}")

    print()
    print("Initialize weights and biases")
    tmp_W1, tmp_W2, tmp_b1, tmp_b2 = initialize_model(tmp_N,tmp_V)

    print(f"tmp_W1.shape {tmp_W1.shape}")
    print(f"tmp_W2.shape {tmp_W2.shape}")
    print(f"tmp_b1.shape {tmp_b1.shape}")
    print(f"tmp_b2.shape {tmp_b2.shape}")

    print()
    print("Forwad prop to get z and h")
    tmp_z, tmp_h = forward_prop(tmp_x, tmp_W1, tmp_W2, tmp_b1, tmp_b2)
    print(f"tmp_z.shape: {tmp_z.shape}")
    print(f"tmp_h.shape: {tmp_h.shape}")

    print()
    print("Get yhat by calling softmax")
    tmp_yhat = softmax(tmp_z)
    print(f"tmp_yhat.shape: {tmp_yhat.shape}")

    tmp_m = (2*tmp_C)
    tmp_grad_W1, tmp_grad_W2, tmp_grad_b1, tmp_grad_b2 = back_prop(tmp_x, tmp_yhat, tmp_y, tmp_h, tmp_W1, tmp_W2, tmp_b1, tmp_b2, tmp_batch_size)

    print()
    print("call back_prop")
    print(f"tmp_grad_W1.shape {tmp_grad_W1.shape}")
    print(f"tmp_grad_W2.shape {tmp_grad_W2.shape}")
    print(f"tmp_grad_b1.shape {tmp_grad_b1.shape}")
    print(f"tmp_grad_b2.shape {tmp_grad_b2.shape}")

    #Test gradien descent
    C = 2
    N = 50
    word2Ind, Ind2word = get_dict(data)
    V = len(word2Ind)
    num_iters = 150
    print("Call gradient_descent")
    W1, W2, b1, b2 = gradient_descent(data, word2Ind, N, V, num_iters)
    """

