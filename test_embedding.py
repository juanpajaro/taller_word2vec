from visuzalizar_embeddings import embedding_vectors
import numpy as np

def test_embedding_vectors_exact():
    try:
        embs, word2Ind, words = embedding_vectors(N=50)
        expected_shape = (23424, 50)
        expected_first_row = np.array([
            0.46098255, 0.81489086, 0.93156337, 0.55286354, 0.57684189, 0.16222413,
            0.59137119, 0.52295451, 0.33392353, 0.93694233, 0.56203495, 0.66300384,
            0.60171298, 0.82751255, 0.47696448, 0.82562782, 0.52777277, 0.33694564,
            0.90680548, 0.74677524, 0.72808069, 0.51131873, 0.79809396, 0.60873604,
            0.22241217, 0.46105669, 0.64762205, 0.70208283, 0.8705913 , 0.16321505,
            0.38524259, 0.56144908, 0.450184  , 0.3278456 , 0.64997737, 0.37045732,
            0.16140709, 0.65735461, 0.72124498, 0.94025211, 0.72436161, 0.41592088,
            0.12573253, 0.57632651, 0.49797665, 0.46819603, 0.8692494 , 0.23977666,
            0.6923717 , 0.36525298
        ])
        if embs.shape == expected_shape and np.allclose(embs[0], expected_first_row, atol=1e-6):
            print(f"Correcto: La matriz de embeddings tiene shape {embs.shape} y la primera fila es la esperada.")
            print("Primera fila de embs:")
            print(embs[0])
        else:
            print(f"Error: La matriz de embeddings o la primera fila no son las esperadas.")
            print(f"Shape obtenido: {embs.shape}, shape esperado: {expected_shape}")
            print("Primera fila obtenida:")
            print(embs[0])
            print("Primera fila esperada:")
            print(expected_first_row)
            print("Indicación: Verifica la función embedding_vectors y la generación de los embeddings.")
    except Exception as e:
        print(f"Error al ejecutar embedding_vectors: {e}")
        print("Indicación: Revisa la función embedding_vectors y los datos de entrada.")

if __name__ == "__main__":
    test_embedding_vectors_exact()

