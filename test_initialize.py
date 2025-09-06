import numpy as np
from taller import initialize_model

def test_initialize_model_shapes():
    tmp_N = 4
    tmp_V = 10
    tmp_W1, tmp_W2, tmp_b1, tmp_b2 = initialize_model(tmp_N, tmp_V)
    try:
        assert tmp_W1.shape == (tmp_N, tmp_V), f"Error: tmp_W1.shape es {tmp_W1.shape}, pero debería ser (4, 10). Revisa la inicialización de W1."
        assert tmp_W2.shape == (tmp_V, tmp_N), f"Error: tmp_W2.shape es {tmp_W2.shape}, pero debería ser (10, 4). Revisa la inicialización de W2."
        assert tmp_b1.shape == (tmp_N, 1), f"Error: tmp_b1.shape es {tmp_b1.shape}, pero debería ser (4, 1). Revisa la inicialización de b1."
        assert tmp_b2.shape == (tmp_V, 1), f"Error: tmp_b2.shape es {tmp_b2.shape}, pero debería ser (10, 1). Revisa la inicialización de b2."
        print("Las dimensiones de los parámetros son correctas:")
        print("tmp_W1.shape: (4, 10)")
        print("tmp_W2.shape: (10, 4)")
        print("tmp_b1.shape: (4, 1)")
        print("tmp_b2.shape: (10, 1)")
    except AssertionError as e:
        print(str(e))
        print("\nIndicación: Verifica que la función initialize_model esté retornando los parámetros con las dimensiones especificadas en el enunciado.")

if __name__ == "__main__":
    test_initialize_model_shapes()
