import numpy as np
from taller import forward_prop, initialize_model

def test_forward_prop():
    # Ejemplo de entrada
    tmp_N = 2
    tmp_V = 3
    tmp_x = np.array([[0,1,0]]).T
    tmp_W1, tmp_W2, tmp_b1, tmp_b2 = initialize_model(N=tmp_N, V=tmp_V, random_seed=1)
    try:
        tmp_z, tmp_h = forward_prop(tmp_x, tmp_W1, tmp_W2, tmp_b1, tmp_b2)
        # Resultado esperado calculado manualmente
        # Puedes ajustar estos valores si tienes el resultado esperado exacto
        expected_z = np.dot(tmp_W2, np.maximum(0, np.dot(tmp_W1, tmp_x) + tmp_b1)) + tmp_b2
        expected_h = np.maximum(0, np.dot(tmp_W1, tmp_x) + tmp_b1)
        if np.allclose(tmp_z, expected_z, atol=1e-6) and np.allclose(tmp_h, expected_h, atol=1e-6):
            print("La función forward_prop retorna los valores esperados.")
            print("z:", tmp_z)
            print("h:", tmp_h)
        else:
            print("Respuesta incorrecta. Los valores retornados fueron:")
            print("z:", tmp_z)
            print("h:", tmp_h)
            print("\nLos valores esperados son:")
            print("z:", expected_z)
            print("h:", expected_h)
            print("\nIndicación: Verifica la implementación de forward_prop. Revisa el uso de np.dot, la suma de los bias y la función de activación relu.")
    except Exception as e:
        print(f"Error al ejecutar forward_prop: {e}")
        print("\nIndicación: Revisa que la función forward_prop esté correctamente definida y que los argumentos sean válidos.")

if __name__ == "__main__":
    test_forward_prop()
