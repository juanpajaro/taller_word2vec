import numpy as np
from taller import softmax

def test_softmax_output():
    tmp = np.array([[1,2,3],
                    [1,1,1]
                   ])
    expected = np.array([[0.5, 0.73105858, 0.88079708],
                         [0.5, 0.26894142, 0.11920292]])
    try:
        result = softmax(tmp)
        if np.allclose(result, expected, atol=1e-6):
            print("La función softmax retorna la matriz esperada:")
            print(result)
        else:
            print("Respuesta incorrecta. La matriz retornada fue:")
            print(result)
            print("\nLa respuesta esperada es:")
            print(expected)
            print("\nIndicación: Verifica la implementación de la función softmax. Revisa el uso de np.exp y la suma en el eje correcto.")
    except Exception as e:
        print(f"Error al ejecutar softmax: {e}")
        print("\nIndicación: Revisa que la función softmax esté correctamente definida y que los argumentos sean válidos.")

if __name__ == "__main__":
    test_softmax_output()
