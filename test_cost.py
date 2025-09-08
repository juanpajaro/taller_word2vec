import numpy as np
from taller import forward_prop, initialize_model, compute_cost, get_dict, cargar_datos, softmax, get_batches


def test_compute_cost():
    try:
        tmp_C = 2
        tmp_N = 50
        tmp_batch_size = 4
        data = cargar_datos()
        tmp_word2Ind, tmp_Ind2word = get_dict(data)
        tmp_V = len(tmp_word2Ind)

        tmp_x, tmp_y = next(get_batches(data, tmp_word2Ind, tmp_V, tmp_C, tmp_batch_size))

        tmp_W1, tmp_W2, tmp_b1, tmp_b2 = initialize_model(tmp_N, tmp_V)
        tmp_z, tmp_h = forward_prop(tmp_x, tmp_W1, tmp_W2, tmp_b1, tmp_b2)
        tmp_yhat = softmax(tmp_z)
        tmp_cost = compute_cost(tmp_y, tmp_yhat, tmp_batch_size)

        print(f"tmp_x.shape {tmp_x.shape}")
        print(f"tmp_y.shape {tmp_y.shape}")
        print(f"tmp_W1.shape {tmp_W1.shape}")
        print(f"tmp_W2.shape {tmp_W2.shape}")
        print(f"tmp_b1.shape {tmp_b1.shape}")
        print(f"tmp_b2.shape {tmp_b2.shape}")
        print(f"tmp_z.shape: {tmp_z.shape}")
        print(f"tmp_h.shape: {tmp_h.shape}")
        print(f"tmp_yhat.shape: {tmp_yhat.shape}")
        print("call compute_cost")
        print(f"tmp_cost {tmp_cost:.4f}")

        expected_shapes = {
            'tmp_x': (23424, 4),
            'tmp_y': (23424, 4),
            'tmp_W1': (50, 23424),
            'tmp_W2': (23424, 50),
            'tmp_b1': (50, 1),
            'tmp_b2': (23424, 1),
            'tmp_z': (23424, 4),
            'tmp_h': (50, 4),
            'tmp_yhat': (23424, 4)
        }
        actual_shapes = {
            'tmp_x': tmp_x.shape,
            'tmp_y': tmp_y.shape,
            'tmp_W1': tmp_W1.shape,
            'tmp_W2': tmp_W2.shape,
            'tmp_b1': tmp_b1.shape,
            'tmp_b2': tmp_b2.shape,
            'tmp_z': tmp_z.shape,
            'tmp_h': tmp_h.shape,
            'tmp_yhat': tmp_yhat.shape
        }
        for key in expected_shapes:
            if actual_shapes[key] != expected_shapes[key]:
                print(f"Error: {key} tiene shape {actual_shapes[key]}, pero se esperaba {expected_shapes[key]}")
                print("Indicación: Revisa la inicialización y el flujo de datos para que las dimensiones coincidan.")
        # Se acepta como correcto si la diferencia es <= 0.05 o si el valor es menor a 14.11
        if abs(tmp_cost - 14.1139) <= 0.05 or tmp_cost < 14.11:
            print("La función compute_cost retorna un valor aceptable.")
        else:
            print(f"Error: tmp_cost es {tmp_cost:.4f}, pero se esperaba 14.1139 o un valor menor a 14.11")
            print("Indicación: Revisa la función compute_cost y los datos de entrada.")
    except Exception as e:
        print(f"Error al ejecutar el test: {e}")
        print("Indicación: Revisa que todas las funciones estén correctamente implementadas y que los datos sean válidos.")

if __name__ == "__main__":
    test_compute_cost()