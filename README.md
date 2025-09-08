
# Taller Word2Vec

Este repositorio contiene un taller para construir y visualizar embeddings de palabras usando técnicas de PLN y redes neuronales simples.

## Estructura principal

- `taller.py`: contiene la lógica principal para el procesamiento de texto, inicialización de matrices, forward y backward propagation, y entrenamiento.
- `utils2.py`: funciones auxiliares para el taller.
- `visuzalizar_embeddings.py`: script para visualizar los embeddings generados.

## ¿Qué debes modificar?

Las funciones que debes completar están marcadas con:
```python
### START CODE HERE (Replace instances of 'None' with your own code) ###
```

Estas funciones son:
- `initialize_model`: inicializa las matrices y vectores del modelo.
- `softmax`: implementa la función softmax para normalizar las salidas.
- `forward_prop`: realiza la propagación hacia adelante en la red.
- `compute_cost`: calcula la función de costo (cross-entropy).

## Indicaciones para construir embeddings de palabras

Como profesor de PLN, te recomiendo seguir estos pasos para construir embeddings de palabras:

1. **Inicialización de parámetros**: En `initialize_model`, debes crear las matrices de pesos `W1` y `W2` y los vectores de sesgo `b1` y `b2` con las dimensiones correctas. Usa inicialización aleatoria y asegúrate de que las formas sean las especificadas en los comentarios.


2. **Propagación hacia adelante**: En `forward_prop`, implementa el cálculo del vector oculto `h` y la salida `z` usando las matrices y vectores inicializados. Aplica la función de activación ReLU sobre `h`.
	 - Fórmulas matemáticas:
		 - Cálculo del vector oculto:
			 $$ h = \text{ReLU}(W_1 x + b_1) $$
		 - Cálculo de la salida:
			 $$ z = W_2 h + b_2 $$
		 - Donde $W_1$ y $W_2$ son matrices de pesos, $b_1$ y $b_2$ son vectores de sesgo, $x$ es el vector de entrada (contexto), y ReLU es la función de activación $\text{ReLU}(a) = \max(0, a)$.

3. **Softmax**: En la función `softmax`, normaliza la salida $z$ para obtener probabilidades.
	 - Fórmula matemática:
		 $$ \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} $$
	 - Explicación: La función softmax convierte el vector de scores $z$ en un vector de probabilidades, donde cada elemento está en el rango [0, 1] y la suma total es 1. Es útil para problemas de clasificación multiclase.

4. **Función de costo**: En `compute_cost`, implementa la función de cross-entropy para comparar las predicciones con las etiquetas verdaderas.
	 - Fórmula matemática:
		 $$ J = -\frac{1}{m} \sum_{i=1}^m \sum_{j=1}^V y_{ij} \log(\hat{y}_{ij}) $$
		 donde $m$ es el tamaño del batch, $V$ es el tamaño del vocabulario, $y_{ij}$ es la etiqueta verdadera (one-hot), y $\hat{y}_{ij}$ es la probabilidad predicha por softmax.
	 - Explicación: La función de costo de entropía cruzada mide la diferencia entre las distribuciones de probabilidad predicha y verdadera. Penaliza más los errores en las predicciones de alta confianza.

5. **Entrenamiento**: Usa la función `gradient_descent` para actualizar los parámetros del modelo usando los gradientes calculados en `back_prop`.

6. **Visualización de embeddings**: En `visuzalizar_embeddings.py`, asegúrate de que la selección de los embeddings se haga correctamente. El promedio de los embeddings se calcula como `(W1.T + W2) / 2.0`. Puedes modificar la lista de palabras a visualizar y ajustar el número de dimensiones en la reducción PCA.

## Sugerencias de modificación

### En `taller.py`:
- Revisa y completa todas las funciones marcadas con `### START CODE HERE`. Asegúrate de que las formas de las matrices sean correctas y que las operaciones matemáticas sean las adecuadas para el modelo.
- Puedes agregar más pruebas unitarias para verificar que las dimensiones y valores sean los esperados.

### En `visuzalizar_embeddings.py`:
- Verifica que la función que selecciona los embeddings (`embs = (W1.T + W2)/2.0`) esté correctamente implementada.
- Puedes modificar la lista de palabras para visualizar otras palabras relevantes del corpus.
- Ajusta el número de iteraciones, tamaño de los embeddings (`N`) y parámetros de PCA para obtener mejores visualizaciones.

## Recomendaciones generales

- Lee cuidadosamente los comentarios y las instrucciones en cada función.
- Usa los tests incluidos para validar tus implementaciones.
- Si tienes dudas sobre alguna función matemática, revisa la documentación de NumPy y los recursos de PLN recomendados en clase.


## Ejercicio adicional

Debes cargar un libro de tu preferencia (que hayas leído) en formato de texto plano. Utiliza ese libro para construir los embeddings de palabras siguiendo el flujo del taller. Al final, explica brevemente (en el README o en un archivo aparte) cómo los embeddings representan la semántica del libro seleccionado y qué observaciones puedes hacer sobre las palabras visualizadas.

## Tabla de calificación

| Función / Tarea                      | Puntaje |
|--------------------------------------|---------|
| initialize_model                     |   0.5   |
| softmax                              |   0.5   |
| forward_prop                         |   0.5   |
| compute_cost                         |   0.5   |
| Visualización de embeddings          |   0.5   |
| Cargar y procesar un libro propio    |  1.25   |
| Explicación de los embeddings        |  1.25   |
| **Total**                            | **5**   |

¡Mucho éxito construyendo tus propios embeddings de palabras!
