# TP - Redes Neuronales
### - Materia: Matematica III
### - Participantes: Alcaraz, Maria Sol y Palomba, Damian

## **Conclusiones:**
El trabajo final tiene armado tres modelos distintos de redes donde dos tienen formas distinas de clasificar los datos para asi ver como varia la efectividad del entrenamiento. Por otro lado, esta el modelo con tensorflow que fue armado con los calculos hechos en el modelo "softmax", el objetivo de armarla fue para permitirnos tener un control.

El primer modelo de la red que tiene como funcion de activacion relu, se la entreno con los valores optimos pero se llego a la conclusion que la red fue entrenada obteniendo una desviacion estandar de 13 en 128 notas, es decir que en promedio los datos estan a 13 tonos aproximados del promedio. Por lo cual se puede considerar un buen resultado teniendo en cuenta que la red no es apta para procesar los datos que le dimos.

Luego de esto, se cambio la funcion de activacion de relu por la funcion softmax, donde se puede observar que al reducir la tasa de aprendizaje disminuye la red se va ajustando mejor, sn embargo el valor del accuracy no tiene cambios significativos. Obteniendo datoa que estan aproximadamente 5 tonos de la nota promedio. 

Teniendo en cuenta que este es el grafico de 'loss' para tensorflow, con 1000 iteraciones y una tasa de aprendizaje de 0.05 para la red hecha con tensorflow. Que resulta de un loss = 3.6 y un accuracy = 3.6%
(grafico_tensorflow.jpg)

Y este es el mejor grafico que determinamos de 'loss' para el modelo softmax, con 500 iteraciones y una tasa de aprendizaje de 0.01 para la red hecha con la funcion de activacion softmax para la capa de salida en la red. Que resulta finalmente de un loss = 4.08 y un accuracy = 2.64%
(grafico_softmax.png)
grafico_softmax.png

Entendiendo que con tensorflow obtenemos valores mas exactos en comparacion con los obtenidos a traves de forward propagation, podemos concluir que se ajusta lo suficientemente bien para poder predecir la siguiente nota lo mas acertado posible.
