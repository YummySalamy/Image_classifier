import tensorflow as tf
import tensorflow_datasets as tfds

datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

metadatos

datos_entrenamiento, datos_pruebas = datos['train'], datos['test']

nombres_clases = metadatos.features['label'].names

nombres_clases

def normalizar(imagenes, etiquetas):
  imagenes = tf.cast(imagenes, tf.float32)
  imagenes /= 255 #From [0-255] to [0-1]
  return imagenes, etiquetas

#normalizar los datos de entrenamiento y pruebas con la funcion de normalización
datos_entrenamiento = datos_entrenamiento.map(normalizar)
datps_pruebas = datos_pruebas.map(normalizar)

#Agregar datos a caché (velocidad por uso de memoria)
datos_entrenamiento = datos_entrenamiento.cache()
datos_pruebas = datos_pruebas.cache()

#Mostrar una imagen de los datos de pruebas, mostrar primera.

for imagen, etiqueta in datos_entrenamiento.take(1):
  break
imagen = imagen.numpy().reshape((28,28)) # Redimensionar

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(imagen, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10,10))
for i, (imagen, etiqueta) in enumerate(datos_entrenamiento.take(25)):
  imagen = imagen.numpy().reshape((28,28))
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(imagen, cmap=plt.cm.binary)
  plt.xlabel(nombres_clases[etiqueta])
plt.show

modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28,28,1]), #1 - Blanco y negro
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax), #Funcion de activacion de capa para red de clasificacion
])

#Compilar

modelo.compile(
    optimizer ='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

cant_ej_entrenamiento = metadatos.splits["train"].num_examples
cant_ej_pruebas = metadatos.splits["test"].num_examples
print(f"La cantidad de ejemplos para el entrenamiento es: {cant_ej_entrenamiento}")
print(f"La cantidad de ejemplos para las pruebas es de: {cant_ej_pruebas}")

tamano_lote = 32

datos_entrenamiento = datos_entrenamiento.repeat().shuffle(cant_ej_entrenamiento).batch(tamano_lote)
datos_pruebas = datos_pruebas.batch(tamano_lote)

import math

historial = modelo.fit(datos_entrenamiento, epochs=5, steps_per_epoch=math.ceil(cant_ej_entrenamiento/tamano_lote))

plt.xlabel("# Epoch")
plt.ylabel("Escala de perdida")
plt.plot(historial.history['loss'])

import numpy as np

for imagenes_prueba, etiquetas_prueba in datos_pruebas.take(1):
  imagenes_prueba = imagenes_prueba.numpy()
  etiquetas_prueba = etiquetas_prueba.numpy()
  predicciones = modelo.predict(imagenes_prueba)

def graficar_imagen(i, arr_predicciones, etiquetas_reales, imagenes):
  arr_predicciones, etiqueta_real, img = arr_predicciones[i], etiquetas_reales[i], imagenes[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img[...,0], cmap=plt.cm.binary)

  etiqueta_prediccion = np.argmax(arr_predicciones)
  if etiqueta_prediccion == etiqueta_real:
    color = 'blue' #Prediccion correcta
  else:
    color = 'red' #Prediccion incorrecta

  plt.xlabel("{} {:2.0f}% ({})".format(
      nombres_clases[etiqueta_prediccion],
      100*np.max(arr_predicciones),
      nombres_clases[etiqueta_real],
      color=color
))
  
def graficar_valor_arreglo(i, arr_predicciones, etiqueta_real):
  arr_predicciones, etiqueta_real = arr_predicciones[i], etiqueta_real[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  grafica = plt.bar(range(10), arr_predicciones, color="#777777")
  plt.ylim([0,1])
  etiqueta_prediccion = np.argmax(arr_predicciones)

  grafica[etiqueta_prediccion].set_color('red')
  grafica[etiqueta_real].set_color('blue')

filas = 5
columnas = 5
num_imagenes = filas * columnas
plt.figure(figsize=(2*2*columnas, 2*filas))
for i in range (num_imagenes):
  plt.subplot(filas, 2*columnas, 2*i+1)
  graficar_imagen(i, predicciones, etiquetas_prueba, imagenes_prueba)
  plt.subplot(filas, 2*columnas, 2*i+2)
  graficar_valor_arreglo(i, predicciones, etiquetas_prueba)

#Acceder a culaquier indice del set de pruebas para ver su prediccion
imagen = imagenes_prueba[5]
imagen = np.array([imagen])
prediccion = modelo.predict(imagen)

print("Prediccion: " + nombres_clases[np.argmax(prediccion[0])])

