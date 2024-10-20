﻿Descripción del Código
El código está estructurado para realizar las siguientes tareas:

Configuración de Entorno:

Se importan las bibliotecas necesarias, incluyendo math, os, numpy, PIL, tensorflow, cv2, ultralytics (YOLO), y otras para la manipulación de imágenes y la detección de rostros.
Se establecen las rutas de las bibliotecas de CUDA y GStreamer para permitir el uso de aceleración por GPU y procesamiento de flujos de video.

Carga del Modelo TFLite:

Se carga un modelo de TensorFlow Lite (ferplus.tflite) que se utiliza para la predicción de emociones.
Se inicializan las configuraciones del modelo, como los detalles de entrada y salida.
Definición de Clases:

Se definen las clases de emociones que el modelo puede detectar, incluyendo emociones como "feliz", "triste", "sorpresa", etc.
Generación de Clases Aleatorias:

Se define una función que genera clases aleatorias de un conjunto de 80 clases y presenta estas clases al usuario durante intervalos de tiempo.
Detección de Objetos:

La función process_video_stream_objects abre un flujo de video y utiliza un modelo YOLO para detectar objetos en cada cuadro.
Los objetos detectados se dibujan en el cuadro y se contabilizan si son de clases específicas seleccionadas aleatoriamente.
Preprocesamiento de Imágenes:

Se define una función preprocess_image para convertir las imágenes a escala de grises, redimensionarlas, y normalizarlas para la predicción de emociones.
Predicción de Emociones:

La función predict_emotion toma una imagen de una persona, la preprocesa, y realiza la inferencia en el modelo TFLite para obtener las emociones más probables.
Registro de Emociones:

Se crean funciones para registrar las emociones y puntuaciones en un archivo CSV cada 3 segundos.
Detección Facial y Reconocimiento de Emociones:

La función process_video_stream_face captura el video y detecta rostros usando un modelo YOLO especializado.
Los rostros detectados se recortan y se pasan a la función de predicción de emociones, y se muestran los resultados en la ventana de video.
Hilos para Ejecución Concurrente:

Se utilizan hilos para permitir el registro de emociones en segundo plano mientras se procesan los flujos de video.
Explicación de la Lógica
Importación y Configuración: El código inicia importando las bibliotecas necesarias y configurando los directorios de las bibliotecas de CUDA y GStreamer. Esto es crucial para asegurar que los modelos de aprendizaje profundo pueden utilizar la GPU y que los flujos de video pueden procesarse correctamente.

Carga del Modelo: La carga del modelo TFLite se realiza para que el código pueda predecir las emociones basándose en la imagen de una persona. Se utiliza interpreter.allocate_tensors() para asignar tensores en la memoria.

Detección de Objetos y Emociones: A través del uso de YOLO y el modelo TFLite, el código detecta objetos en tiempo real y predice emociones en las caras detectadas. Los resultados de las detecciones se muestran en la interfaz gráfica del usuario (GUI).

Registro y Almacenamiento: Cada 3 segundos, el sistema registra la emoción actual y su puntuación en un archivo CSV, lo que permite un seguimiento temporal de las emociones detectadas durante la ejecución.

Ejecución Concurrente: El uso de hilos permite que el registro de emociones y el procesamiento de video ocurran simultáneamente, lo que mejora la eficiencia y la fluidez del sistema.


Luciano Ernesto Díaz Salinas

Este proyecto está licenciado bajo la licencia [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/).
