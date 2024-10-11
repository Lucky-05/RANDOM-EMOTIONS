import math
import os
c = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\CUDA\\v12.1\\bin"
a = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\lib"
b= "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\include"
gstreamerPath = "C:\gstreamer\\1.0\\msvc_x86_64\\bin"
os.add_dll_directory(gstreamerPath)
os.add_dll_directory(c)
os.add_dll_directory(a)
os.add_dll_directory(b)
import numpy as np
from PIL import Image
import tensorflow as tf  # O usa tflite_runtime.interpreter si estás usando tflite_runtime
import csv
import time
import cv2
import numpy as np
from ultralytics import YOLO
import concurrent.futures
from sort import Sort
import threading
import dlib
from imutils import face_utils
from collections import deque
import face_recognition
from datetime import datetime
import tkinter as tk
import random
import threading

# Cargar el modelo TFLite
model_path = "ferplus.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
tracker = Sort()
tracker0 = Sort()
name = ""
current_emotion = "Desconocida"
# Obtener detalles de entrada y salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
counter = 0

# Clases para las detecciones de las redes neuronales
emotion_classes = ['neutral', 'Happy', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']  # Ajusta las clases según tu modelo


def preprocess_image(image):
    img = Image.fromarray(image).convert('L')  
    img = img.resize((48, 48))  
    img = np.array(img, dtype=np.float32)
    img = img / 255.0 

    img = np.expand_dims(img, axis=-1)  
    img = np.expand_dims(img, axis=0)   
    return img

def predict_emotion(person_img):
    input_data = preprocess_image(person_img)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_probs = output_data[0]

    predicted_indices = np.argsort(predicted_probs)[-2:][::-1]
    top_emotions = [(emotion_classes[i], predicted_probs[i]) for i in predicted_indices]
    
    return top_emotions[0]  

emotion_log = []
puntaje = 0
log_interval = 3  # Cada 3 segundos
current_emotion = "Desconocida"  # Inicializa la emoción como desconocida
csv_file = "emotion_log.csv"  # Nombre del archivo CSV

# Función para crear el archivo CSV si no existe
def create_csv_file():
    if not os.path.exists(csv_file):
        try:
            with open(csv_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Tiempo", "Emoción", "Puntaje"])  # Escribir encabezados
            print(f"Archivo CSV creado: {csv_file}")
        except IOError as e:
            print(f"Error al crear el archivo CSV: {e}")

# Función para registrar emoción y puntaje en el archivo CSV
def log_emotion(emotion, puntaje):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # Formato del tiempo
    emotion_log.append((current_time, emotion, puntaje))  # Guardar en una lista
    print(f"Tiempo: {current_time}, Emoción: {emotion}, Puntaje: {puntaje}")

    # Intentar escribir en el archivo CSV
    try:
        with open(csv_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([current_time, emotion, puntaje])
    except IOError as e:
        print(f"Error al escribir en el archivo CSV: {e}")

# Función para iniciar el registro cada 3 segundos
def start_logging():
    while True:
        log_emotion(current_emotion, puntaje)  # Usa la variable global 'current_emotion'
        time.sleep(log_interval)

def process_video_stream_face(source, stream_id):
    global current_emotion, puntaje

    face_model = YOLO("yolov8n-face.pt")
    face_model.to('cuda')
    cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)

    # Crear el archivo CSV si no existe
    create_csv_file()

    # Iniciar un hilo para registrar la emoción y puntaje cada 3 segundos
    logging_thread = threading.Thread(target=start_logging, daemon=True)
    logging_thread.start()

    while True:
        if not cap.isOpened():
            print(f"Unable to open video source {source}. Retrying...")
            cap = cv2.VideoCapture(source)
            continue

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Retrying...")
            cap.release()
            cap = cv2.VideoCapture(source)
            continue

        # Detección facial usando el modelo YOLO
        face_results = face_model(frame, stream=True)

        # Procesar los resultados de la detección facial
        for res in face_results:
            filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.27)[0]
            face_boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
            tracks = tracker.update(face_boxes)
            tracks = tracks.astype(int)

            for xmin, ymin, xmax, ymax, track_id in tracks:
                new_box = (xmin, ymin, xmax, ymax)
                padding = 10
                image_height, image_width, _ = frame.shape
                x1_new = max(0, xmin - padding)
                y1_new = max(0, ymin - padding)
                x2_new = min(image_width, xmax + padding)
                y2_new = min(image_height, ymax + padding)
                person_img = frame[y1_new:y2_new, x1_new:x2_new]  # Extraer la cara detectada

                # Predecir la emoción en el ROI (persona detectada)
                current_emotion, prob = predict_emotion(person_img)

                # Mostrar el nombre y la emoción en la imagen
                label = f"Bienvenido {name}, Emocion: {current_emotion} ({prob:.2f})"
                cv2.putText(img=frame, text=label, org=(xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255,255,255), thickness=2)
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)

        # Mostrar el frame actualizado
        cv2.imshow(f"Stream {stream_id}", frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

process_video_stream_face(0,"Face")
