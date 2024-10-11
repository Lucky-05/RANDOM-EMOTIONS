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

# initialize

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



#Code for cv detection
counter = 0
exluded_classes = []
yolo_classes = {0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}

current_classes = []  # Almacena las clases actuales
showing_classes = True  # Indica si estamos mostrando clases o preguntando
start_time = time.time()
current_class = ""
emotion_log = []
log_interval = 3  # Cada 3 segundos
current_emotion = "Desconocida"  # Inicializa la emoción como desconocida
csv_file = "emotion_log.csv"  # Nombre del archivo CSV
time_to_show_classes = 20

def generate_random_classes():
    global current_classes, showing_classes, start_time, exluded_classes
    while True:
        if showing_classes:
            exluded_classes = []
            current_classes = [random.randint(0, 79) for _ in range(3)]  # Seleccionar 5 clases aleatorias
            print(f"Clases mostradas: {[yolo_classes[class_id] for class_id in current_classes]}")
            start_time = time.time()  # Reiniciar el tiempo para la siguiente fase (pregunta)
            time.sleep(9)  # Mostrar cada clase durante 10 segundos, por un total de 50 segundos
            showing_classes = False  # Cambiar a la fase de preguntas
        else:
            start_time = time.time()  # Registrar el inicio del periodo de preguntas
            print("¿Cuáles eran las clases mostradas?")
            time.sleep(time_to_show_classes+6)  # Mostrar la pregunta durante 20 segundos
            showing_classes = True  

class_thread = threading.Thread(target=generate_random_classes)
class_thread.start()

def process_video_stream_objects(source, stream_id):
    global counter,current_classes, exluded_classes
    device_model = YOLO("yolov8n.pt")  # Second model for detecting cell phones/tablets
    
    device_model.to('cuda')
    
    cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
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

            results = device_model(frame, stream=True)
            
            for r in results:
                for box in r.boxes:
                    class_id = r.names[box.cls[0].item()]
                    class_id_num = box.cls[0].item()
                    cords = box.xyxy[0].tolist()
                    cords = [round(x) for x in cords]
                    conf = round(box.conf[0].item(), 2)
                    if conf > 0.6:
                        if ((class_id_num in current_classes) and (class_id not in exluded_classes)):
                            exluded_classes.append(class_id)
                            counter +=10
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        w, h = x2 - x1, y2 - y1
                        cv2.rectangle(frame, (x1, y1 - 30), (x1 + 200, y1), (255, 0, 255), cv2.FILLED)
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        text = f'{class_id} {conf} counter {counter}'
                        cv2.putText(frame, text, (max(0, x1), max(35, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(f"Stream {stream_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

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

# Code for emotion and facial detection and recogntion

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
    global current_emotion,counter
    while True:
        log_emotion(current_emotion, counter)  # Usa la variable global 'current_emotion'
        time.sleep(log_interval)

def getCurrentClass():
    global current_class
    time.sleep(6)
    while True:
        for i in range(len(current_classes)):
            current_class = yolo_classes[current_classes[i]]
            time.sleep(3)
        current_class = "Hora de memorizar"
        time.sleep(time_to_show_classes)

def process_video_stream_face(source, stream_id):
    global current_emotion, counter, current_class

    face_model = YOLO("yolov8n-face.pt")
    face_model.to('cuda')
    cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)

    # Crear el archivo CSV si no existe
    create_csv_file()

    # Iniciar un hilo para registrar la emoción y puntaje cada 3 segundos
    logging_thread = threading.Thread(target=start_logging, daemon=True)
    logging_thread.start()
    show_class_thread = threading.Thread(target=getCurrentClass, daemon=True)
    show_class_thread.start()

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
        
        cv2.putText(img=frame, text=str(current_class), org=(200, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255,255,255), thickness=2)

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

#process_video_stream_face(0, "Face")
#process_video_stream_objects(1, "Objects")

thread = threading.Thread(target=process_video_stream_face, args=(0, "Face"))
thread0 = threading.Thread(target=process_video_stream_objects, args=(1, "Detection"))
thread.start()
thread0.start()
thread.join()
thread0.join()
