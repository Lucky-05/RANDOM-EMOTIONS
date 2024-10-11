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
start_time = time.time()  # Registro de tiempo inicial para controlar las transiciones

def generate_random_classes():
    global current_classes, showing_classes, start_time
    while True:
        if showing_classes:
            current_classes = [random.randint(0, 79) for _ in range(3)]  # Seleccionar 5 clases aleatorias
            print(f"Clases mostradas: {[yolo_classes[class_id] for class_id in current_classes]}")
            start_time = time.time()  # Reiniciar el tiempo para la siguiente fase (pregunta)
            time.sleep(10)  # Mostrar cada clase durante 10 segundos, por un total de 50 segundos
            showing_classes = False  # Cambiar a la fase de preguntas
        else:
            start_time = time.time()  # Registrar el inicio del periodo de preguntas
            print("¿Cuáles eran las clases mostradas?")
            time.sleep(20)  # Mostrar la pregunta durante 20 segundos
            showing_classes = True  

# Hilo para manejar la generación de clases
class_thread = threading.Thread(target=generate_random_classes)
class_thread.start()


def guardar_nombre():
    global name
    name = entry.get()
    # Puedes imprimir el nombre para verificar que se guardó
    print(f"Nombre guardado: {name}")
    root.destroy() 

# Crear la ventana principal
root = tk.Tk()
root.title("Bienvenido a Simon AI")

# Etiqueta de bienvenida
label = tk.Label(root, text="¡Bienvenido al juego Simon AI!", font=("Arial", 16))
label.pack(pady=10)

# Etiqueta para el nombre
label_nombre = tk.Label(root, text="Por favor, ingresa tu nombre:", font=("Arial", 12))
label_nombre.pack(pady=5)

# Campo de texto para el nombre
entry = tk.Entry(root, font=("Arial", 12))
entry.pack(pady=5)

# Botón para enviar el nombre
button = tk.Button(root, text="Empezar", command=guardar_nombre, font=("Arial", 12))
button.pack(pady=10)

emotion_log = []
log_interval = 2.5

def log_emotion(emotion, puntaje):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # Formato del tiempo
    emotion_log.append((current_time, emotion, puntaje))  # Guardar en una lista
    print(f"Tiempo: {current_time}, Emoción: {emotion}, Puntaje: {puntaje}")

    # También puedes escribir en un archivo CSV (opcional)
    with open("emotion_log.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([current_time, emotion, puntaje])

def start_logging():
    while True:
        # Asume que 'current_emotion' es la emoción actual y 'puntaje' es el puntaje acumulado
        log_emotion(current_emotion, puntaje)
        time.sleep(log_interval)

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

def process_video_stream_face(source, stream_id):
    global current_classes, showing_classes, start_time, puntaje, current_emotion

    face_model = YOLO("yolov8n-face.pt")
    face_model.to('cuda')
    cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)

    # Obtener el tamaño de la pantalla para calcular la mitad de la resolución
    screen_width = 1920  # Asume una pantalla Full HD (1920x1080)
    screen_height = 1080
    frame_width = screen_width // 2  # Ajusta a la mitad de la pantalla
    frame_height = screen_height  # Ajusta a la mitad de la pantalla

    # Iniciar un hilo para registrar la emoción y puntaje cada 3 segundos
    logging_thread = threading.Thread(target=start_logging, daemon=True)
    logging_thread.start()

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

            # Redimensionar el frame a la mitad del tamaño de la pantalla
            frame = cv2.resize(frame, (frame_width, frame_height))

            # Mostrar el frame actualizado
            cv2.imshow(f"Stream {stream_id}", frame)

            # Salir si se presiona la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def process_video_stream_objects(source, stream_id):
    global counter,current_classes
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
                    cords = box.xyxy[0].tolist()
                    cords = [round(x) for x in cords]
                    conf = round(box.conf[0].item(), 2)
                    if conf > 0.6:
                        if (class_id in current_classes):
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

process_video_stream_face(0,"Face")
#process_video_stream_objects(0, "Objects")