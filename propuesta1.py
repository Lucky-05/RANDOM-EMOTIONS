import random
import time

# Lista de colores (o números si prefieres)
colores = ["rojo", "verde", "azul", "amarillo"]

# Secuencia del juego (comienza vacía)
secuencia = []

# Función para mostrar la secuencia al jugador
def mostrar_secuencia(secuencia):
    print("Mira la secuencia:")
    for color in secuencia:
        print(color)
        time.sleep(1)  # Espera un segundo entre cada color
    print("\n" * 10)  # Limpiamos la pantalla con espacios para ocultar la secuencia

# Función para comparar la secuencia del jugador con la del juego
def obtener_intento():
    intento = []
    for i in range(len(secuencia)):
        intento.append(input(f"Color {i+1}: ").lower())
    return intento

# Juego principal
nivel = 1
while True:
    print(f"Nivel {nivel}")
    
    # Agregar un nuevo color aleatorio a la secuencia
    secuencia.append(random.choice(colores))
    
    # Mostrar la secuencia
    mostrar_secuencia(secuencia)
    
    # Pedir al jugador que repita la secuencia
    intento = obtener_intento()
    
    # Verificar si el intento es correcto
    if intento == secuencia:
        print("¡Correcto!")
        nivel += 1  # Subir de nivel
    else:
        print("¡Incorrecto! Has perdido.")
        print(f"Llegaste al nivel {nivel}.")
        break