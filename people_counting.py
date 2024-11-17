import cv2
import numpy as np
from ultralytics import YOLO
import cvzone

# Cargar el modelo YOLOv8
model = YOLO('yolov8s.pt')

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)  # 0 para la cámara por defecto

# Obtener las dimensiones del cuadro de video
ret, frame = cap.read()
height, width, _ = frame.shape

# Definir la posición de la línea central (en el centro horizontal)
line_position = width // 2

# Variables para el conteo
count_left_to_right = 0
count_right_to_left = 0
tracked_objects = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar la detección con YOLOv8
    results = model(frame)

    # Extraer las cajas delimitadoras
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            if cls == 0 and conf > 0.5:  # Clase 0 corresponde a 'persona'
                detections.append((x1, y1, x2, y2))

    # Dibujar la línea central (vertical)
    cv2.line(frame, (line_position, 0), (line_position, height), (0, 255, 0), 2)

    # Procesar cada detección
    new_tracked_objects = {}
    for (x1, y1, x2, y2) in detections:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Encontrar objetos ya rastreados cercanos
        found = False
        for obj_id, (prev_cx, prev_cy) in tracked_objects.items():
            if abs(cx - prev_cx) < 50 and abs(cy - prev_cy) < 50:  # Considerar cercano
                new_tracked_objects[obj_id] = (cx, cy)
                found = True

                # Detectar cruce de la línea
                if prev_cx < line_position and cx >= line_position:  # Izquierda a derecha
                    count_left_to_right += 1
                elif prev_cx > line_position and cx <= line_position:  # Derecha a izquierda
                    count_right_to_left += 1
                break

        if not found:
            # Crear un nuevo ID para el objeto
            new_id = len(tracked_objects) + 1
            new_tracked_objects[new_id] = (cx, cy)

        # Dibujar la caja delimitadora y el centroide
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Actualizar los objetos rastreados
    tracked_objects = new_tracked_objects

    # Mostrar el conteo en la pantalla
    cv2.putText(frame, f'Left to Right: {count_left_to_right}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Right to Left: {count_right_to_left}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Mostrar el cuadro de video
    cv2.imshow('People Counting', frame)

    # Verificar si la ventana fue cerrada con la "X"
    if cv2.getWindowProperty('People Counting', cv2.WND_PROP_VISIBLE) < 1:
        break

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Guardar los resultados en el archivo .txt
with open("people_count.txt", "w") as f:
    f.write(f"Total Left to Right: {count_left_to_right}\n")
    f.write(f"Total Right to Left: {count_right_to_left}\n")
    f.write(f"Total Count: {count_left_to_right + count_right_to_left}\n")

# Liberar recursos
cap.release()
cv2.destroyAllWindows()