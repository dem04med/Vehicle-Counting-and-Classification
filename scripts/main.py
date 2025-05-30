import sys
import os
import torch
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deep_sort.deep_sort import DeepSort

# Caminhos do modelo e vídeo
model_path = r'C:\YOLO\Vehicle-Counting-and-Classification\yolov5\runs\train\exp2\weights\best.pt'
video_path = r'C:\YOLO\Vehicle-Counting-and-Classification\input_videos\video.mp4'
output_path = r'C:\YOLO\Vehicle-Counting-and-Classification\output\video1.mp4'
counts_file = r'C:\YOLO\Vehicle-Counting-and-Classification\output\vehicle_counts.txt'

# Carregar modelo YOLOv5
model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')
model.conf = 0.6  # Limiar de confiança

# Inicializar Deep SORT
deepsort = DeepSort(r"C:\YOLO\Vehicle-Counting-and-Classification\deep_sort\deep_sort\deep\checkpoint\ckpt.t7",
                    max_age=100, n_init=2, nn_budget=100)

# Abrir vídeo
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[ERRO] Não foi possível abrir o vídeo: {video_path}")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Criar pasta de output
os.makedirs(os.path.dirname(output_path), exist_ok=True)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Dicionários auxiliares
vehicle_counts = {}
counted_ids = set()
track_id_to_class = {}
class_confidences = {}

# Cores por classe
class_colors = {
    'car': (0, 255, 0),
    'truck': (0, 128, 255),
    'bus': (255, 0, 0),
    'motorcycle': (255, 0, 255),
    'bicycle': (0, 255, 255),
}
counter_text_color = (255, 0, 255)

# Função de fallback para classe por IoU
def find_class_for_box(tracking_box, bbox_xywh_list, class_list, iou_threshold=0.3):
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0

    bbox_converted = []
    for box in bbox_xywh_list:
        x_c, y_c, w, h = box
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        bbox_converted.append([x1, y1, x2, y2])

    best_iou = 0
    best_idx = None
    for i, box in enumerate(bbox_converted):
        current_iou = iou(tracking_box, box)
        if current_iou > best_iou and current_iou > iou_threshold:
            best_iou = current_iou
            best_idx = i
    return class_list[best_idx] if best_idx is not None else None

# Loop principal
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()

        if len(detections) == 0:
            out.write(frame)
            cv2.imshow("Vehicle Tracking", frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break
            continue

        bbox_xywh = []
        confidences = []
        classes = []

        for *box, conf, cls in detections:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            x_c = x1 + w / 2
            y_c = y1 + h / 2
            bbox_xywh.append([x_c, y_c, w, h])
            confidences.append(conf)
            classes.append(int(cls))

        bbox_xywh = np.array(bbox_xywh)
        confidences = np.array(confidences)

        outputs, _ = deepsort.update(bbox_xywh, confidences, classes, frame)

        for output in outputs:
            if len(output) == 6:
                x1, y1, x2, y2, track_id, class_id = output
            elif len(output) == 5:
                x1, y1, x2, y2, track_id = output
                class_id = None
            else:
                continue

            track_id = int(track_id)
            if class_id is not None:
                class_id = int(class_id)

            if class_id is None:
                class_id = find_class_for_box([x1, y1, x2, y2], bbox_xywh, classes)

            if class_id is not None:
                class_name = model.names[class_id]
            else:
                class_name = "unknown"

            track_id_to_class[track_id] = class_name

            if track_id not in counted_ids and class_name != "unknown":
                counted_ids.add(track_id)
                vehicle_counts[class_name] = vehicle_counts.get(class_name, 0) + 1

            # Registar confiança média por classe
            if class_name not in class_confidences:
                class_confidences[class_name] = []
            if class_id is not None and class_id in classes:
                idx = classes.index(class_id)
                conf = confidences[idx]
                class_confidences[class_name].append(float(conf))
                label = f"ID:{track_id} {class_name} ({conf:.2f})"
            else:
                label = f"ID:{track_id} {class_name}"

            color = class_colors.get(class_name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Overlay com contagens
        start_y = 10
        line_height = 25
        for i, (cls, cnt) in enumerate(vehicle_counts.items()):
            y = start_y + (i + 1) * line_height
            text = f"{cls}: {cnt}"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, counter_text_color, 2)

        out.write(frame)
        cv2.imshow("Vehicle Tracking", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"[ERRO] Exceção durante o processamento: {e}")

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# =====================
# Estatísticas finais
# =====================
total_count = sum(vehicle_counts.values())

with open(counts_file, 'w', encoding='utf-8') as f:
    f.write("=== Estatísticas de Contagem de Veículos ===\n\n")
    for cls, count in vehicle_counts.items():
        percent = (count / total_count) * 100 if total_count > 0 else 0
        confs = class_confidences.get(cls, [])
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        f.write(f"Classe: {cls}\n")
        f.write(f" - Contagem: {count}\n")
        f.write(f" - Percentagem: {percent:.2f}%\n")
        f.write(f" - Confiança média: {avg_conf:.2f}\n\n")

    f.write(f"TOTAL DETETADO: {total_count}\n")

print(f"[INFO] Estatísticas gravadas em: {counts_file}")
