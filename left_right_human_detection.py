import cv2
import torch
from pathlib import Path

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    labels = results.pred[0][:, -1].tolist()

    person_indices = [i for i, label in enumerate(labels) if label == 0]

    left_count = 0
    right_count = 0

    center_x = frame.shape[1] // 2

    for idx in person_indices:
        x_center = (results.pred[0][idx][0] + results.pred[0][idx][2]) / 2
        if x_center < center_x:
            left_count += 1
        else:
            right_count += 1

    cv2.putText(frame, f"Left: {left_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Right: {right_count}", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('YOLOv5 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
