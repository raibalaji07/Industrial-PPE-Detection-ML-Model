from ultralytics import YOLO
import cv2
import numpy as np
from playsound import playsound
import pygame

pygame.mixer.init()
# alert_sound = pygame.mixer.Sound('awaz.wav')

person_model = YOLO("yolov8l.pt")
equipment_model = YOLO("/Users/balajimac/Desktop/Project/IPPE/runs/detect/train/weights/best.pt ")

cap = cv2.VideoCapture('/Users/balajimac/Desktop/Project/IPPE/UncleAuntySafety.mp4')
class_name = ['Helmet', 'Goggles', 'Jacket', 'Gloves', 'Footwear']


def detect_objects(model, frame, conf_threshold=0.65):
    results = model.predict(frame, conf=conf_threshold, save=False)
    boxes = []
    scores = []
    classes = []

    for result in results:
        boxes.extend(result.boxes.xyxy.cpu().numpy())
        scores.extend(result.boxes.conf.cpu().numpy())
        classes.extend(result.boxes.cls.cpu().numpy())

    return np.array(boxes), np.array(scores), np.array(classes)


def adjust_box(box, reduction_factor=0.2):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    x1_new = x1 + reduction_factor * width / 2
    y1_new = y1 + reduction_factor * height / 2
    x2_new = x2 - reduction_factor * width / 2
    y2_new = y2 - reduction_factor * height / 2

    return int(x1_new), int(y1_new), int(x2_new), int(y2_new)


w = 1000
h = 800
# alert_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (w, h))

    person_boxes, _, _ = detect_objects(person_model, frame, conf_threshold=0.65)

    equipment_boxes, equipment_scores, equipment_classes = detect_objects(equipment_model, frame, conf_threshold=0.65)

    person_boxes = person_boxes.astype(int)
    safety_equipment_boxes = [(x1, y1, x2, y2, class_name[int(cls)]) for (x1, y1, x2, y2), cls in
                              zip(equipment_boxes.astype(int), equipment_classes)]
    safety_equipment_scores = [score for score, cls in zip(equipment_scores, equipment_classes)]

    for (px1, py1, px2, py2) in person_boxes:
        worn = {'Helmet': False, 'Jacket': False}
        missing_item = []

        for (ex1, ey1, ex2, ey2, equipment), score in zip(safety_equipment_boxes, safety_equipment_scores):
            if equipment in worn:
                ex1, ey1, ex2, ey2 = adjust_box((ex1, ey1, ex2, ey2), reduction_factor=0.5)

            if (px1 <= ex1 <= px2 and py1 <= ey1 <= py2 and px1 <= ex2 <= px2 and py1 <= ey2 <= py2):
                if equipment in worn:
                    worn[equipment] = True

        safe = worn['Helmet'] and worn["Jacket"]
        warning_label = "SAFE" if safe else "NOT SAFE"
        color = (0, 255, 0) if safe else (0, 0, 255)

        if not safe:
            if not worn["Helmet"]:
                missing_item.append("Helmet")
            if not worn["Jacket"]:
                missing_item.append("Jacket")
            missing_text = " , ".join(missing_item)

            cv2.putText(frame, f"Missing: {missing_text}", (px1, py1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, warning_label, (px1, py1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 3)

    # if not safe and not alert_playing:
    #     alert_sound.play()
    #     alert_playing = True
    # elif safe and alert_playing:
    #     alert_sound.stop()
    #     alert_playing = False

    for (x1, y1, x2, y2, equipment), score in zip(safety_equipment_boxes, safety_equipment_scores):
        label = f"{equipment}: {score * 100:.2f}%"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("video", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
