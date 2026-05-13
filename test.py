import cv2
from ultralytics import YOLO

model = YOLO("weights/yolov8s.pt")
cap = cv2.VideoCapture(0)
for _ in range(10): cap.read()

while True:
    ret, frame = cap.read()
    if not ret: break
    
    results = model(frame, verbose=False, conf=0.40)
    labels = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
    print(labels)
    
    cv2.imshow("feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()