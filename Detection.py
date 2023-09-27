import cv2
import numpy as np

net = cv2.dnn.readNet("yolov4-tiny-custom_best1.weights", "yolov4-tiny-custom1.cfg")
cap=cv2.VideoCapture(0)


classes =['Helthy','Defect']


while True:
    _,image = cap.read() 
    height, width = image.shape[:2]


    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)


    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.85: 
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                color = (0, 255, 0) 
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                print("YES")
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("YOLOv4-Tiny Object Detection", image)
    cv2.waitKey(1)
cv2.destroyAllWindows()
