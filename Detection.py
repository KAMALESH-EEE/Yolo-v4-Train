import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='detect.tflite')

# Allocate the tensors
interpreter.allocate_tensors()

# Capture a frame from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

     # Resize the frame to the expected input size of the model
    frame = cv2.resize(frame, (320, 320))
   
    # Preprocess the frame

    frame = np.expand_dims(frame, axis=0)
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']

    
    frame = frame.astype(np.float32) / 255.0

    # Run the model
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], frame)
    interpreter.invoke()

    # Get the output results
    output_details = interpreter.get_output_details()
    output_tensor = interpreter.get_tensor(output_details[0]['index'])
    
    # Draw bounding boxes around the detected objects
    for detection in output_tensor:
        class_id = int(detection[0])
        score = detection[1]
        if score > 0.75:
            bbox = detection[2:6]
            (startX, startY, endX, endY) = (bbox[0] * frame.shape[1], bbox[1] * frame.shape[0], bbox[2] * frame.shape[1], bbox[3] * frame.shape[0])
            cv2.rectangle(frame, (int(startX),int ( startY)), (int(endX), int(endY)), (0, 255, 0), 2)
            cv2.putText(frame, str(class_id), (int(startX),int ( startY)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if class_id==0:
                print("yes")
    
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = np.squeeze(frame, axis=0)
# Display the frame
    cv2.imshow('Frame', frame)

# Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Destroy all windows
cv2.destroyAllWindows()
