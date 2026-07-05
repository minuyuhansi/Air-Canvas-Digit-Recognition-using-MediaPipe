import cv2
import numpy as np
import mediapipe as mp
import time
import math

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

# Kept free of strict type 
def print_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # Thumb
    (0,5),(5,6),(6,7),(7,8),         # Index
    (5,9),(9,10),(10,11),(11,12),    # Middle
    (9,13),(13,14),(14,15),(15,16),  # Ring
    (13,17),(17,18),(18,19),(19,20), # Pinky
    (0,17)                           # Palm
]

# Load the 0-9 Digit Recognition Model using OpenCV 
try:
    net = cv2.dnn.readNetFromONNX('mnist.onnx')
except Exception as e:
    print("Error: Could not find 'mnist.onnx'. Make sure it's in this folder!")
    exit()

drawing_points = []
webCam = cv2.VideoCapture(1)

webCam.set(cv2.CAP_PROP_FRAME_WIDTH,1200)
webCam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_counter = 0
MOVEMENT_THRESHOLD = 6 
detected_number = "Waiting..."
should_predict = False

with GestureRecognizer.create_from_options(options) as recognizer:
    print("Air Canvas Active! Point to draw, pinch index & middle fingers to move without drawing.")
    
    while webCam.isOpened():
        success, frame = webCam.read()
        if not success: 
            continue

        # frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        canvas = np.zeros((h, w), dtype=np.uint8)

        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)

        frame_counter += 1
        recognizer.recognize_async(mp_image, frame_counter)

        if latest_result is not None:
            gesture_name = "None"
            if latest_result.gestures and len(latest_result.gestures) > 0:
                gesture_name = latest_result.gestures[0][0].category_name
            
            cv2.putText(frame, f"Gesture: {gesture_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (163, 35, 3), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Predicted Digit: {detected_number}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

            if gesture_name == "Open_Palm":
                drawing_points.clear()
                detected_number = "Waiting..."

            if latest_result.hand_landmarks and len(latest_result.hand_landmarks) > 0:
                hand = latest_result.hand_landmarks[0]
                
                # Draw skeleton
                for lm in hand:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
                for start, end in HAND_CONNECTIONS:
                    cv2.line(frame, (int(hand[start].x * w), int(hand[start].y * h)), (int(hand[end].x * w), int(hand[end].y * h)), (255, 0, 0), 1)

                # Get finger positions
                ix, iy = int(hand[8].x * w), int(hand[8].y * h)
                mx, my = int(hand[12].x * w), int(hand[12].y * h)
                
                # Calculate physical finger gap space
                finger_distance = math.sqrt((ix - mx)**2 + (iy - my)**2)

                if finger_distance > 40 and (gesture_name == "Pointing_Up" or iy < my - 20):
                    detected_number = "Analyzing..."
                    cv2.circle(frame, (ix, iy), 6, (0, 255, 255), -1) 
                    should_predict = True 
                    
                    if len(drawing_points) > 0 and drawing_points[-1] is not None:
                        last_x, last_y = drawing_points[-1]
                        if math.sqrt((ix - last_x)**2 + (iy - last_y)**2) > MOVEMENT_THRESHOLD:
                            drawing_points.append((ix, iy))
                    else:
                        drawing_points.append((ix, iy))
                else:
                    if len(drawing_points) > 0 and drawing_points[-1] is not None:
                        drawing_points.append(None)
            else:
                if len(drawing_points) > 0 and drawing_points[-1] is not None:
                    drawing_points.append(None)

        for i in range(1, len(drawing_points)):
            if drawing_points[i - 1] is None or drawing_points[i] is None: 
                continue
            cv2.line(frame, drawing_points[i - 1], drawing_points[i], (7, 0, 219), 6)
            cv2.line(canvas, drawing_points[i - 1], drawing_points[i], 255, 12)

        # RUN 0-9 DIGIT PREDICTION
        if should_predict and (len(drawing_points) == 0 or drawing_points[-1] is None):
            valid_pts = [p for p in drawing_points if p is not None]
            if len(valid_pts) > 10:
                xs = [p[0] for p in valid_pts]
                ys = [p[1] for p in valid_pts]
                x, y, wid, hei = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
                
                pad = 20
                x_start, y_start = max(0, x - pad), max(0, y - pad)
                x_end, y_end = min(w, x + wid + pad), min(h, y + hei + pad)
                
                if (x_end - x_start) > 10 and (y_end - y_start) > 10:
                    roi = canvas[y_start:y_end, x_start:x_end]
                    roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                    
                    # Package array into MNIST form tensor
                    blob = cv2.dnn.blobFromImage(roi_resized, 1.0/255.0, (28, 28), 0, swapRB=False, crop=False)
                    net.setInput(blob)
                    predictions = net.forward()
                    
                    detected_number = str(np.argmax(predictions))
            
            should_predict = False

        cv2.imshow('Air Canvas - 0-9 Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

webCam.release()
cv2.destroyAllWindows()