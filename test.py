import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)

    if cap.isOpened():
        print(f"Camera {i} is available")

        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(2000)   # Show for 2 seconds
            cv2.destroyAllWindows()

        cap.release()