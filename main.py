import cv2
from ultralytics import YOLO


def main():
    print("Hello from object-detection!")
    # Load YOLOv8 model (this downloads the model if not already present)
    model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' or 'yolov8m.pt' for better accuracy

    # Open the webcam
    cap = cv2.VideoCapture(0)  # '0' is usually your default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run YOLO detection
        results = model(frame)

        # Draw detection results
        annotated_frame = results[0].plot()

        # Show the output
        cv2.imshow("Live YOLOv8 Object Detection", annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
