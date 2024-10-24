from ultralytics import YOLO
import cv2

BEST_MODEL_PATH = "./runs/detect/train52/weights/best.pt"

# Load your trained YOLO model
model = YOLO(BEST_MODEL_PATH)

# Open a connection to the webcam (0 for the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Make predictions using the YOLO model
    results = model(frame)

    # YOLO returns a list of results, we need to access each result
    for result in results:
        # Annotate the frame with the predictions
        annotated_frame = result.plot()  # Draw the results on the frame

    # Display the frame
    cv2.imshow("YOLO v11 Predictions", annotated_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
