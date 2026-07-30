import cv2

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Webcam started! Press Q to quit.")

while True:
    # Read frame
    ret, frame = cap.read()

    # If frame not captured
    if not ret:
        print("Failed to grab frame")
        break

    # Show the frame
    cv2.imshow("My Webcam", frame)

    # Wait for key press (IMPORTANT FIX)
    key = cv2.waitKey(10) & 0xFF

    # Press 'q' to exit
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()