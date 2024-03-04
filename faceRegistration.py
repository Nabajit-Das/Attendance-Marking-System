import cv2
import os

# Function to create a folder if it doesn't exist
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Function to collect face data
def collect_face_data(parent_folder, person_name, num_samples=100):
    # Create a folder for each person's face data within the parent folder
    person_folder = os.path.join(parent_folder, person_name)
    create_folder(person_folder)

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Counter for collected samples
    sample_count = 0

    while sample_count < num_samples:
        # Read frame from video capture
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Crop the detected face region
            face_roi = frame[y:y+h, x:x+w]

            # Save the face image in the person's folder
            face_filename = os.path.join(person_folder, f"{person_name}_{sample_count}.jpg")
            cv2.imwrite(face_filename, face_roi)

            # Display the collected sample
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('Collecting Face Data', frame)

            # Increment the sample count
            sample_count += 1

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Specify the parent folder for face data
parent_folder = "all_face_data"

# Specify the person's name
person_name = "nitin"

# Collect face data for the specified person (adjust num_samples as needed)
collect_face_data(parent_folder, person_name, num_samples=100)
