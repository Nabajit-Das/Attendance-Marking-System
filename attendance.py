import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd
from datetime import datetime, timedelta

# Function to load face data and labels
def load_face_data(parent_folder, target_size=(100, 100)):
    X = []
    y = []

    for person_folder in os.listdir(parent_folder):
        person_path = os.path.join(parent_folder, person_folder)
        if os.path.isdir(person_path):
            for file in os.listdir(person_path):
                img_path = os.path.join(person_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                # Resize the image to a common size
                img = cv2.resize(img, target_size)
                X.append(img.flatten())
                y.append(person_folder)

    return np.array(X), np.array(y)

# Load face data and labels
parent_folder = "all_face_data"
X, y = load_face_data(parent_folder)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train a Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear', C=10.0)
svm_classifier.fit(X, y_encoded)


# Create attendance DataFrame
attendance_df = pd.DataFrame(columns=['Name', 'Time', 'UpdateCount'])

# Dictionary to store the update count for each person
update_counts = {person: 0 for person in label_encoder.classes_}

# Initialize video capture
cap = cv2.VideoCapture(0)

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set the time slots
time_slots = 12
time_interval = 5  # seconds

# Set the total duration for capturing images (1 minute)
total_duration = 60  # seconds

# File to log attendance data
attendance_log_file = "attendance_log.xlsx"

# Capture images for each time slot
for slot in range(time_slots):
    start_time = datetime.now()

    while (datetime.now() - start_time).total_seconds() < total_duration / time_slots:
        # Read frame from video capture
        ret, frame = cap.read()

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # If faces are detected, mark attendance for all persons on the first detection
        if len(faces) > 0:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for (x, y, w, h) in faces:
                # Crop the detected face region
                face_roi = gray[y:y+h, x:x+w]

                # Resize the face image to the target size
                face_resized = cv2.resize(face_roi, (100, 100)).flatten()

                # Predict the label using the SVM classifier
                predicted_label = svm_classifier.predict([face_resized])

                # Convert the predicted label back to the original person's name
                predicted_name = label_encoder.inverse_transform(predicted_label)[0]

                # Increment the update count for the person
                update_counts[predicted_name] += 1

                # Add attendance record for each person
                attendance_df = attendance_df._append({'Name': predicted_name, 'Time': timestamp, 'UpdateCount': update_counts[predicted_name]}, ignore_index=True)

                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display the name of the face
                cv2.putText(frame, predicted_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display frame
            cv2.imshow('Attendance Marking', frame)

        # Wait for the specified time interval
        cv2.waitKey(100)

    # Log attendance data to the Excel file every 5 seconds
    attendance_df.to_excel(attendance_log_file, index=False)

    # Exit if 'Escape' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()

# Create 'final' DataFrame for people with more than 13 updates
final_df = pd.DataFrame(columns=['Name', 'UpdateCount'])
for name, count in update_counts.items():
    if count > 13:
        final_df = final_df._append({'Name': name, 'UpdateCount': count}, ignore_index=True)


# Save 'final' DataFrame to 'final.xlsx' file
final_file = "final_attendance.xlsx"
final_df.to_excel(final_file, index=False)