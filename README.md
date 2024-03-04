


# Attendance Marking System

Attendance Marking System is a facial recognition-based solution for automating attendance tracking. This system utilizes OpenCV, NumPy, and pandas for face detection, recognition, and attendance logging.

## Overview

The system captures video from a camera, detects faces, and matches them to pre-trained images to mark attendance. The attendance data, including the person's name, timestamp, and update count, is logged into an Excel file.

## Features

- Face detection and recognition
- Attendance logging
- Update count for each individual
- Real-time face matching and marking

## Prerequisites

- Python 3.x
- OpenCV
- NumPy
- pandas

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Nabajit-Das/Attendance-Marking-System.git
   cd Attendance-Marking-System
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare face data: Organize individual face images into folders.
2. Train the face recognition model:

   ```bash
   python train_model.py
   ```

3. Run the attendance marking system:

   ```bash
   python attendance_marking_system.py
   ```

4. The system will start capturing video, recognizing faces, and logging attendance.

## Contributing

Contributions are welcome! If you encounter issues or have improvements, feel free to open an issue or create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

You can copy and paste this content into your README.md file. Feel free to modify it based on your specific needs.
