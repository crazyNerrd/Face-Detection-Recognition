---

# Face Detection and Recognition

This project implements a real-time face detection and recognition system using OpenCV and TensorFlow's Keras. The project consists of two main scripts: one for capturing face images from a webcam and saving them into a directory, and another for recognizing faces from the saved images using a pre-trained Keras model.

## Features

- **Real-time Face Detection**: Uses OpenCV's Haar Cascade Classifier to detect faces.
- **Face Recognition**: Recognizes detected faces using a trained Keras model.
- **Image Capture**: Captures images from a webcam and stores them in a structured directory.
- **Probability Display**: Shows the prediction probability along with the recognized face's name.
- **Simple and Intuitive UI**: Displays video feed with annotations and handles user input for capturing and recognizing faces.

## Requirements

- Python 3.9
- TensorFlow
- OpenCV
- NumPy

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/face-detection-recognition.git
   cd face-detection-recognition
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Capturing Images from Webcam

Run the script to capture images from the webcam and save them into a specified directory:

```bash
python datacollect.py
```

### 2. Recognizing Faces from Captured Images

Run the script to load images from the directory and recognize faces using the Keras model:

```bash
python recognize_faces.py
```

### License

This project is licensed under the MIT License.

---

Provides a clear and detailed overview of the project, its features, requirements, installation steps, usage instructions, and script details. This should help users understand and utilize your project effectively.
