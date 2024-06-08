## Face Detection and Recognition

This project implements a real-time face detection and recognition system using OpenCV and TensorFlow's Keras. It utilizes a pre-trained deep learning model to identify faces captured from a webcam feed.

### Features

- **Real-time Face Detection**: Uses OpenCV's Haar Cascade Classifier to detect faces in video frames.
- **Face Recognition**: Classifies detected faces using a trained Keras model.
- **Probability Display**: Shows the prediction probability along with the recognized face's name.
- **Simple and Intuitive UI**: Displays real-time video feed with annotations.

### Requirements

- Python 3.9
- TensorFlow
- OpenCV
- NumPy

### Usage

1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/yourusername/face-detection-recognition.git](https://github.com/crazyNerrd/Face-Detection.git)
   cd face-detection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python face_recognition.py
   ```

### Files

- `face_recognition.py`: Main script for running the face detection and recognition.
- `haarcascade_frontalface_default.xml`: Haar Cascade file for face detection.
- `keras_model.h5`: Pre-trained Keras model for face recognition.

### License

This project is licensed under the MIT License.

---

This description provides an overview of the project, lists the main features, outlines the requirements, and includes basic usage instructions.
