# EmoTrack - Real-Time Emotion Detection System

EmoTrack is an AI-powered real-time facial emotion detection system designed to identify and display emotional states such as happiness, sadness, anger, surprise, and more using deep learning and computer vision. Built with Python and OpenCV, the system utilizes a CNN-based model to detect and classify human emotions from live webcam feed.

## Features

- Real-time emotion detection through webcam
- Accurate facial recognition using Haar Cascade classifiers
- Emotion classification using a trained CNN model
- Easy-to-use and interactive interface
- Lightweight and fast performance

## Demo

![Demo GIF or Screenshot](link-to-demo.gif or screenshot)

## Technologies Used

- Python
- OpenCV
- Keras / TensorFlow
- NumPy
- Haar Cascade Classifier
- Convolutional Neural Networks (CNN)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/emotrack.git
   cd emotrack
Create a virtual environment (optional but recommended)

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the application

bash
Copy
Edit
python emotrack.py
Project Structure
bash
Copy
Edit
emotrack/
│
├── dataset/                 # Optional: for training data
├── model/                   # Trained model (.h5)
├── haarcascades/            # Haar cascade files for face detection
├── emotrack.py              # Main script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
How It Works
EmoTrack uses Haar cascades to detect faces from webcam input.

The detected face is cropped and preprocessed.

A pre-trained CNN model predicts the emotion from the facial features.

The emotion is displayed on the screen in real-time.

Emotions Detected
Happy

Sad

Angry

Neutral

Surprise

Fear

Disgust

Contributions
Feel free to fork the project and contribute via pull requests! Suggestions and improvements are welcome.

License
This project is licensed under the MIT License.

Author
Hajra Abid
Gold Medalist | AI & ML Developer
