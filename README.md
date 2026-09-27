# 🤟 Sign Language Recognition Using Machine Learning

A machine learning and computer vision project that recognizes **sign language hand gestures** using a Convolutional Neural Network (CNN). The project uses image-based hand gesture data from the **Sign Language MNIST** dataset and includes a camera interface for capturing and processing hand gestures.

## 📌 Project Overview

Sign language provides an important means of communication for people with hearing and speech disabilities. This project explores how **computer vision and deep learning** can be used to recognize hand gestures and map them to corresponding alphabet characters.

The system processes hand gesture images, trains a CNN model, and predicts the corresponding sign language character.

The project consists of:

* 📷 Camera-based hand gesture capture
* 🖐️ Hand segmentation and image preprocessing
* 🧠 CNN-based image classification
* 📊 Model training and validation
* 🔤 Conversion of predicted classes into alphabet characters
* ☁️ Google Colab-based model training

## ✨ Features

* Real-time camera interface for capturing hand gestures
* Background subtraction for hand segmentation
* Region of Interest (ROI) based gesture detection
* Image preprocessing using OpenCV
* CNN-based classification
* Sign Language MNIST dataset support
* 28 × 28 grayscale image processing
* Alphabet prediction from trained classes
* Model training using TensorFlow/Keras
* Training and validation accuracy visualization

## 🛠️ Technologies Used

| Technology             | Purpose                               |
| ---------------------- | ------------------------------------- |
| **Python**             | Programming language                  |
| **OpenCV**             | Computer vision and camera processing |
| **NumPy**              | Numerical and image-data processing   |
| **Pandas**             | Dataset handling                      |
| **Matplotlib**         | Visualization                         |
| **Seaborn**            | Dataset visualization                 |
| **Scikit-learn**       | Data preprocessing and evaluation     |
| **TensorFlow / Keras** | CNN model development                 |
| **Google Colab**       | Model training                        |

## 🧠 Machine Learning Model

The project uses a **Convolutional Neural Network (CNN)** for sign language classification.

The model architecture contains:

```text
Input Image
    ↓
Conv2D (64 filters)
    ↓
MaxPooling2D
    ↓
Conv2D (64 filters)
    ↓
MaxPooling2D
    ↓
Conv2D (64 filters)
    ↓
MaxPooling2D
    ↓
Flatten
    ↓
Dense (128 neurons)
    ↓
Dropout (20%)
    ↓
Softmax Output
```

The input images are resized/represented as:

```text
28 × 28 × 1
```

The model uses:

* **ReLU** activation for convolutional and dense layers
* **Softmax** activation for classification
* **Categorical Cross-Entropy** as the loss function
* **Adam** optimizer
* **Accuracy** as the evaluation metric

The current training code uses a batch size of `128` and defines `24` output classes.

## 📂 Project Structure

```text
Sign-Language-Recognition/
│
├── Camera Interface
│   └── Camera interface and hand segmentation code
│
├── Dataset Link
│   └── Link to Sign Language MNIST dataset
│
├── Google Colab Code
│   └── CNN training and prediction code
│
└── README.md
```

The repository currently contains these main components in the `main` branch.

## 📊 Dataset

This project uses the **Sign Language MNIST** dataset.

The dataset contains grayscale images of hand gestures represented as numerical pixel values. The images are processed into `28 × 28` matrices before being supplied to the CNN.

Dataset:

[Sign Language MNIST – Kaggle](https://www.kaggle.com/datamunge/sign-language-mnist)

The dataset link is also included directly in the repository.

## 🔄 Workflow

```text
Sign Language Dataset
        ↓
Data Loading
        ↓
Image Preprocessing
        ↓
28 × 28 Image Conversion
        ↓
Normalization
        ↓
Train / Test Split
        ↓
CNN Model
        ↓
Model Training
        ↓
Prediction
        ↓
Alphabet Character
```

### Camera Workflow

The camera interface uses OpenCV to capture frames from the webcam. It defines a Region of Interest and performs background-based hand segmentation before displaying the processed hand image.

```text
Webcam
   ↓
Capture Frame
   ↓
Flip / Preprocess
   ↓
Region of Interest
   ↓
Background Subtraction
   ↓
Hand Segmentation
   ↓
Processed Gesture
   ↓
CNN Prediction
   ↓
Recognized Character
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Uday-Sekhar-Raju-Biruduraju/Sign-Language-Recognition.git
```

### 2. Navigate to the Project

```bash
cd Sign-Language-Recognition
```

### 3. Install Required Libraries

Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn tensorflow
```

> **Note:** The exact package versions may depend on your Python and TensorFlow environment.

## ☁️ Running the Model in Google Colab

The repository contains a `Google Colab Code` file containing the model-training workflow.

The training process:

1. Loads the Sign Language MNIST training and testing CSV files.
2. Separates labels from image pixels.
3. Converts pixel data into `28 × 28` images.
4. Normalizes pixel values.
5. Splits the training data.
6. Builds the CNN.
7. Trains the model.
8. Evaluates predictions.
9. Maps numerical classes to alphabet characters.

The current notebook code saves the trained model as:

```text
sign_mnist_cnn_50_Epochs.h5
```

The repository's training code demonstrates this workflow using TensorFlow/Keras.

## 📷 Camera Interface

The camera interface uses:

```python
import cv2
import numpy as np
```

A webcam is accessed using:

```python
cam = cv2.VideoCapture(0)
```

The system establishes a background model and uses frame differences to isolate the hand region.

### Region of Interest

The camera interface defines a specific ROI where the user places their hand:

```text
┌──────────────────────────────┐
│                              │
│       Camera Feed            │
│                              │
│       ┌────────────┐         │
│       │    HAND    │         │
│       │     ✋      │         │
│       └────────────┘         │
│                              │
└──────────────────────────────┘
```

Keeping the hand inside the designated region helps the preprocessing stage isolate the gesture.

## 🔤 Recognized Classes

The current model maps its 24 output classes to alphabet characters:

```text
A B C D E F G H I K L M
N O P Q R S T U V W X Y
```

The class mapping is defined in the Google Colab code.

> Note: The Sign Language MNIST alphabet classes do not represent every letter of the English alphabet in this implementation.

## 📈 Model Evaluation

The project evaluates the trained model using classification accuracy.

Training and validation accuracy are also plotted to help visualize the learning process:

```text
Accuracy
   │
   │       ╭────────────
   │     ╭─╯
   │   ╭─╯
   │ ╭─╯
   └──────────────────────
        Epochs
```

The training code uses a validation split during model training and evaluates predictions against the test dataset.

## 🎯 Applications

This project can serve as a foundation for:

* Sign language learning applications
* Gesture-controlled interfaces
* Accessibility tools
* Human-computer interaction
* Educational applications
* Computer vision research
* Real-time gesture recognition systems

## 🔮 Future Improvements

Possible improvements include:

* [ ] Real-time CNN prediction directly from webcam input
* [ ] Improve hand detection and segmentation
* [ ] Add more sign language classes
* [ ] Support continuous gesture recognition
* [ ] Convert recognized signs into words and sentences
* [ ] Add text-to-speech output
* [ ] Build a web-based interface
* [ ] Deploy the model as an API
* [ ] Improve model accuracy using data augmentation
* [ ] Experiment with transfer learning
* [ ] Add support for Indian Sign Language (ISL)

## ⚠️ Limitations

The current implementation is primarily an **image classification** system. Sign languages can involve hand movement, orientation, facial expressions, and temporal information, which cannot always be represented by a single static image.

Therefore, this project should be considered a machine-learning prototype for gesture classification rather than a complete sign-language translation system.

## 👨‍💻 Author

**Uday Sekhar Raju Biruduraju**

GitHub:
https://github.com/Uday-Sekhar-Raju-Biruduraju

## 📜 License

This project is intended for **educational and academic purposes**.

---

⭐ If you find this project useful, consider giving the repository a star!

**Repository:**
https://github.com/Uday-Sekhar-Raju-Biruduraju/Sign-Language-Recognition
