# Arabic OCR System

## Overview

This project implements a complete Optical Character Recognition (OCR) system for converting images of typed Arabic text into machine-readable text. Due to the cursive and complex nature of the Arabic script, this system uses a multi-stage pipeline that includes image preprocessing, text segmentation (lines, words, and characters), and character recognition using a custom-trained neural network.

The core of the project is a Jupyter Notebook (`character recognition.ipynb`) that details the entire process, supported by various Python scripts for modular execution.

---

## Table of Contents

- [Features](#features)
- [System Pipeline](#system-pipeline)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Troubleshooting](#troubleshooting)

---

## Features

- **End-to-End OCR:** Full pipeline from image input to text output.
- **Advanced Segmentation:** Implements algorithms for segmenting images into lines, words, and individual characters, which is critical for Arabic script.
- **Custom Neural Network:** Uses a Keras-based neural network trained specifically on an Arabic character dataset (`arabic.csv`).
- **Modular Design:** The project is broken down into logical Python scripts for preprocessing, segmentation, model building, and testing.
- **Flask Integration:** Includes a script to serve the OCR model via a Flask web server for API-based access.

---

## System Pipeline

The OCR process is executed in several sequential stages[1]:
1.  **Preprocessing:** The input image is converted to grayscale, binarized using thresholding, and de-skewed to align the text horizontally.
2.  **Line Segmentation:** The preprocessed image is analyzed to identify and separate individual lines of text.
3.  **Word Segmentation:** Each line is further processed to isolate individual words or connected components.
4.  **Character Segmentation:** This is the most complex step, where each word is segmented into its constituent characters.
5.  **Character Recognition:** A trained neural network model predicts the character corresponding to each segmented image patch.
6.  **Text Combination:** The recognized characters are combined to reconstruct the original words and sentences.

---

## Dataset

- **Character Data:** `arabic.csv`
    - This file contains the dataset used for training the neural network. It likely consists of flattened pixel values for images of individual Arabic characters and their corresponding labels.

---

## Project Structure

The repository is organized into several key files[2]:
- `character recognition.ipynb`: The main Jupyter Notebook that showcases the entire pipeline, from data loading to final text recognition.
- `AOCR.py`: A main script that likely integrates the entire OCR pipeline.
- `pre_processing.py`: Contains functions for image cleaning and preparation.
- `segmentation.py`, `segmentation_algorithms.py`, `segmentation_character.py`: These files handle the logic for segmenting the image into lines, words, and characters.
- `build_NN_model.py`: Script to define, train, and save the character recognition neural network.
- `load_NN.py`: Utility to load the pre-trained neural network model for inference.
- `test_NN_model.py`: Script to evaluate the trained model's performance.
- `run_keras_server.py`: A Flask application to expose the OCR model as a web service.
- `requirements.txt`: A list of required Python packages.

---

## Setup and Installation

### Prerequisites
- Python 3.6+
- `pip` package manager

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nada-mossad/Arabic_OCR.git
    cd Arabic_OCR
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install libraries such as TensorFlow/Keras, OpenCV, scikit-learn, Flask, and others.

---

## Usage

There are a few ways to use this project, depending on your goal.

### 1. Explore the Full Pipeline in the Notebook
The best way to understand the system is to run the `character recognition.ipynb` notebook.
```bash
jupyter notebook "character recognition.ipynb"
```
This notebook provides a step-by-step walkthrough with code, explanations, and visual outputs for each stage of the OCR process.

### 2. Run the Main OCR Script
To process an image from the command line, use the main OCR script. Place your test image in the root directory.
```bash
python AOCR.py --image path/to/your/image.png
```
This script will execute the full pipeline and print the recognized text to the console.

### 3. Train the Model
If you want to retrain the neural network (e.g., with a different dataset or architecture):
```bash
python build_NN_model.py
```
This will use `arabic.csv` to train the model and save it as a file (e.g., `ocr_model.h5`).

### 4. Run as a Web Service
To serve the OCR model via a web API:
```bash
python run_keras_server.py
```
This will start a Flask server, typically on `http://localhost:5000`. You can then send image data to its endpoint to get back the recognized text.

---

## Implementation Details

- **Image Processing:** The system heavily relies on **OpenCV** for image manipulation tasks like grayscaling, thresholding, finding contours, and geometric transformations.
- **Segmentation Logic:** The segmentation scripts use techniques like horizontal and vertical histogram projections to find separators between lines and words. Character segmentation is more advanced, potentially using contour analysis or connected components analysis.
- **Neural Network:** The character recognition model is a **Convolutional Neural Network (CNN)** or a standard **Artificial Neural Network (ANN)** built with **Keras**. It is trained to classify an input image of a character into one of the 29 possible Arabic letters.
- **Web Service:** The `run_keras_server.py` script uses **Flask** to create a simple REST API, allowing the OCR functionality to be integrated into other applications.

---

## Troubleshooting

- **Dependency Errors:** Ensure all packages from `requirements.txt` are installed correctly in your virtual environment. Errors related to TensorFlow or OpenCV are common; make sure you have compatible versions.
- **Model Not Found:** If you are running inference scripts, ensure a pre-trained model file (e.g., `ocr_model.h5`) exists. You may need to run `build_NN_model.py` first to generate it.
- **Poor Accuracy:** The performance of an OCR system is highly dependent on the quality of the input image and the dataset it was trained on. For best results, use clear, high-contrast, and properly aligned images similar to the training data.

---

