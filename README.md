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

The repository is organized into several key files:
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
