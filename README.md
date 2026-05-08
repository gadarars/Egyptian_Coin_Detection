# Egyptian Coin Counter — CV Pipeline Visualizer

A Computer Vision project that detects and classifies Egyptian coins from an image using OpenCV and Tkinter.

The system identifies:

* 1 EGP
* 50 Piastres
* 25 Piastres

It also visualizes every stage of the computer vision pipeline to help understand how the algorithm works and debug detection problems.

---

## Features

* Upload and process coin images
* HSV-based background suppression
* CLAHE contrast enhancement
* Hough Circle detection
* Contour-based fallback detection
* IoU-based Non-Maximum Suppression (NMS)
* Coin classification using HSV saturation analysis
* GUI visualization for each processing stage
* Displays:

  * Total detected coins
  * Total monetary value

---

## Pipeline Overview

### 1. Image Loading & Resizing

The uploaded image is resized for consistent processing speed and detection accuracy.

### 2. Background Suppression

HSV masking removes unnecessary background regions while preserving coin surfaces.

### 3. Image Enhancement

* CLAHE improves local contrast
* Gaussian Blur reduces noise

### 4. Coin Detection

* Primary method: Hough Circle Transform
* Fallback method: Contour-based detection for overlapping coins

### 5. Coin Classification

Coins are classified based on HSV saturation characteristics:

* 1 EGP → bimetallic structure
* 50 Pt → high saturation
* 25 Pt → lower saturation

### 6. Final Visualization

Detected coins are labeled and displayed with their values.

---

## Technologies Used

* Python
* OpenCV
* NumPy
* Tkinter
* Pillow (PIL)

---

## Installation

Install the required libraries:

```bash
pip install opencv-python numpy pillow
```

Tkinter is included with most Python installations.

---

## Run the Project

```bash
python main.py
```

---

## Project Structure

```text
project/
│
├── main.py
├── README.md

```

---

## GUI Preview

The interface displays:

* Original image
* Background mask
* Preprocessed image
* Raw detections
* Final classified result

---


## Future Improvements

* Improve classification accuracy
* Support more Egyptian coin types
* Add real-time webcam detection
* Train a deep learning classifier

---

Computer Vision Project
CSE 473 — Spring 2025
