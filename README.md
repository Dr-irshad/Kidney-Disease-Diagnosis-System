# Kidney Disease Diagnosis System (Customer: Wisio, Hong Kong)

## Overview

The **Kidney Disease Diagnosis System** is an AI-powered solution developed to assist healthcare professionals in diagnosing kidney-related diseases with high accuracy. The system uses computer vision algorithms to automate the analysis of medical imaging modalities, such as **ultrasounds**, **CT scans**, and **MRIs**, to detect abnormalities like **tumors**, **cysts**, and **stones**. This real-time, automated analysis supports early disease detection, enhances monitoring, and helps in making informed treatment decisions.

## Features

- **AI-powered diagnosis**: Leverages deep learning models to analyze medical images and detect abnormalities associated with kidney diseases.
- **Support for multiple imaging modalities**: The system supports **ultrasound**, **CT scans**, and **MRI** images, enabling comprehensive analysis of different medical imaging formats.
- **Real-time image analysis**: The system can process and analyze medical images in real time, providing instant results to healthcare professionals.
- **Tumor, cyst, and stone detection**: Using computer vision, the system accurately identifies key issues like kidney tumors, cysts, and stones in medical images.
- **High precision**: With advanced YOLO-based segmentation algorithms, the system ensures highly accurate detection and classification of kidney abnormalities.
- **Enhanced disease monitoring**: The system continuously monitors and tracks changes in kidney conditions, providing data to support ongoing treatment decisions.

## Technologies Used

- **Computer Vision**: Utilized for analyzing medical images, identifying abnormalities, and segmenting areas of interest within kidney scans.
- **YOLO Segmentation**: YOLO (You Only Look Once) is used for real-time object detection and segmentation of kidney-related abnormalities like tumors, cysts, and stones.
- **Real-time Analysis**: Ensures that images are analyzed instantly upon input, allowing for fast diagnosis and decision-making.
- **Deep Learning**: Convolutional Neural Networks (CNNs) were used for image classification, object detection, and segmentation tasks.
- **TensorFlow/Keras**: Used for training and deploying deep learning models for accurate image analysis.

## Installation Guide

### Prerequisites

Before setting up the system, ensure you have the following installed:

- Python 3.7 or above
- TensorFlow 2.0 or higher
- OpenCV
- Keras
- NumPy
- Matplotlib
- PyTorch (optional for additional models)
- Required dependencies for YOLO model

### Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-repo/kidney-disease-diagnosis.git
   cd kidney-disease-diagnosis

2. **Create a virtual environment (optional but recommended)**

    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`

3. **Install dependencies**
  Ensure that pip is updated, then install the required libraries:

   ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
   
4. **Download the YOLO pre-trained model weights**


The system uses **YOLO** (You Only Look Once) for segmentation, and you will need to download the pre-trained model weights in order to perform the analysis. You can either download the pre-trained weights directly or use a local path if you already have the weights.

### Steps to Download YOLO Weights

  1. **Download from the official YOLO repository:**
     - Go to the [YOLO Model Weights page](https://github.com/AlexeyAB/darknet) and download the pre-trained weights.
     
  2. **Direct link to download YOLO weights**:
     - You can directly download the weights from the following link:
       - [YOLOv4 weights (darknet)](https://pjreddie.com/media/files/yolov3.weights)
     
     After downloading, place the weights file (`yolov3.weights`) in the directory where your model code expects it. This can be a custom directory you create within your project.
  
  3. **Using a Local Path:**
     - If you already have the weights file saved locally, you can specify the path to the weights in the configuration file or directly in the system's script when running the model.

       Example (for local path):
       ```python
       weights_path = "/path/to/your/yolov3.weights"

5. **Run the System**

Once dependencies are installed and the model is downloaded, you can run the system to analyze medical images:
    ```bash
    python src/diagnose.py --image image.jpg

This will process the input image and return the analysis, including the detection of tumors, cysts, and stones.

## Usage
The Kidney Disease Diagnosis System can be used in the following ways:

1. **Diagnosis of Medical Images:**

    - Healthcare professionals can upload medical images (e.g., ultrasound, CT, or MRI scans) into the system.
    - The AI system will process these images, identifying areas with potential kidney abnormalities like tumors, cysts, or stones.
    - Detailed results will be provided, with visual highlights indicating detected abnormalities.

2. **Continuous Monitoring:**

    - The system can be used to monitor kidney health over time, tracking changes in the size and number of detected abnormalities.
    - It can be integrated with electronic health records (EHR) to assist doctors with ongoing patient care.
   
## Architecture

### System Components

  - **Frontend:** User interface for healthcare professionals to upload and view medical images.
  - **Backend:** AI-powered server running the trained YOLO segmentation model to analyze images.
  - **Database (optional):** For storing image history, patient data, and analysis results.
  - **Model API:** API endpoint that communicates with the trained YOLO model for real-time analysis.

## Data Flow

  - The user uploads an image (CT scan, ultrasound, MRI) via the frontend.
  - The backend processes the image, sending it to the model for analysis.
  - The model detects and segments areas of interest (e.g., tumors, cysts, stones) using YOLO.
  - Results are displayed on the frontend, showing abnormal regions and classification.

## Contributing

We welcome contributions to improve the system! To contribute:

  - Fork the repository.
  - Create a new branch.
  - Make changes and test them.
  - Submit a pull request with a clear explanation of the changes.

## Contribution Guidelines

  - Ensure the code adheres to existing formatting and structure conventions.
  - Include relevant tests for new features or changes.
  - Add documentation for new functions or modules.
  - Keep the commit history clean and well-organized.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

  - **Wisio (Hong Kong):** For being the customer and driving the development of the Kidney Disease Diagnosis System.
  - **YOLO Contributors:** For developing the powerful YOLO algorithm for real-time object detection and segmentation.
  - **TensorFlow/Keras:** For providing the deep learning framework that powers the system's model training and inference.
