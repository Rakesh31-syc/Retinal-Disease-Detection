# Retinal Disease Detection

## Overview
Fundus imaging provides physicians with a snapshot of the inside of a patient's eye. By analyzing these images, doctors can detect abnormalities in the retina, helping diagnose various eye diseases such as diabetic retinopathy, glaucoma, and macular degeneration.

This project leverages deep learning to automatically detect multiple retinal diseases from fundus images. It is a multi-label classification problem, as a single retinal image can exhibit multiple diseases simultaneously.

## Features
- Deep Learning Model: Trained using convolutional neural networks (CNNs) to classify different retinal diseases.
- Multi-Label Classification: Supports detecting multiple diseases in a single image.
- Automated Report Generation: Generates a detailed diagnosis report after analyzing an image.
- Email Notification System: Sends the generated report to the concerned doctor or patient.
- User-Friendly Interface: Simplifies the process of uploading images and retrieving results.

## Dataset
The dataset (sourced from [this Kaggle competition](https://www.kaggle.com/c/vietai-advance-course-retinal-disease-detection/overview)) consists of:
- 3,285 images from CTEH** (3,210 abnormal and 75 normal)
- 500 normal images from Messidor and EYEPACS datasets

### Diseases Detected
- Opacity
- Diabetic Retinopathy
- Glaucoma
- Macular Edema
- Macular Degeneration
- Retinal Vascular Occlusion

## How It Works
1. Image Upload: Users upload a fundus image through the interface.
2. Model Prediction: The deep learning model processes the image and predicts the presence of diseases.
3. Report Generation: A detailed report is created with findings and confidence scores.
4. Email Notification: The report is automatically sent to the registered doctor or patient.

## Installation & Usage
Prerequisites
- Python 3.8+
- TensorFlow / PyTorch
- OpenCV
- Flask / Django (for web interface)
- SMTP for email notifications

## Steps to Run
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/Retinal-Disease-Detection.git
   cd Retinal-Disease-Detection
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:
   ```sh
   python app.py
   ```
4. Access the interface at `http://localhost:5000` (if using Flask).

## Future Enhancements
- Improve model accuracy with larger datasets.
- Develop a mobile app for easy access.
- Integrate real-time image analysis.
- Enhance explainability with heatmaps for disease detection.

## Contributors
- A.RAKESH([GitHub](https://github.com/Rakesh31-syc)

## License
This project is licensed under the MIT License.

