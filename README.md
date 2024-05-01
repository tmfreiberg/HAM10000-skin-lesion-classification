# Skin Lesion Classification Project

This project aims to classify skin lesions based on images using machine learning models. Early detection of melanoma and other skin cancers is crucial for improving survival rates. The project includes a Streamlit app for testing classification ability.

## Key Stakeholders

- **Healthcare Workers:** They may use a reliable model to assist with diagnoses, helping confirm their diagnosis or prompting further consideration.
  
- **Medical Students:** This project provides valuable hands-on experience in applying machine learning to healthcare challenges.
  
- **Public Awareness Campaigns:** Initiatives like [SunSmart](https://www.sunsmart.com.au/)" can benefit from improved skin lesion classification, aiding in public awareness and education.

## Project Overview

- **Dataset:** The project uses the Human Against Machine 10000 (HAM10000) dataset containing images of skin lesions labeled into different classes, including melanoma, nevi (moles), and other types of lesions.

- **Extensive Data Exploration:** The dataset underwent thorough exploration to understand its characteristics, including class distribution, imbalances, and potential biases.
  
- **Models:** Popular convolutional neural network architectures such as ResNet and EfficientNet are fine-tuned on the HAM10000 dataset for skin lesion classification.
  
- **Data Augmentation:** Random transformations such as random cropping, color jitter, and rotations are applied to augment the dataset and improve model generalization.

## Streamlit App

The Streamlit app allows users to:

- Test their ability to classify skin lesions as moles or melanoma.
  
- Compete against one of our fine-tuned machine learning models to see how well they can classify lesions compared to the model.

## Future Work

- Incorporate dropout regularization to mitigate overfitting.
  
- Address artifacts in images, such as markings from dermatologists, for improved classification accuracy.

## How to Use the Streamlit App

1. Clone the repository to your local machine.
2. Install the required packages listed in `requirements.txt`.
3. Run the Streamlit app using the command `streamlit run app.py`.
4. Follow the instructions on the app to test your classification ability and compete against the model.

