# Skin Lesion Classification Project

This project aims to classify skin lesions based on images using machine learning models. Early detection of melanoma and other skin cancers is crucial for improving survival rates. The project includes a Streamlit app for testing classification ability.

![Dermatoscope](dermatoscope.png)

## Key Stakeholders

- **Healthcare Workers:** They may use a reliable model to assist with diagnoses, helping confirm their diagnosis or prompting further consideration.
  
- **Medical Students:** This project provides valuable hands-on experience in applying machine learning to healthcare challenges.
  
- **Public Awareness Campaigns:** Initiatives like [SunSmart](https://www.sunsmart.com.au/) can benefit from improved skin lesion classification, aiding in public awareness and education.

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

- Incorporate other available features, such as age, sex, and localization, into a model.

## How to Use the Streamlit App

1. Clone the repository to your local machine.
2. Install the required packages listed in `requirements.txt`.
3. Run the Streamlit app using the command `streamlit run app.py`.
4. Follow the instructions on the app to test your classification ability and compete against the model.

## Sources

HAM10000 dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

[1] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern: "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)", 2018; https://arxiv.org/abs/1902.03368

[2] Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018).


