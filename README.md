🐟 Multiclass Fish Image Classification
A Deep Learning Approach using CNN & Transfer Learning

Author: Sudharsan Udhayakumar


🎯 Project Overview

This project focuses on classifying fish species using Deep Learning.
It compares a custom CNN model and multiple pre-trained transfer learning models, and deploys the best model using a Streamlit web app for real-time fish image prediction.

💡 Domain & Problem Statement

Domain: Image Classification
Problem: Accurately identify fish categories using CNN and Transfer Learning.
Includes model training, fine-tuning, evaluation, and deployment.

💼 Business Use Cases

🎯 Improved accuracy using transfer learning.

💻 Streamlit app for real-time fish classification.

📊 Compare multiple models to identify the best-performing one for production.

🧠 Approach Overview

Data Preprocessing & Augmentation:

Image scaling, rotation, zoom, and flip applied for better generalization.

Model Training:

Trained a custom CNN model from scratch.

Fine-tuned five pre-trained models: VGG16, ResNet50, MobileNet, InceptionV3, and EfficientNetV2B0.

Model Saving & Deployment:

Saved best-performing models in .h5 or .keras format.

Integrated model with Streamlit for live predictions.

📈 Model Performance
Model	Type	Test Accuracy
Custom CNN	From Scratch	98.56%
VGG16	Transfer Learning	96.42%
ResNet50	Transfer Learning	77.94%
MobileNet	Transfer Learning	99.50%
InceptionV3	Transfer Learning	99.56%
EfficientNetV2B0	Transfer Learning	16.32%

🧩 Bar chart comparison visualization can be inserted here.

📊 Model Evaluation & Metrics

Metrics Used:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Recommended Visuals:

Accuracy vs Epochs plot

Loss vs Epochs plot

Confusion Matrix image

⚙️ Deployment (Streamlit App)

Upload any fish image through the Streamlit interface.

The model predicts the fish category and displays confidence score.

Uses saved .keras / .h5 model for inference.

🖼️ Insert Streamlit App Screenshot Here.

🔍 Key Results & Insights

Top Models:

InceptionV3 – 99.56%

MobileNet – 99.50%

Custom CNN and VGG16 also performed strongly.

ResNet50 showed moderate accuracy.

EfficientNetV2B0 underperformed due to improper layer freezing during training.

Transfer Learning clearly improved model accuracy.

🚀 Conclusion & Future Work

✅ Deep learning models achieved high accuracy in fish species classification.

Future Enhancements:

Fix EfficientNetV2B0 fine-tuning issue.

Expand dataset with more species and samples.

Optimize hyperparameters for further accuracy gains.

Improve Streamlit UI with better visuals and interactive feedback.

📂 GitHub repository includes code, models, and documentation.
