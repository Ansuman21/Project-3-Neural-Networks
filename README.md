# Transforming Meteorology: Deep Learning-Based Classification of Weather Conditions

## Overview
This project leverages deep learning techniques to classify images of weather phenomena into three categories: lightning, snow, and rain. By using a convolutional neural network (CNN) based on the **ResNet18** architecture, we created a robust model that can accurately identify these weather conditions from images. The model was trained on a pre-processed dataset, which underwent augmentation and transformation to ensure robustness.

## Business Case
Accurate classification of weather phenomena from images has several practical applications:
- **Meteorological Analysis**: Helping weather forecasting systems to automatically categorize and analyze images from weather satellites, reducing manual intervention.
- **Climate Monitoring**: Assisting climate researchers by automating the identification of weather conditions for further analysis.
- **Disaster Management**: Classifying severe weather conditions like lightning, snowstorms, and heavy rainfall to provide early warnings to affected regions.

By automating the process of weather image classification, businesses and governmental agencies can respond faster to changing weather conditions, make more accurate predictions, and improve public safety.

## Key Steps in the Solution

### 1. Dataset Preparation
The dataset used in this project contained images labeled as **lightning**, **snow**, and **rain**. The images were standardized and preprocessed to ensure consistency in model training:
- **Resizing**: All images were resized to uniform dimensions.
- **Normalization**: Pixel values were normalized to improve the model's performance.

### 2. Data Augmentation
To increase the model's ability to generalize to new, unseen data, data augmentation was applied. This included transformations such as:
- **Rotation**
- **Flipping**
- **Zooming**
- **Shearing**
- **Brightness adjustment**

### 3. Model Selection and Training
The **ResNet18** architecture was chosen for its proven effectiveness in image classification tasks. The model was pre-trained on ImageNet and fine-tuned for our specific weather-related categories. The training process included:
- **Loss Function**: Cross-entropy loss
- **Optimizer**: Adam optimizer
- **Early Stopping**: To prevent overfitting, training was halted early if validation accuracy stopped improving.

### 4. Model Evaluation
The model's performance was evaluated using several metrics:
- **Accuracy**: The overall classification accuracy on the test dataset was **94.4%**.
- **Precision, Recall, and F1-Score**: These metrics were calculated for each category (lightning, snow, and rain), showing excellent performance across all classes.

### 5. Inference and Predictions
The model was used to predict weather conditions in new images. The Gradio interface was set up for real-time predictions, allowing users to upload images and get the predicted class along with the probability.

### 6. Results
The model successfully predicted:
- **Lightning**: 100% accuracy on new lightning images.
- **Snow**: 95% accuracy on new snow images.
- **Rain**: 94% accuracy on new rain images.

## Recommendations for Improvement
- **Expand the Dataset**: Including more weather conditions (fog, hail, etc.) and more diverse images could help improve the model's generalization.
- **Model Enhancement**: Experiment with more complex models like **ResNet50** or **DenseNet** for potentially better feature extraction.
- **Hyperparameter Tuning**: Fine-tuning hyperparameters (learning rate, batch size, etc.) could further optimize the model’s performance.

## Future Scope
- **Real-time Deployment**: The model can be deployed in real-time weather monitoring systems, using satellite images to detect weather conditions instantly.
- **Integration with Weather Prediction Systems**: Integrating this model into meteorological prediction systems to enhance the accuracy of weather forecasts.

## Technologies Used
- **Python**: Main programming language
- **PyTorch**: Deep learning framework
- **Gradio**: For building the user interface
- **ResNet18**: Pretrained model for feature extraction
- **OpenCV**: For image processing and augmentation

## Conclusion
The project demonstrated a successful application of deep learning techniques to classify weather images into three major categories. With high accuracy and the potential for real-time applications, the model can be used in various domains like meteorology, disaster management, and climate monitoring.

---

**Author**  
Ansuman Patnaik  
MS in Data Science & Analytics, Yeshiva University  
Email: ansu1p89k@gmail.com
