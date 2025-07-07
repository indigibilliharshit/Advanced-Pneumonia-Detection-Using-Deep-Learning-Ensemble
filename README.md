# Advanced Pneumonia Detection Using Deep Learning Ensemble

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Dataset-Chest%20X--Ray-blue.svg)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

A comprehensive deep learning pipeline for pneumonia detection from chest X-ray images using ensemble modeling with state-of-the-art CNN architectures and advanced machine learning techniques.

## 🎯 Project Overview

This project implements an advanced machine learning pipeline for pneumonia detection using:
- **Ensemble Learning**: Combining multiple deep learning models for improved accuracy
- **Transfer Learning**: Leveraging pre-trained ResNet50V2 and DenseNet121 models
- **Class Imbalance Handling**: Using SMOTE and Tomek Links for balanced datasets
- **Memory Optimization**: Efficient data loading and processing for large datasets
- **Comprehensive Evaluation**: Multiple metrics and visualizations for model assessment

## 🏆 Key Results

### Individual Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **ResNet50V2** | **89%** | **90%** | **87%** | **88%** | **0.9595** |
| **DenseNet121** | **88%** | **89%** | **85%** | **87%** | **0.9551** |
| Lightweight CNN | 62% | 39% | 50% | 38% | 0.8959 |

### Ensemble Model Performance
- **Overall Accuracy**: 84.13%
- **ROC-AUC**: 0.9634
- **Macro Average F1-Score**: 0.813
- **Normal Class Precision**: 95.92%
- **Pneumonia Class Recall**: 98.46%

## 🚀 Features

### Core Capabilities
- **Multi-Model Ensemble**: Combines ResNet50V2, DenseNet121, and custom CNN
- **Advanced Data Preprocessing**: Automated image loading, resizing, and normalization
- **Data Augmentation**: Rotation, shifting, flipping, and zooming for better generalization
- **Class Balancing**: SMOTE oversampling with Tomek Links undersampling
- **Memory Efficiency**: Optimized for limited computational resources

### Advanced Features
- **Early Stopping**: Prevents overfitting during training
- **Model Checkpointing**: Saves best model weights automatically
- **Comprehensive Visualization**: Training curves, confusion matrices, precision-recall curves
- **Individual Prediction**: Single image prediction with ensemble voting
- **Detailed Metrics**: Multiple evaluation metrics with visual comparisons

## 🛠️ Installation

### Prerequisites
```bash
python >= 3.8
tensorflow >= 2.0
opencv-python
scikit-learn
imbalanced-learn
matplotlib
seaborn
pandas
numpy
```

### Install Dependencies
```bash
pip install tensorflow opencv-python scikit-learn imbalanced-learn matplotlib seaborn pandas numpy
```

### Dataset Setup
1. Download the [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Extract to your desired directory
3. Update the `dataset_path` variable in the script

## 📊 Dataset Information

### Structure
```
chest_xray/
├── train/
│   ├── NORMAL/     (1,341 images)
│   └── PNEUMONIA/  (3,875 images)
├── test/
│   ├── NORMAL/     (234 images)
│   └── PNEUMONIA/  (390 images)
└── val/
    ├── NORMAL/     (8 images)
    └── PNEUMONIA/  (8 images)
```

### Class Distribution
- **Before SMOTE**: Imbalanced (PNEUMONIA: 74.3%, NORMAL: 25.7%)
- **After SMOTE**: Balanced (PNEUMONIA: 49.4%, NORMAL: 50.6%)

## 🔧 Usage

### Basic Usage
```python
from advanced_pneumonia_pipeline import AdvancedPneumoniaClassificationPipeline

# Initialize pipeline
dataset_path = "/path/to/chest_xray"
pipeline = AdvancedPneumoniaClassificationPipeline(dataset_path)

# Run complete pipeline
ensemble_models, predictions = pipeline.run_advanced_pipeline()
```

### Single Image Prediction
```python
# Predict on single image
image_path = "/path/to/xray_image.jpeg"
probability, label = pipeline.predict_pneumonia(ensemble_models, image_path)
print(f"Prediction: {label} (Probability: {probability:.4f})")
```

### Configuration Options
```python
# Customize pipeline parameters
pipeline.img_height = 224        # Image height
pipeline.img_width = 224         # Image width
pipeline.batch_size = 16         # Batch size for training
pipeline.epochs = 30             # Maximum training epochs
```

## 🏗️ Architecture Details

### Model Architectures

#### 1. ResNet50V2 (Transfer Learning)
- **Base**: Pre-trained ResNet50V2 (ImageNet weights)
- **Architecture**: ResNet50V2 → GlobalAveragePooling2D → Dense(256) → Dropout(0.3) → Dense(1)
- **Activation**: ReLU for hidden layers, Sigmoid for output
- **Trainable Parameters**: Only classification head (base frozen)

#### 2. DenseNet121 (Transfer Learning)
- **Base**: Pre-trained DenseNet121 (ImageNet weights)
- **Architecture**: DenseNet121 → GlobalAveragePooling2D → Dense(256) → Dropout(0.3) → Dense(1)
- **Activation**: ReLU for hidden layers, Sigmoid for output
- **Trainable Parameters**: Only classification head (base frozen)

#### 3. Lightweight CNN (Custom)
- **Architecture**: 
  - Conv2D(32) → BatchNorm → MaxPool
  - Conv2D(64) → BatchNorm → MaxPool
  - Flatten → Dense(128) → Dropout(0.3) → Dense(1)
- **Activation**: ReLU for hidden layers, Sigmoid for output
- **Purpose**: Fast inference and baseline comparison

### Ensemble Strategy
- **Method**: Soft voting (average probabilities)
- **Decision Threshold**: 0.5
- **Advantage**: Combines strengths of different architectures

## 📈 Training Process

### Data Preprocessing
1. **Image Loading**: OpenCV-based loading with error handling
2. **Resizing**: All images resized to 224×224 pixels
3. **Normalization**: Pixel values scaled to [0, 1]
4. **Augmentation**: Applied only to training data

### Class Balancing
1. **SMOTE**: Synthetic minority oversampling
2. **Tomek Links**: Remove noisy samples
3. **Result**: Balanced dataset for improved training

### Training Configuration
- **Optimizer**: Adam (learning_rate=0.0001)
- **Loss Function**: Binary crossentropy
- **Metrics**: Accuracy
- **Callbacks**: Early stopping, model checkpointing
- **Validation Split**: 20% of training data

## 📊 Evaluation Metrics

### Performance Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Detailed classification breakdown

### Visualization Features
- Training/validation loss and accuracy curves
- Confusion matrices with heatmaps
- Precision-recall curves
- Class distribution plots
- Model performance comparison charts

## 🔍 Model Interpretability

### Available Analysis
- **Confusion Matrix**: Detailed error analysis
- **Precision-Recall Curves**: Threshold selection guidance
- **Class-wise Metrics**: Per-class performance breakdown
- **Training History**: Overfitting detection

### Sample Predictions
```
Pneumonia Image:
- ResNet50V2: 99.99% confidence
- DenseNet121: 99.68% confidence  
- Lightweight CNN: 99.98% confidence
- Ensemble: 99.88% → PNEUMONIA

Normal Image:
- ResNet50V2: 0.83% confidence
- DenseNet121: 5.99% confidence
- Lightweight CNN: 99.80% confidence
- Ensemble: 35.54% → NORMAL
```

## ⚡ Performance Optimization

### Memory Efficiency Features
- **Batch Processing**: Configurable batch sizes
- **Image Limiting**: Optional sample size reduction
- **GPU Memory Growth**: Dynamic GPU memory allocation
- **Efficient Loading**: OpenCV-based image processing

### Speed Optimization
- **Transfer Learning**: Frozen base models for faster training
- **Early Stopping**: Automatic training termination
- **Reduced Augmentation**: Optimized data augmentation pipeline

## 🐛 Troubleshooting

### Common Issues

#### Memory Errors
```python
# Reduce batch size
pipeline.batch_size = 8  # Default: 16

# Limit training samples
X_train, y_train = pipeline.load_and_preprocess_images(
    train_path, max_images=500
)
```

#### GPU Issues
```python
# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

#### Poor Performance
1. Check data quality and preprocessing
2. Increase training epochs (with early stopping)
3. Adjust learning rate
4. Verify class balance

## 🤝 Contributing

### Development Setup
1. Clone the repository
```bash
git clone https://github.com/Harshit0628/Advanced-Pneumonia-Detection-Using-Deep-Learning-Ensemble.git
cd Advanced-Pneumonia-Detection-Using-Deep-Learning-Ensemble
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run tests
```bash
python -m pytest tests/
```

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Submit pull requests with detailed descriptions

## 📋 TODO

### Upcoming Features
- [ ] **LIME/SHAP Integration**: Model interpretability analysis
- [ ] **Gradio Interface**: Web-based prediction interface
- [ ] **Model Optimization**: TensorRT/ONNX conversion
- [ ] **Cross-validation**: K-fold validation implementation
- [ ] **Hyperparameter Tuning**: Automated hyperparameter optimization
- [ ] **Multi-class Extension**: Support for multiple pneumonia types

### Performance Improvements
- [ ] **Mixed Precision Training**: Faster training with FP16
- [ ] **Data Pipeline Optimization**: TensorFlow Data API integration
- [ ] **Distributed Training**: Multi-GPU training support

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Paul Mooney's Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Pre-trained Models**: TensorFlow/Keras model zoo
- **Libraries**: TensorFlow, scikit-learn, imbalanced-learn, OpenCV

## 📞 Contact

- **Author**: Indigibilli Harshit
- **Email**: harshitindigibilli@gmail.com
- **LinkedIn**: [Your LinkedIn Profile](www.linkedin.com/in/indigibilli-harshit-394366251)
- **Project Link**: [https://github.com/yourusername/pneumonia-detection]([https://github.com/yourusername/pneumonia-detection](https://github.com/Harshit0628/Advanced-Pneumonia-Detection-Using-Deep-Learning-Ensemble))

## 📊 Citation

If you use this project in your research, please cite:

```bibtex
@misc{pneumonia_detection_2024,
  title={Advanced Pneumonia Detection Using Deep Learning Ensemble},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/pneumonia-detection}},
}
```

---

**Note**: This project is for educational and research purposes. For medical applications, please consult with healthcare professionals and ensure proper validation according to medical standards.
