# Bangla Sign Language Recognition using LSTM

A deep learning project for recognizing Bangla sign language vowels using Long Short-Term Memory (LSTM) neural networks.

## Project Overview

This project implements an LSTM-based model to classify Bangla sign language vowels from hand pose coordinate data. The model achieves **99.74% accuracy** on the test dataset, demonstrating excellent performance in recognizing 7 different Bangla vowel signs.

## Dataset

- **Total Samples**: 1,949 data points
- **Features**: 10 coordinate features (p1-p5, r1-r5) representing hand pose positions
- **Classes**: 7 Bangla vowels - আ, ই, উ, এ, ঐ, ও, ঔ
- **Format**: CSV file with balanced class distribution

### Dataset Structure

```
- p1, p2, p3, p4, p5: Position coordinates
- r1, r2, r3, r4, r5: Rotation/angle coordinates
- label: Bangla vowel character
```

## Model Architecture

The LSTM model consists of:

- **Input Layer**: Reshaped data (samples, 2, 5) for sequence processing
- **LSTM Layer 1**: 64 units with return_sequences=True
- **Dropout Layer**: 30% dropout for regularization
- **LSTM Layer 2**: 32 units
- **Dropout Layer**: 30% dropout
- **Dense Layer**: 64 units with ReLU activation
- **Dropout Layer**: 30% dropout
- **Output Layer**: 7 units with softmax activation (for 7 classes)

**Total Parameters**: 32,903 trainable parameters

## Performance Metrics

- **Test Accuracy**: 99.74%
- **Test Loss**: 0.0472
- **Training Time**: 50 epochs
- **Validation Split**: 20% of training data

### Classification Report

```
              precision    recall  f1-score   support
           আ       1.00      0.98      0.99        55
           ই       1.00      1.00      1.00        55
           উ       1.00      1.00      1.00        56
           এ       1.00      1.00      1.00        55
           ঐ       1.00      1.00      1.00        60
           ও       0.98      1.00      0.99        55
           ঔ       1.00      1.00      1.00        54

    accuracy                           1.00       390
   macro avg       1.00      1.00      1.00       390
weighted avg       1.00      1.00      1.00       390
```

## Files in Repository

### Data Files

- `BdSL_Combined_Dataset.csv` - Main combined dataset
- `P1Sign00(অ,য়)dataset.csv` to `P1Sign08(ঔ)dataset.csv` - Individual vowel datasets
- `bangonbarna/` - Additional consonant datasets
- `sarborno/` - Vowel datasets organized by type

### Model Files

- `bangla_sign_language_lstm_model.h5` - Trained LSTM model
- `lstm_label_encoder.pkl` - Label encoder for class mapping
- `lstm_scaler.pkl` - Feature scaler for data normalization

### Notebook Files

- `LSTMtest.ipynb` - Main LSTM implementation and training
- `combindDataset.ipynb` - Dataset combination and preprocessing
- `dataclening.ipynb` - Data cleaning procedures
- `svmFE.ipynb` - SVM with feature engineering experiments
- `SVM_FEtest.ipynb` - SVM testing and evaluation

### Other Model Files

- `best_svm_model.pkl` - Best performing SVM model
- `optimized_svm_model.pkl` - Optimized SVM model
- `feature_scaler.pkl` - SVM feature scaler
- `feature_selector.pkl` - Feature selection component
- `label_encoder.pkl` - SVM label encoder

## Requirements

```python
pandas
numpy
tensorflow
keras
scikit-learn
matplotlib
seaborn
pickle
```

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd BdSL_Bangla_Sign_Language
```

2. Install required packages:

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn
```

## Usage

### Training the Model

1. Open `LSTMtest.ipynb` in Jupyter Notebook or VS Code
2. Run all cells sequentially to:
   - Load and explore the dataset
   - Preprocess the data
   - Build the LSTM model
   - Train the model
   - Evaluate performance
   - Save the trained model

### Using the Trained Model

```python
import tensorflow as tf
import pickle
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('bangla_sign_language_lstm_model.h5')

# Load preprocessing components
with open('lstm_label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('lstm_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Predict new samples
def predict_sign(features):
    # Scale features
    features_scaled = scaler.transform([features])

    # Reshape for LSTM
    features_reshaped = features_scaled.reshape(1, 2, 5)

    # Make prediction
    prediction = model.predict(features_reshaped)
    predicted_class = np.argmax(prediction)

    # Decode label
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    confidence = np.max(prediction)

    return predicted_label, confidence

# Example usage
# features = [p1, p2, p3, p4, p5, r1, r2, r3, r4, r5]
# label, confidence = predict_sign(features)
```

## Data Preprocessing

1. **Feature Scaling**: StandardScaler normalization
2. **Label Encoding**: Convert Bangla characters to numeric labels
3. **Data Reshaping**: Reshape (samples, 10) to (samples, 2, 5) for LSTM input
4. **Train-Test Split**: 80-20 split with stratification

## Model Training Process

1. **Data Loading**: Load combined dataset
2. **Preprocessing**: Scale features and encode labels
3. **Model Building**: Create LSTM architecture
4. **Training**: 50 epochs with validation monitoring
5. **Evaluation**: Test on held-out data
6. **Visualization**: Plot training curves and confusion matrix

## Results Visualization

The notebook generates:

- Training and validation accuracy curves
- Training and validation loss curves
- Confusion matrix heatmap
- Classification report with precision, recall, F1-scores

## Future Improvements

- [ ] Expand to more Bangla characters (consonants)
- [ ] Real-time sign detection using webcam
- [ ] Mobile app integration
- [ ] Data augmentation techniques
- [ ] Ensemble methods combining LSTM and SVM

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source. Please cite this work if you use it in your research.

## Acknowledgments

- Dataset contributors for Bangla sign language data collection
- TensorFlow and Keras teams for the deep learning framework
- Scikit-learn for preprocessing utilities

## Contact

For questions or collaborations, please open an issue in this repository.

---

**Note**: This model is trained specifically on the provided dataset format. For best results, ensure input data follows the same coordinate system and preprocessing steps.
