HEAD
# Random Forest Classifier for Signal-Based Molecular Biology Classification

This project implements a Random Forest model to classify signal data as either `"perfect"` or `"imperfect"` based on time-series features. The model was trained on experimental signal data, optimized using probability threshold tuning, and evaluated using both train-test split and stratified cross-validation.

---

## Repository Structure

```
├── prediction.py                     # Prediction script (CLI)
├── molecular_biology_experiment_classification.ipynb              # Full training notebook
├── rf_model.pkl                   # Trained Random Forest model with train/test split
├── rf_model2.pkl                  # Trained Random Forest model with stratified K-Fold Cross-Validation 
├── imputer.pkl                    # Fitted SimpleImputer
├── selector.pkl                   # VarianceThreshold feature selector
├── scaler.pkl                     # StandardScaler for feature normalization
├── label_encoder.pkl              # Fitted LabelEncoder for class labels
├── training_data_external.csv     # Training dataset (transposed timepoint format)
├── evaluation_data_external.csv   # Evaluation dataset
├── requirements.txt               # Required Python packages
├── Experiment_Report.docx         # Experiment details
├── Visualizations                 # Folder containing visualizations
└── README.md                      # Project documentation
```

---

## Setup Instructions

1. **Clone the repository**:

```bash
git clone https://github.com/Your-username/Molecular-Biology-Machine-learning-classification
cd Molecular-Biology-Machine-learning-classification
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

---

## Usage: Prediction Script

Use the `prediction.py` script to generate predictions on new evaluation data.

```bash
python prediction.py evaluation_data_external.csv --threshold 0.40
```

- `evaluation_data_external.csv`: Path to your evaluation dataset
- `--threshold`: Optional custom probability threshold (default is 0.40)

Output:
- Predicted class label and associated probability for each sample

---

## Model Highlights

- **Model Type**: Random Forest Classifier (n_estimators=300, class_weight={0:1, 1:5})
- **Threshold Tuning**: 0.40 chosen based on precision-recall tradeoff
- **Evaluation**:  
  - Train-Test Split: 100% accuracy  
  - Stratified K-Fold (n=3):  
    - F1 Score (weighted): 0.97  
    - Perfect class recall: 0.75  
- **Feature Importance**: Top timepoints driving predictions visualized and logged

---

## Evaluation Visualizations

The following charts were used for evaluation:
- ROC Curve (AUC = 1.00)
- Precision-Recall Curve (Avg Precision = 1.00)
- Feature Importance Bar Chart
- Predicted Probability Distribution
- Sorted Confidence Scores

(See `Visualizations/` folder)

---

## Reproducibility

To reproduce this model:
- Use `molecular_biology_experiment_classification.ipynb` to retrain from scratch
- Use `prediction.py` to evaluate new data
- All preprocessing artifacts (`.pkl` files) are saved for reuse

---

## Requirements

```
pandas
numpy
scikit-learn
joblib
matplotlib
```

---

## Contact

For questions or collaboration, contact: **Faith Edafetanure-Ibeh**  
Email: *Edafetanureibeh.faith@gmail.com*


