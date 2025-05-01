import pandas as pd
import numpy as np
import joblib
import argparse

# ------------------ Load Saved Artifacts ------------------
model = joblib.load("rf_model2.pkl")
imputer = joblib.load("imputer.pkl")
selector = joblib.load("selector.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ------------------ Helper Function ------------------
def preprocess_input(csv_path):
    raw_df = pd.read_csv(csv_path)
    
    # Transpose to shape (samples, timepoints)
    X_eval = raw_df.T.reset_index(drop=True)
    X_eval.columns = raw_df.iloc[:, 0].astype(float)
    
    # Convert to numeric
    X_eval = X_eval.apply(pd.to_numeric, errors='coerce')
    
    # Replace 0s with NaN
    X_eval.replace(0, np.nan, inplace=True)
    
    # Align columns by position (to match training columns)
    X_eval = X_eval.iloc[:, :imputer.statistics_.shape[0]]
    
    # Impute, select, scale
    X_imputed = pd.DataFrame(imputer.transform(X_eval))
    X_selected = selector.transform(X_imputed)
    X_scaled = scaler.transform(X_selected)
    
    return X_scaled

# ------------------ Prediction Function ------------------
def predict(csv_path, threshold=0.40):
    X_eval_final = preprocess_input(csv_path)
    
    perfect_index = label_encoder.transform(['perfect'])[0]
    probs = model.predict_proba(X_eval_final)[:, perfect_index]
    
    # Thresholding
    binary_preds = (probs > threshold).astype(int)
    y_pred = np.full_like(binary_preds, fill_value=label_encoder.transform(['imperfect'])[0])
    y_pred[binary_preds == 1] = perfect_index
    
    labels = label_encoder.inverse_transform(y_pred)
    
    for i, (label, prob) in enumerate(zip(labels, probs)):
        print(f"Sample {i+1}: Predicted = {label}, Perfect_Prob = {prob:.4f}")

# ------------------ Command-Line Interface ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Path to evaluation CSV file")
    parser.add_argument("--threshold", type=float, default=0.40, help="Probability threshold for classifying 'perfect'")
    args = parser.parse_args()
    
    predict(args.input_csv, args.threshold)
