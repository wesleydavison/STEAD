import os
import numpy as np
import pandas as pd
import h5py
from seismic_ml import SeismicClassifier, print_timestamp
from seismic_ml import print_timestamp
from emd_classification import EMDFeatureTransformer
from sklearn.metrics import precision_score, f1_score, classification_report, confusion_matrix, average_precision_score, roc_auc_score
import argparse
from sklearn.utils import resample
from tqdm import tqdm

# --- Configuration ---
file_name_merged = r"/users/230442014/archive/STEAD_dataset/merged.hdf5"
# Use separate CSVs for earthquake and noise
parser = argparse.ArgumentParser(description="Test seismic EMD classifier pipeline.")
parser.add_argument('--model_path', type=str, help='Path to the saved EMD model pipeline .pkl file')
parser.add_argument('--eq_csv', type=str, required=True, help='CSV file for earthquake traces')
parser.add_argument('--noise_csv', type=str, required=True, help='CSV file for noise traces')
args = parser.parse_args()
MODEL_PATH = args.model_path
csv_file_eq = args.eq_csv
csv_file_noise = args.noise_csv

dtype_dict = {
    'trace_name': 'category',
    'trace_category': 'category',
    'source_magnitude': 'float32'
}

# --- Step 1: Read trace names and metadata from separate CSVs ---
print_timestamp("Reading earthquake and noise metadata from separate CSVs...")
eq_df = pd.read_csv(csv_file_eq, usecols=['trace_name', 'trace_category', 'source_magnitude'], dtype=dtype_dict)
noise_df = pd.read_csv(csv_file_noise, usecols=['trace_name', 'trace_category', 'source_magnitude'], dtype=dtype_dict)

print_timestamp(f"  Earthquake samples: {len(eq_df)}")
print_timestamp(f"  Noise samples: {len(noise_df)}")

# Concatenate for unified processing
sampled_df = pd.concat([eq_df, noise_df], ignore_index=True)

# --- Step 2: Load only the sampled traces from HDF5 ---
print_timestamp("Loading sampled traces from HDF5 and extracting features...")
X_test = []
y_test = []
failed_traces = 0

with h5py.File(file_name_merged, 'r') as h5f:
    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc='Loading traces'):
        trace_name = row['trace_name']
        trace_category = row['trace_category']
        label = 1 if trace_category == 'earthquake_local' else 0
        # Load the trace data
        ds = h5f.get(f"data/{trace_name}")
        if ds is None:
            failed_traces += 1
            print_timestamp(f"Warning: Could not load trace {trace_name}")
            continue
        try:
            data = np.array(ds)
            fs = float(ds.attrs.get('sampling_rate', 100.0))
            # Basic data validation
            if data.shape[1] != 3:  # Should have 3 components
                failed_traces += 1
                print_timestamp(f"Warning: Invalid data shape for trace {trace_name}: {data.shape}")
                continue
            X_test.append((data, fs))
            y_test.append(label)
        except Exception as e:
            failed_traces += 1
            print_timestamp(f"Error processing trace {trace_name}: {str(e)}")
            continue

print_timestamp(f"Successfully loaded {len(X_test)} traces ({failed_traces} failed)")
print_timestamp(f"Class distribution: {sum(y_test)} earthquakes, {len(y_test)-sum(y_test)} noise")

# Convert to numpy arrays
X_test = np.array(X_test, dtype=object)
y_test = np.array(y_test)

# Load and apply model
print_timestamp("Loading EMD pipeline...")
classifier = SeismicClassifier(feature_transformer=EMDFeatureTransformer(n_imfs=5))
classifier.load_pipeline(MODEL_PATH)

# Extract features using the model's EMD feature transformer
feature_transformer = classifier.pipeline.named_steps['feature']
test_features = feature_transformer.transform(X_test)
# Filter y_test to only valid indices
if hasattr(feature_transformer, 'valid_indices_') and feature_transformer.valid_indices_ is not None:
    y_test = y_test[feature_transformer.valid_indices_]
np.save("emd_test_features.npy", test_features)
print(f"Test features shape: {test_features.shape}")

print_timestamp("Predicting...")
y_pred = classifier.predict(X_test)
y_pred_proba = classifier.predict_proba(X_test)[:, 1]

# Calculate and print metrics
metrics = classifier.evaluate(y_test, y_pred, y_pred_proba)

print("Accuracy:", metrics['accuracy'])
print("Precision:", metrics['precision'])
print("Recall:", metrics['recall'])
print("F1 Score:", metrics['f1'])
print("PR-AUC:", metrics['pr_auc'])
print("ROC-AUC:", metrics['roc_auc'])
print("Confusion Matrix:\n", metrics['confusion_matrix'])
print("Classification Report:\n", metrics['classification_report'])

# Save predictions for analysis
np.save("emd_test_predictions.npy", y_pred)
np.save("emd_test_true_labels.npy", y_test)
np.save("emd_test_probabilities.npy", y_pred_proba) 