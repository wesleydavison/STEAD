import os
import numpy as np
import pandas as pd
import h5py
from seismic_ml import SeismicClassifier, print_timestamp
from decompose_emd import decompose_and_save_emd
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
parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to process (default: process all)')
args = parser.parse_args()
MODEL_PATH = args.model_path
csv_file_eq = args.eq_csv
csv_file_noise = args.noise_csv
n_samples = args.n_samples

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

# --- Step 2: Extract features for both earthquake and noise data ---
print_timestamp("Extracting EMD features...")

# Process earthquake data
print_timestamp("Processing earthquake data...")
eq_features, eq_labels, _ = decompose_and_save_emd(
    file_name_hdf5=file_name_merged,
    csv_file=csv_file_eq,
    output_file="temp_emd_features_eq.h5",
    n_samples=n_samples,
    return_features=True,
    generate_plots=False,
    generate_boxplots=False
)

# Process noise data
print_timestamp("Processing noise data...")
noise_features, noise_labels, _ = decompose_and_save_emd(
    file_name_hdf5=file_name_merged,
    csv_file=csv_file_noise,
    output_file="temp_emd_features_noise.h5",
    n_samples=n_samples,
    return_features=True,
    generate_plots=False,
    generate_boxplots=False
)

# Combine features and labels
features = np.vstack([eq_features, noise_features])
labels = np.concatenate([eq_labels, noise_labels])

# Clean up temporary files
for temp_file in ["temp_emd_features_eq.h5", "temp_emd_features_noise.h5"]:
    if os.path.exists(temp_file):
        os.remove(temp_file)

# Save features for analysis
np.save("emd_test_features.npy", features)
print(f"Test features shape: {features.shape}")
print(f"Test labels shape: {labels.shape}")
print(f"Class distribution: {sum(labels)} earthquakes, {len(labels)-sum(labels)} noise")

# Load and apply model
print_timestamp("Loading EMD pipeline...")
classifier = SeismicClassifier()
classifier.load_pipeline(MODEL_PATH)

print_timestamp("Predicting...")
y_pred = classifier.predict(features)
y_pred_proba = classifier.predict_proba(features)[:, 1]

# Calculate and print metrics
metrics = classifier.evaluate(labels, y_pred, y_pred_proba)

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
np.save("emd_test_true_labels.npy", labels)
np.save("emd_test_probabilities.npy", y_pred_proba) 