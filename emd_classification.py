# emd_classification.py
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix, classification_report)
import joblib
from PyEMD import EMD
from tqdm import tqdm
import argparse
import os
import pandas as pd
import h5py
from seismic_ml import SeismicDataProcessor, SeismicClassifier
from pysdkit import FAEMD

# Utility function for timestamped prints
def print_timestamp(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

class EMDFeatureExtractor:
    """Class for extracting features using Empirical Mode Decomposition"""

    def __init__(self, n_imfs=None, max_sifts=10):
        self.n_imfs = n_imfs
        self.max_sifts = max_sifts
        # self.emd_decomposer = EMD(max_imf=n_imfs, max_sifts=self.max_sifts, spline_kind='linear')  # Try linear instead of cubic)
        self.emd_decomposer = FAEMD(max_imfs=n_imfs)

    def _calculate_imf_energy(self, imfs):
        energies = np.array([np.sum(imf**2) for imf in imfs])
        total_energy = np.sum(energies)
        if total_energy == 0:
            return np.zeros_like(energies)
        return energies / total_energy
    
    def _calculate_imf_frequency(self, imf, fs):
        if np.all(imf == 0) or len(imf) < 2:
            return 0.0
        f, Pxx = welch(imf, fs=fs, nperseg=min(256, len(imf)))
        if len(Pxx) == 0 or np.all(Pxx == 0):
            return 0.0
        return f[np.argmax(Pxx)]
    
    def _calculate_reconstruction_error(self, original, imfs):
        if imfs is None or len(imfs) == 0:
            return 0.0
        reconstructed = np.sum(imfs, axis=0)
        if len(original) != len(reconstructed):
            return 0.0
        return np.mean((original - reconstructed)**2)

    def extract_features(self, data, fs):
        features = []
        for comp_idx in range(3):
            signal = data[:, comp_idx]
            try:
                # imfs = self.emd_decomposer.emd(signal)
                imfs = self.emd_decomposer.fit_transform(signal)
                if imfs is None or len(imfs) == 0:
                    print_timestamp(f"    Warning: EMD decomposition failed for component {comp_idx}")
                    return None

                # Use all IMFs without filtering
                valid_imfs = imfs
                valid_imfs = [imf if np.all(np.isfinite(imf)) else np.zeros_like(imf) for imf in valid_imfs]

                # Pad or truncate to required number of IMFs
                n_needed = self.n_imfs or len(valid_imfs)
                if len(valid_imfs) < n_needed:
                    print_timestamp(f"    Warning: Not enough IMFs for component {comp_idx} (found {len(valid_imfs)}, need {n_needed})")
                    valid_imfs = list(valid_imfs) + [np.zeros_like(signal)] * (n_needed - len(valid_imfs))
                valid_imfs = valid_imfs[:n_needed]
                valid_imfs = np.array(valid_imfs)

                # Feature extraction
                features.extend(self._calculate_imf_energy(valid_imfs))
                features.extend([self._calculate_imf_frequency(imf, fs) for imf in valid_imfs])
                features.append(self._calculate_reconstruction_error(signal, valid_imfs))

            except Exception as e:
                print_timestamp(f"    Error in EMD decomposition for component {comp_idx}: {str(e)}")
                return None
        features = np.array(features)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features
    
    def process_data(self, eq_data, noise_data):
        all_features = []
        all_labels = []
        all_targets = []

        for trace_name, (data, fs, mag) in tqdm(eq_data.items(), desc='EMD EQ', total=len(eq_data)):
            print_timestamp(f"Decomposing trace: {trace_name}")
            features = self.extract_features(data, fs)
            if features is None:
                print_timestamp(f"    Discarding trace {trace_name} due to EMD failure.")
                continue
            all_features.append(features)
            all_labels.append(1)
            all_targets.append(mag)

        for trace_name, (data, fs) in tqdm(noise_data.items(), desc='EMD Noise', total=len(noise_data)):
            print_timestamp(f"Decomposing trace: {trace_name}")
            features = self.extract_features(data, fs)
            if features is None:
                print_timestamp(f"    Discarding trace {trace_name} due to EMD failure.")
                continue
            all_features.append(features)
            all_labels.append(0)
            all_targets.append(0)

        return np.array(all_features), np.array(all_labels), np.array(all_targets)

class EMDFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_imfs=5):
        self.n_imfs = n_imfs
        self.extractor = EMDFeatureExtractor(n_imfs=n_imfs)
        self.valid_indices_ = None  # Store valid indices after transform

    def fit(self, X, y=None):
        print_timestamp("Fitting data...")
        return self
    def transform(self, X):
        print_timestamp("Transforming data...")
        valid_indices = []
        features = []
        for i, (data, fs) in enumerate(tqdm(X, desc='Extracting EMD features', total=len(X))):
            feat = self.extractor.extract_features(data, fs)
            if feat is not None:
                features.append(feat)
                valid_indices.append(i)
            else:
                print_timestamp(f"Feature extraction failed for sample {i}, skipping.")
        features = np.array(features)
        self.valid_indices_ = valid_indices  # Store for later use
        return features



# Main execution for EMD analysis
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seismic EMD Classification")
    parser.add_argument('--mode', type=str, default='prod', choices=['fast', 'prod'], help='Mode: test or prod')
    parser.add_argument('--nrows', type=int, default=20000, help='Number of rows to load in test mode')
    parser.add_argument('--chunksize', type=int, default=20000, help='Chunk size to use in test mode')
    # Add arguments for train/val file paths
    parser.add_argument('--eq_h5', type=str, required=True, help='Earthquake HDF5 file (shared)')
    parser.add_argument('--noise_h5', type=str, required=True, help='Noise HDF5 file (shared)')
    parser.add_argument('--train_eq_csv', type=str, required=True, help='Training earthquake CSV file')
    parser.add_argument('--train_noise_csv', type=str, required=True, help='Training noise CSV file')
    parser.add_argument('--val_eq_csv', type=str, required=True, help='Validation earthquake CSV file')
    parser.add_argument('--val_noise_csv', type=str, required=True, help='Validation noise CSV file')
    args = parser.parse_args()

    MODE = args.mode
    EQ_FILTERS = {'trace_category': 'earthquake_local'}

    print_timestamp("Loading training data...")
    load_start = time.time()
    train_processor = SeismicDataProcessor(
        file_name_eq=args.eq_h5,
        file_name_noise=args.noise_h5,
        csv_file_eq=args.train_eq_csv,
        csv_file_noise=args.train_noise_csv,
        mode=MODE,
        eq_filters=EQ_FILTERS,
        nrows=args.nrows if MODE in ['test', 'fast'] else None,
        chunksize=args.chunksize if MODE in ['test', 'fast'] else None
    )
    eq_data_train, noise_data_train = train_processor.load_data()
    print_timestamp("Training data loaded successfully.")
    
    print_timestamp("Loading validation data...")
    val_processor = SeismicDataProcessor(
        file_name_eq=args.eq_h5,
        file_name_noise=args.noise_h5,
        csv_file_eq=args.val_eq_csv,
        csv_file_noise=args.val_noise_csv,
        mode=MODE,
        eq_filters=EQ_FILTERS,
        nrows=args.nrows if MODE in ['test', 'fast'] else None,
        chunksize=args.chunksize if MODE in ['test', 'fast'] else None
    )
    eq_data_val, noise_data_val = val_processor.load_data()
    load_end = time.time()
    print_timestamp(f"Data loading took {load_end - load_start:.2f} seconds.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_size = "all" if train_processor.nrows is None else f"n{train_processor.nrows}"
    MODEL_DIR = "saved_models"
    EMD_MODEL_PATH = os.path.join(MODEL_DIR, f"{timestamp}_{dataset_size}_emd_classifier.pkl")

    print_timestamp("Training classifier with EMD features...")
    emd_clf_start = time.time()
    emd_classifier = SeismicClassifier(feature_transformer=EMDFeatureTransformer(n_imfs=5))
    emd_metrics = emd_classifier.train(eq_data_train, noise_data_train, eq_data_val, noise_data_val)
    emd_clf_end = time.time()
    
    print_timestamp("EMD Classification results:")
    for metric, value in emd_metrics.items():
        print_timestamp(f"  {metric}: {value}")
    print_timestamp(f"EMD Classification took {emd_clf_end - emd_clf_start:.2f} seconds.")

    emd_classifier.save_pipeline(EMD_MODEL_PATH)
    total_time = time.time() - load_start
    print_timestamp(f"Total execution time: {total_time:.2f} seconds.") 