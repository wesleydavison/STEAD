# %%
# =============================================================================
# 1. Import Libraries
# =============================================================================

import time
from datetime import datetime
import PyEMD
import argparse
from seismic_ml import SeismicClassifier

print(PyEMD.__version__)



print_timestamp("Importing libraries...")
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import matplotlib.ticker as ticker
import os
from matplotlib.colors import LogNorm
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, precision_score, accuracy_score, f1_score
from PyEMD import EMD, EEMD
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix, classification_report)
import joblib

# %%
# =============================================================================
# 2. Data Reading Module
# =============================================================================

# Remove SeismicDataProcessor and SeismicClassifier class definitions

# %%
# =============================================================================
# 3. Feature Extraction Module
# =============================================================================

class PSDFeatureExtractor:
    """Class for extracting PSD features from seismic data"""
    
    def __init__(self, n_bins=50):
        """Initialize the feature extractor"""
        self.n_bins = n_bins
    
    def extract_features(self, data, fs):
        """Extract PSD features from seismic data"""
        nperseg = min(256, len(data))
        
        f, Pxx_e = welch(data[:, 0], fs=fs, nperseg=nperseg, nfft=nperseg*2)
        f, Pxx_n = welch(data[:, 1], fs=fs, nperseg=nperseg, nfft=nperseg*2)
        f, Pxx_z = welch(data[:, 2], fs=fs, nperseg=nperseg, nfft=nperseg*2)
        
        Pxx_e = np.log10(Pxx_e + 1e-10)
        Pxx_n = np.log10(Pxx_n + 1e-10)
        Pxx_z = np.log10(Pxx_z + 1e-10)
        
        freq_bins = np.logspace(np.log10(f[1]), np.log10(f[-1]), self.n_bins)
        
        binned_e = np.interp(freq_bins, f, Pxx_e)
        binned_n = np.interp(freq_bins, f, Pxx_n)
        binned_z = np.interp(freq_bins, f, Pxx_z)
        
        features = np.concatenate([binned_e, binned_n, binned_z])
        
        return features, freq_bins
    
    def process_data(self, eq_data, noise_data):
        """Process all data and extract features"""
        all_features = []
        all_labels = []
        all_targets = []
        
        # Process earthquake data
        for trace_name, (data, fs, mag) in eq_data.items():
            features, _ = self.extract_features(data, fs)
            all_features.append(features)
            all_labels.append(1)  # 1 for earthquake
            all_targets.append(mag)
        
        # Process noise data
        for trace_name, (data, fs) in noise_data.items():
            features, _ = self.extract_features(data, fs)
            all_features.append(features)
            all_labels.append(0)  # 0 for noise
            all_targets.append(0)
        
        return np.array(all_features), np.array(all_labels), np.array(all_targets)

# %%
# =============================================================================
# 4. Visualization Module
# =============================================================================

class FeatureVisualizer:
    """Class for visualizing seismic features"""
    
    def __init__(self, save_dir="figs"):
        """Initialize the visualizer"""
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def plot_psd_features(self, data, fs, features, freq_bins, title="PSD Features"):
        """Plot PSD features for a single trace"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title)
        
        # Plot original signals
        t = np.arange(len(data)) / fs
        components = ['East', 'North', 'Vertical']
        for i, (ax, comp) in enumerate(zip(axes[0], components)):
            ax.plot(t, data[:, i], 'k', lw=0.5)
            ax.set_title(f'{comp} Component')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.grid(True)
        
        # Plot PSD features
        n_bins = len(freq_bins)
        for i, (ax, comp) in enumerate(zip(axes[1], components)):
            start_idx = i * n_bins
            end_idx = (i + 1) * n_bins
            ax.semilogx(freq_bins, features[start_idx:end_idx], 'b-')
            ax.set_title(f'{comp} PSD')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('log10(PSD)')
            ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_emd_features(self, data, fs, imfs, features, n_imfs, title="EMD Features"):
        """Plot EMD features for a single trace"""
        n_imfs_to_plot = min(n_imfs, max(imf.shape[0] for imf in imfs))
        n_features_per_comp = n_imfs_to_plot * 2 + 1  # energies + frequencies + error
        
        fig = plt.figure(figsize=(15, 4*(n_imfs_to_plot + 2)))
        gs = fig.add_gridspec(n_imfs_to_plot + 2, 3)
        fig.suptitle(title)
        
        # Plot original signal and IMFs
        t = np.arange(len(data)) / fs
        components = ['East', 'North', 'Vertical']
        
        for comp_idx, comp in enumerate(components):
            # Original signal
            ax = fig.add_subplot(gs[0, comp_idx])
            ax.plot(t, data[:, comp_idx], 'k', lw=0.5)
            ax.set_title(f'{comp} Component')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.grid(True)
            
            # IMFs
            for i in range(n_imfs_to_plot):
                if i < imfs[comp_idx].shape[0]:
                    ax = fig.add_subplot(gs[i+1, comp_idx])
                    ax.plot(t, imfs[comp_idx][i], 'b', lw=0.5)
                    ax.set_title(f'IMF {i+1}')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Amplitude')
                    ax.grid(True)
        
        # Plot feature distributions
        ax_energy = fig.add_subplot(gs[-1, 0])
        ax_freq = fig.add_subplot(gs[-1, 1])
        ax_error = fig.add_subplot(gs[-1, 2])
        
        # Energy distribution
        for comp_idx, comp in enumerate(components):
            start_idx = comp_idx * n_features_per_comp
            energies = features[start_idx:start_idx + n_imfs_to_plot]
            ax_energy.plot(range(1, n_imfs_to_plot + 1), energies, 'o-', label=comp)
        ax_energy.set_title('Energy Distribution')
        ax_energy.set_xlabel('IMF Index')
        ax_energy.set_ylabel('Normalized Energy')
        ax_energy.legend()
        ax_energy.grid(True)
        
        # Dominant frequencies
        for comp_idx, comp in enumerate(components):
            start_idx = comp_idx * n_features_per_comp + n_imfs_to_plot
            freqs = features[start_idx:start_idx + n_imfs_to_plot]
            ax_freq.semilogy(range(1, n_imfs_to_plot + 1), freqs, 'o-', label=comp)
        ax_freq.set_title('Dominant Frequencies')
        ax_freq.set_xlabel('IMF Index')
        ax_freq.set_ylabel('Frequency (Hz)')
        ax_freq.legend()
        ax_freq.grid(True)
        
        # Reconstruction errors
        errors = [features[comp_idx * n_features_per_comp + 2*n_imfs_to_plot] 
                 for comp_idx in range(3)]
        ax_error.bar(components, errors)
        ax_error.set_title('Reconstruction Error')
        ax_error.set_ylabel('MSE')
        ax_error.grid(True)
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig, filename):
        """Save figure to file"""
        save_path = os.path.join(self.save_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved figure to {save_path}")

# %%
# =============================================================================
# 5. Classification and Regression Module
# =============================================================================

class PSDFeatureTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for PSD feature extraction."""
    def __init__(self, n_bins=50):
        self.n_bins = n_bins
        self.extractor = PSDFeatureExtractor(n_bins=n_bins)
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # X is a list/array of (data, fs) tuples
        features = [self.extractor.extract_features(data, fs)[0] for data, fs in X]
        return np.array(features)

# %%
# =============================================================================
# 6. Main Execution
# =============================================================================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Seismic PSD/EMD Classification")
    parser.add_argument('--mode', type=str, default='fast', choices=['fast', 'prod'], help='Mode: fast or prod')
    parser.add_argument('--nrows', type=int, default=20000, help='Number of rows to load in fast mode')
    parser.add_argument('--chunksize', type=int, default=20000, help='Chunk size to use in fast mode')
    # Add arguments for train/val file paths
    parser.add_argument('--eq_h5', type=str, required=True, help='Earthquake HDF5 file (shared)')
    parser.add_argument('--noise_h5', type=str, required=True, help='Noise HDF5 file (shared)')
    parser.add_argument('--train_eq_csv', type=str, required=True, help='Training earthquake CSV file')
    parser.add_argument('--train_noise_csv', type=str, required=True, help='Training noise CSV file')
    parser.add_argument('--val_eq_csv', type=str, required=True, help='Validation earthquake CSV file')
    parser.add_argument('--val_noise_csv', type=str, required=True, help='Validation noise CSV file')
    args = parser.parse_args()

    # File paths for training and validation
    eq_h5 = args.eq_h5
    noise_h5 = args.noise_h5
    train_csv_file_eq = args.train_eq_csv
    train_csv_file_noise = args.train_noise_csv
    val_csv_file_eq = args.val_eq_csv
    val_csv_file_noise = args.val_noise_csv
    
    # Configuration
    MODE = args.mode
    EQ_FILTERS = {'trace_category': 'earthquake_local'}
    
    print_timestamp("Loading training data...")
    load_start = time.time()
    train_processor = SeismicDataProcessor(
        file_name_eq=eq_h5,
        file_name_noise=noise_h5,
        csv_file_eq=train_csv_file_eq,
        csv_file_noise=train_csv_file_noise,
        mode=MODE,
        eq_filters=EQ_FILTERS,
        nrows=args.nrows if MODE == 'fast' else None,
        chunksize=args.chunksize if MODE == 'fast' else None
    )
    eq_data_train, noise_data_train = train_processor.load_data()
    print_timestamp("Training data loaded successfully.")
    
    print_timestamp("Loading validation data...")
    val_processor = SeismicDataProcessor(
        file_name_eq=eq_h5,
        file_name_noise=noise_h5,
        csv_file_eq=val_csv_file_eq,
        csv_file_noise=val_csv_file_noise,
        mode=MODE,
        eq_filters=EQ_FILTERS,
        nrows=args.nrows if MODE == 'fast' else None,
        chunksize=args.chunksize if MODE == 'fast' else None
    )
    eq_data_val, noise_data_val = val_processor.load_data()
    load_end = time.time()
    print_timestamp(f"Data loading took {load_end - load_start:.2f} seconds.")
    
    # Get current timestamp and dataset size for model filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_size = "all" if train_processor.nrows is None else f"n{train_processor.nrows}"
    
    # Model save paths
    MODEL_DIR = "saved_models"
    PSD_MODEL_PATH = os.path.join(MODEL_DIR, f"{timestamp}_{dataset_size}_psd_classifier.pkl")
    
    print_timestamp("Initializing feature extractor...")
    psd_extractor = PSDFeatureExtractor()       
   
    print_timestamp("Training classifier with PSD features...")
    psd_clf_start = time.time()
    psd_classifier = SeismicClassifier(feature_transformer=PSDFeatureTransformer(n_bins=50))
    psd_metrics = psd_classifier.train(eq_data_train, noise_data_train, eq_data_val, noise_data_val)
    psd_clf_end = time.time()
    print_timestamp("PSD Classification results:")
    for metric, value in psd_metrics.items():
        print_timestamp(f"  {metric}: {value}")
    print_timestamp(f"PSD Classification took {psd_clf_end - psd_clf_start:.2f} seconds.")
    
    # Save PSD model
    psd_classifier.save_pipeline(PSD_MODEL_PATH)
    
    # Print total execution time
    total_time = time.time() - load_start
    print_timestamp(f"Total execution time: {total_time:.2f} seconds.")



