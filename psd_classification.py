# %%
# =============================================================================
# 1. Import Libraries
# =============================================================================

import time
from datetime import datetime
import PyEMD
import argparse

print(PyEMD.__version__)

def print_timestamp(message):
    """Print message with current timestamp"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

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

class SeismicDataProcessor:
    """Class for reading and storing seismic data"""
    
    def __init__(self, file_name_eq, file_name_noise, csv_file_eq, csv_file_noise, 
                 mode='prod', eq_filters=None, nrows=None, chunksize=None):
        """Initialize the data processor"""
        self.file_name_eq = file_name_eq
        self.file_name_noise = file_name_noise
        self.csv_file_eq = csv_file_eq
        self.csv_file_noise = csv_file_noise
        self.mode = mode
        self.eq_filters = eq_filters or {'trace_category': 'earthquake_local'}
        
        # Initialize data storage
        self.eq_data = {}  # {trace_name: (data, fs, magnitude)}
        self.noise_data = {}  # {trace_name: (data, fs)}
        
        # Configure chunking based on mode
        if mode == 'fast':
            self.chunksize = chunksize if chunksize is not None else 20000
            self.nrows = nrows if nrows is not None else 20000
        else:
            self.chunksize = None
            self.nrows = None
    
    def _load_data_chunks(self):
        """Load data chunks from CSV files"""
        needed_columns = ['trace_name', 'trace_category', 'source_magnitude', 'source_distance_km']
        dtype_dict = {
            'trace_name': 'category',
            'trace_category': 'category',
            'source_magnitude': 'float32',
            'source_distance_km': 'float32'
        }
        
        if self.mode == 'fast':
            chunks_eq = pd.read_csv(self.csv_file_eq, chunksize=self.chunksize, nrows=self.nrows, 
                                  usecols=needed_columns, dtype=dtype_dict, low_memory=False)
            chunks_noise = pd.read_csv(self.csv_file_noise, chunksize=self.chunksize, nrows=self.nrows, 
                                     usecols=needed_columns, dtype=dtype_dict)
            total_chunks = min(self.nrows // self.chunksize + (1 if self.nrows % self.chunksize else 0), 
                             len(list(pd.read_csv(self.csv_file_eq, chunksize=self.chunksize, nrows=self.nrows))))
        else:
            chunks_eq = [pd.read_csv(self.csv_file_eq, usecols=needed_columns, dtype=dtype_dict)]
            chunks_noise = [pd.read_csv(self.csv_file_noise, usecols=needed_columns, dtype=dtype_dict)]
            total_chunks = 1
        
        return chunks_eq, chunks_noise, total_chunks
    
    def _filter_data(self, chunk_eq, chunk_noise):
        """Filter earthquake and noise data based on specified criteria"""
        if self.mode == 'fast':
            filtered_eq = chunk_eq[chunk_eq.trace_category == self.eq_filters['trace_category']]
            
            if 'source_distance_km' in self.eq_filters and self.eq_filters['source_distance_km'] is not None:
                filtered_eq = filtered_eq[filtered_eq.source_distance_km <= self.eq_filters['source_distance_km']]
                
            if 'source_magnitude' in self.eq_filters and self.eq_filters['source_magnitude'] is not None:
                mag_min, mag_max = self.eq_filters['source_magnitude']
                filtered_eq = filtered_eq[
                    (filtered_eq.source_magnitude > mag_min) & 
                    (filtered_eq.source_magnitude < mag_max)
                ]
        else:
            filtered_eq = chunk_eq[chunk_eq.trace_category == self.eq_filters['trace_category']]
        
        filtered_noise = chunk_noise[chunk_noise.trace_category == 'noise']
        
        return filtered_eq, filtered_noise, filtered_noise
    
    def load_data(self):
        """Load and store all seismic data"""
        chunks_eq, chunks_noise, total_chunks = self._load_data_chunks()
        
        with tqdm(total=total_chunks, desc=f"Loading chunks ({self.mode} mode)") as pbar:
            for chunk_eq, chunk_noise in zip(chunks_eq, chunks_noise):
                filtered_eq, filtered_noise, _ = self._filter_data(chunk_eq, chunk_noise)
                
                if filtered_eq.empty:
                    print("No events found in this chunk")
                    pbar.update(1)
                    continue
                
                ev_list_eq = filtered_eq['trace_name'].tolist()
                ev_list_noise = filtered_noise['trace_name'].tolist()
                
                with h5py.File(self.file_name_eq, 'r') as dtfl_eq, h5py.File(self.file_name_noise, 'r') as dtfl_noise:
                    # Process earthquake data
                    for trace_name_eq in tqdm(ev_list_eq, desc="Earthquake traces", leave=False):
                        ds_eq = dtfl_eq.get(f"data/{trace_name_eq}")
                        if ds_eq is None:
                            continue
                            
                        data_eq = np.array(ds_eq)
                        fs_eq = float(ds_eq.attrs.get('sampling_rate', 100.0))
                        mag = filtered_eq[filtered_eq.trace_name == trace_name_eq].source_magnitude.iloc[0]
                        
                        self.eq_data[trace_name_eq] = (data_eq, fs_eq, mag)
                    
                    # Process noise data
                    for trace_name_noise in tqdm(ev_list_noise, desc="Noise traces", leave=False):
                        ds_noise = dtfl_noise.get(f"data/{trace_name_noise}")
                        if ds_noise is None:
                            continue
                            
                        data_noise = np.array(ds_noise)
                        fs_noise = float(ds_noise.attrs.get('sampling_rate', 100.0))
                        
                        self.noise_data[trace_name_noise] = (data_noise, fs_noise)
                
                pbar.update(1)
        
        print(f"\nLoaded data:")
        print(f"- {len(self.eq_data)} earthquake traces")
        print(f"- {len(self.noise_data)} noise traces")
        
        return self.eq_data, self.noise_data

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

class SeismicClassifier:
    """Class for seismic signal classification using a pipeline."""
    
    def __init__(self, test_size=0.3, random_state=42, n_bins=50, feature_transformer=None):
        self.test_size = test_size
        self.random_state = random_state
        self.n_bins = n_bins
        self.pipeline = None
        self.training_time = None
        if feature_transformer is not None:
            self.feature_transformer = feature_transformer
        else:
            self.feature_transformer = PSDFeatureTransformer(n_bins=self.n_bins)

    def _prepare_data(self, eq_data, noise_data):
        # Prepare raw input for the pipeline: list of (data, fs), labels, and trace names
        X = []
        y = []
        trace_names = []
        for k, v in eq_data.items():
            X.append((v[0], v[1]))
            y.append(1)
            trace_names.append(k)
        for k, v in noise_data.items():
            X.append((v[0], v[1]))
            y.append(0)
            trace_names.append(k)
        return np.array(X, dtype=object), np.array(y), np.array(trace_names)
    
    def train(self, eq_data_train, noise_data_train, eq_data_val, noise_data_val):
        print_timestamp("Starting training process...")
        
        # Prepare training and validation sets
        X_train, y_train, trace_names_train = self._prepare_data(eq_data_train, noise_data_train)
        X_val, y_val, trace_names_val = self._prepare_data(eq_data_val, noise_data_val)
       
        # Check for overlap
        overlap = set(trace_names_train) & set(trace_names_val)
        print_timestamp(f"Number of overlapping trace names between train and validation: {len(overlap)}")
        if overlap:
            print_timestamp(f"Example overlaps: {list(overlap)[:10]}")
        else:
            print_timestamp("No overlap detected!")

        # Print dataset sizes
        print_timestamp(f"Train set: {len(X_train)} samples ({sum(y_train)} earthquakes, {len(y_train)-sum(y_train)} noise)")
        print_timestamp(f"Validation set: {len(X_val)} samples ({sum(y_val)} earthquakes, {len(y_val)-sum(y_val)} noise)")

        # Initialize the pipeline
        self.pipeline = Pipeline([
            ('feature', self.feature_transformer),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=1000,
                solver='saga',
                verbose=1,
                C=0.01,
                class_weight='balanced',
                n_jobs=-1,
                random_state=self.random_state,
                tol=1e-4,
                penalty='elasticnet',
                l1_ratio=0.5
            ))
        ])

        print_timestamp("Extracting features...")
        # Extract and save features for distribution analysis
        feature_extractor = self.pipeline.named_steps['feature']
        X_train_features = feature_extractor.transform(X_train)
        X_val_features = feature_extractor.transform(X_val)
        np.save('train_features.npy', X_train_features)
        np.save('val_features.npy', X_val_features)
        np.save('train_labels.npy', y_train)
        np.save('val_labels.npy', y_val)

        # Train the pipeline    
        print_timestamp("Training...")
        import time
        start_time = time.time()
        self.pipeline.fit(X_train, y_train)
        self.training_time = time.time() - start_time

        # Evaluate the pipeline
        print_timestamp("Evaluating...")
        y_val_pred = self.pipeline.predict(X_val)
        y_pred_proba = self.pipeline.predict_proba(X_val)[:, 1]
        metrics = self.evaluate(y_val, y_val_pred, y_pred_proba)
        if hasattr(self.pipeline['clf'], 'coef_'):
            coef = self.pipeline['clf'].coef_[0]
            print_timestamp("Top 10 most important features:")
            for idx in np.argsort(np.abs(coef))[-10:]:
                print_timestamp(f"  Feature {idx}: {coef[idx]:.4f}")

        return metrics
    
    def predict(self, X):
        # X: list/array of (data, fs)
        if self.pipeline is None:
            raise ValueError("Pipeline must be trained or loaded before prediction.")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        if self.pipeline is None:
            raise ValueError("Pipeline must be trained or loaded before prediction.")
        return self.pipeline.predict_proba(X)
    
    def save_pipeline(self, path):
        if self.pipeline is None:
            raise ValueError("No trained pipeline to save")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        print_timestamp(f"Pipeline saved to {path}")

    def load_pipeline(self, path):
        self.pipeline = joblib.load(path)
        print_timestamp(f"Pipeline loaded from {path}")

    @staticmethod
    def evaluate(y_true, y_pred, y_pred_proba):
        results = {}
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred)
        results['recall'] = recall_score(y_true, y_pred)
        results['f1'] = f1_score(y_true, y_pred)
        results['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        results['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        results['classification_report'] = classification_report(y_true, y_pred, digits=3)
        return results


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



