# %%
# =============================================================================
# 1. Import Libraries
# =============================================================================

import time
from datetime import datetime
import PyEMD

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
from PyEMD import EMD
from scipy.stats import pearsonr

# %%
# =============================================================================
# 2. Data Reading Module
# =============================================================================

class SeismicDataProcessor:
    """Class for reading and storing seismic data"""
    
    def __init__(self, file_name_eq, file_name_noise, csv_file_eq, csv_file_noise, 
                 mode='prod', eq_filters=None):
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
        if mode == 'test':
            self.chunksize = 10000
            self.nrows = 2
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
        
        if self.mode == 'test':
            chunks_eq = pd.read_csv(self.csv_file_eq, chunksize=self.chunksize, nrows=self.nrows, 
                                  usecols=needed_columns, dtype=dtype_dict)
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
        if self.mode == 'test':
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
        
        all_noise_data = chunk_noise[chunk_noise.trace_category == 'noise']
        filtered_noise = all_noise_data.sample(n=len(filtered_eq), random_state=42) if not filtered_eq.empty else all_noise_data
        
        return filtered_eq, filtered_noise, all_noise_data
    
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

class EMDFeatureExtractor:
    """Class for extracting features using Empirical Mode Decomposition"""
    
    def __init__(self, n_imfs=None):
        """Initialize the EMD feature extractor"""
        self.n_imfs = n_imfs
        self.emd = EMD()
    
    def _calculate_imf_energy(self, imfs):
        """Calculate energy distribution across IMFs"""
        energies = np.array([np.sum(imf**2) for imf in imfs])
        total_energy = np.sum(energies)
        return energies / total_energy  # Normalized energy distribution
    
    def _calculate_imf_frequency(self, imf, fs):
        """Calculate dominant frequency of an IMF using Welch's method"""
        f, Pxx = welch(imf, fs=fs, nperseg=min(256, len(imf)))
        return f[np.argmax(Pxx)]  # Return frequency with maximum power
    
    def _calculate_reconstruction_error(self, original, imfs):
        """Calculate reconstruction error"""
        reconstructed = np.sum(imfs, axis=0)
        return np.mean((original - reconstructed)**2)
    
    def extract_features(self, data, fs):
        """Extract EMD features from seismic data"""
        features = []
        
        # Process each component (E, N, Z)
        for comp_idx in range(3):
            # Get IMFs for this component
            imfs = self.emd(data[:, comp_idx])
            
            # Limit number of IMFs if specified
            if self.n_imfs is not None:
                imfs = imfs[:self.n_imfs]
            
            # Calculate features for this component
            # 1. Energy distribution
            energy_dist = self._calculate_imf_energy(imfs)
            features.extend(energy_dist)
            
            # 2. Dominant frequencies
            freqs = [self._calculate_imf_frequency(imf, fs) for imf in imfs]
            features.extend(freqs)
            
            # 3. Reconstruction error
            error = self._calculate_reconstruction_error(data[:, comp_idx], imfs)
            features.append(error)
        
        return np.array(features)
    
    def process_data(self, eq_data, noise_data):
        """Process all data and extract EMD features"""
        all_features = []
        all_labels = []
        all_targets = []
        
        # Process earthquake data
        for trace_name, (data, fs, mag) in eq_data.items():
            features = self.extract_features(data, fs)
            all_features.append(features)
            all_labels.append(1)  # 1 for earthquake
            all_targets.append(mag)
        
        # Process noise data
        for trace_name, (data, fs) in noise_data.items():
            features = self.extract_features(data, fs)
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

class SeismicClassifier:
    """Class for seismic signal classification"""
    
    def __init__(self, test_size=0.3, random_state=42):
        """Initialize the classifier"""
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.clf = None
        self.training_time = None
        
    def _prepare_data(self, features, labels):
        """Prepare data for training and testing"""
        X_train, X_test, Y_train, Y_test = train_test_split(
            features, labels, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=labels
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, Y_train, Y_test
    
    def train(self, features, labels):
        """Train the classifier"""
        X_train_scaled, X_test_scaled, Y_train, Y_test = self._prepare_data(features, labels)
        
        self.clf = LogisticRegression(
            max_iter=100,
            solver='saga',
            C=0.1,
            class_weight='balanced',
            n_jobs=-1,
            random_state=self.random_state,
            tol=1e-3
        )
        
        start_time = time.time()
        self.clf.fit(X_train_scaled, Y_train)
        self.training_time = time.time() - start_time
        
        Y_pred = self.clf.predict(X_test_scaled)
        precision = precision_score(Y_test, Y_pred)
        accuracy = accuracy_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred)
        
        return precision, accuracy, f1
    
    def predict(self, features):
        """Make predictions on new data"""
        if self.clf is None:
            raise ValueError("Model must be trained before making predictions")
            
        features_scaled = self.scaler.transform(features)
        return self.clf.predict(features_scaled)
    
    def get_training_time(self):
        """Get the training time in seconds"""
        return self.training_time

class SeismicRegressor:
    """Class for seismic magnitude regression"""
    
    def __init__(self, test_size=0.3, random_state=42):
        """Initialize the regressor"""
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.reg = None
        self.training_time = None
        
    def _prepare_data(self, features, targets):
        """Prepare data for training and testing"""
        X_train, X_test, Y_train, Y_test = train_test_split(
            features, targets,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, Y_train, Y_test
    
    def train(self, features, targets):
        """Train the regressor"""
        X_train_scaled, X_test_scaled, Y_train, Y_test = self._prepare_data(features, targets)
        
        self.reg = LinearRegression(n_jobs=-1)
        
        start_time = time.time()
        self.reg.fit(X_train_scaled, Y_train)
        self.training_time = time.time() - start_time
        
        Y_pred = self.reg.predict(X_test_scaled)
        r2 = r2_score(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        
        return r2, mae
    
    def predict(self, features):
        """Make predictions on new data"""
        if self.reg is None:
            raise ValueError("Model must be trained before making predictions")
            
        features_scaled = self.scaler.transform(features)
        return self.reg.predict(features_scaled)
    
    def get_training_time(self):
        """Get the training time in seconds"""
        return self.training_time

# %%
# =============================================================================
# 6. Main Execution
# =============================================================================

if __name__ == "__main__":
    # File paths
    file_name_eq = r"/users/230442014/archive/STEAD_dataset/chunk2.hdf5"
    csv_file_eq = r"/users/230442014/archive/STEAD_dataset/chunk2.csv"
    file_name_noise = r"/users/230442014/archive/STEAD_dataset/chunk1.hdf5"
    csv_file_noise = r"/users/230442014/archive/STEAD_dataset/chunk1.csv"
    
    # Configuration
    MODE = 'test'
    EQ_FILTERS = {'trace_category': 'earthquake_local'}
    
    # Load data
    processor = SeismicDataProcessor(
        file_name_eq=file_name_eq,
        file_name_noise=file_name_noise,
        csv_file_eq=csv_file_eq,
        csv_file_noise=csv_file_noise,
        mode=MODE,
        eq_filters=EQ_FILTERS
    )
    eq_data, noise_data = processor.load_data()
    
    # Initialize feature extractors and visualizer
    psd_extractor = PSDFeatureExtractor()
    emd_extractor = EMDFeatureExtractor(n_imfs=5)
    visualizer = FeatureVisualizer()
    
    # Visualize features for one earthquake and one noise trace
    print("\nVisualizing features...")
    
    # Get first earthquake and noise trace
    eq_trace = next(iter(eq_data.items()))
    noise_trace = next(iter(noise_data.items()))
    
    # PSD visualization
    eq_trace_data, eq_fs, _ = eq_trace[1]
    eq_features, eq_freq_bins = psd_extractor.extract_features(eq_trace_data, eq_fs)
    fig_psd_eq = visualizer.plot_psd_features(
        eq_trace_data, eq_fs, eq_features, eq_freq_bins,
        title="PSD Features - Earthquake"
    )
    visualizer.save_figure(fig_psd_eq, "psd_features_earthquake.png")
    
    noise_trace_data, noise_fs = noise_trace[1]
    noise_features, noise_freq_bins = psd_extractor.extract_features(noise_trace_data, noise_fs)
    fig_psd_noise = visualizer.plot_psd_features(
        noise_trace_data, noise_fs, noise_features, noise_freq_bins,
        title="PSD Features - Noise"
    )
    visualizer.save_figure(fig_psd_noise, "psd_features_noise.png")
    
    # EMD visualization
    eq_imfs = []
    for i, comp in enumerate(['East', 'North', 'Vertical']):
        print(f"{comp} data shape: {eq_trace_data[:, i].shape}")
        imfs = emd_extractor.emd(eq_trace_data[:, i])
        print(f"{comp} IMFs: {imfs.shape}")
        eq_imfs.append(imfs[:emd_extractor.n_imfs])  # Limit to n_imfs

    eq_features = emd_extractor.extract_features(eq_trace_data, eq_fs)
    fig_emd_eq = visualizer.plot_emd_features(
        eq_trace_data, eq_fs, eq_imfs, eq_features, emd_extractor.n_imfs,
        title="EMD Features - Earthquake"
    )
    visualizer.save_figure(fig_emd_eq, "emd_features_earthquake.png")
    
    noise_imfs = [emd_extractor.emd(noise_trace_data[:, i])[:emd_extractor.n_imfs] for i in range(3)]
    noise_features = emd_extractor.extract_features(noise_trace_data, noise_fs)
    fig_emd_noise = visualizer.plot_emd_features(
        noise_trace_data, noise_fs, noise_imfs, noise_features, emd_extractor.n_imfs,
        title="EMD Features - Noise"
    )
    visualizer.save_figure(fig_emd_noise, "emd_features_noise.png")
    
    # Extract features for all data
    print("\nExtracting PSD features...")
    psd_features, psd_labels, psd_targets = psd_extractor.process_data(eq_data, noise_data)
    
    print("\nExtracting EMD features...")
    emd_features, emd_labels, emd_targets = emd_extractor.process_data(eq_data, noise_data)
    
    # Classification with PSD features
    print("\nTraining classifier with PSD features...")
    psd_classifier = SeismicClassifier()
    psd_precision, psd_accuracy, psd_f1 = psd_classifier.train(psd_features, psd_labels)
    print(f"PSD Classification results: Precision={psd_precision:.3f}, Accuracy={psd_accuracy:.3f}, F1={psd_f1:.3f}")
    
    # Classification with EMD features
    print("\nTraining classifier with EMD features...")
    emd_classifier = SeismicClassifier()
    emd_precision, emd_accuracy, emd_f1 = emd_classifier.train(emd_features, emd_labels)
    print(f"EMD Classification results: Precision={emd_precision:.3f}, Accuracy={emd_accuracy:.3f}, F1={emd_f1:.3f}")



