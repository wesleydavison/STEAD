# %%
# =============================================================================
# 1. Import Libraries
# =============================================================================
import time
from datetime import datetime

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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, precision_score, accuracy_score, f1_score

# %%
# =============================================================================
# 2. Directory Setup
# =============================================================================
print_timestamp("Setting up directories...")
# Create figs directory if it doesn't exist
figs_dir = "figs"
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)

# %%
# =============================================================================
# 3. File Paths and Parameters
# =============================================================================
print_timestamp("Setting up file paths and parameters...")
# File paths
file_name_eq = r"/users/230442014/archive/STEAD_dataset/chunk2.hdf5"
csv_file_eq = r"/users/230442014/archive/STEAD_dataset/chunk2.csv"

file_name_noise = r"/users/230442014/archive/STEAD_dataset/chunk1.hdf5"
csv_file_noise = r"/users/230442014/archive/STEAD_dataset/chunk1.csv"

# %%
# =============================================================================
# 4. Processing Mode Configuration
# =============================================================================
print_timestamp("Configuring processing mode...")
# Processing mode
MODE = 'prod'  # 'test' or 'prod'

if MODE == 'test':
    chunksize = 1000  # Smaller chunk size for testing
    nrows = 30     # Limit number of rows for testing
    
else:  # prod mode
    chunksize = None  # No chunking in production mode
    nrows = None     # No row limit
    # Production mode filters (minimal or none)  

# %%
# =============================================================================
# 5. Data Loading and Chunking
# =============================================================================
print_timestamp("Loading data...")
# Initialize data readers
if MODE == 'test':
    chunks_eq = pd.read_csv(csv_file_eq, chunksize=chunksize, nrows=nrows)
    chunks_noise = pd.read_csv(csv_file_noise, chunksize=chunksize, nrows=nrows)
    total_chunks = min(nrows // chunksize + (1 if nrows % chunksize else 0), 
                      len(list(pd.read_csv(csv_file_eq, chunksize=chunksize, nrows=nrows))))
else:
    # Load entire datasets in production mode
    print("Loading full datasets...")
    chunks_eq = [pd.read_csv(csv_file_eq)]
    chunks_noise = [pd.read_csv(csv_file_noise)]
    total_chunks = 1

# %%
# =============================================================================
# 6. Data Filtering Configuration
# =============================================================================
print_timestamp("Setting up data filtering...")
EQ_FILTERS = {
        'trace_category': 'earthquake_local' #,
        #'source_distance_km': 20,  # <= 20 km
        #'source_magnitude': (0, 3)  # between 1 and 3
    }

# EQ_FILTERS = {
#         'trace_category': 'earthquake_local',
#         'source_distance_km': None,  # No distance limit
#         'source_magnitude': None     # No magnitude limit
#     }

# %%
# =============================================================================
# 7. Plotting Configuration
# =============================================================================
print_timestamp("Configuring plotting settings...")
# Plotting parameters
plotting = True     # Toggle all plotting on/off
quiet = True    # If True, skip waveform+PSD plots
plot_aggregated_only = True  # If True, only plot aggregated PSD, skip individual waveform plots

# %%
# =============================================================================
# 8. Data Storage Initialization
# =============================================================================
print_timestamp("Initializing data storage...")
# Initialize storage variables
max_amp_e_eq = []
max_amp_n_eq = []
max_amp_z_eq = []
max_amp_e_noise = []
max_amp_n_noise = []
max_amp_z_noise = []
psd_data_eq = []  # [(f, Pxx_e, Pxx_n, Pxx_z), ...]
psd_data_noise = []  # [(f, Pxx_e, Pxx_n, Pxx_z), ...]



# Dictionary to store noise PSDs for reuse
noise_psd_cache = {}  # {trace_name: (f, Pxx_e, Pxx_n, Pxx_z)}

# %%
# =============================================================================
# 9. Data Processing Functions
# =============================================================================
print_timestamp("Setting up data processing functions...")
def filter_data(chunk_eq, chunk_noise, mode='test', eq_filters=None):
    """Filter earthquake and noise data based on specified criteria
    
    Parameters:
    -----------
    chunk_eq : pandas.DataFrame
        Chunk of earthquake data
    chunk_noise : pandas.DataFrame
        Chunk of noise data
    mode : str
        Processing mode ('test' or 'prod')
    eq_filters : dict
        Dictionary containing earthquake filtering criteria
        
    Returns:
    --------
    tuple
        (filtered_eq_chunk, filtered_noise_chunk, all_noise_data)
    """
    print(f"\nFiltering data:")
    print(f"Total earthquake records in chunk: {len(chunk_eq)}")
    print(f"Total noise records in chunk: {len(chunk_noise)}")
    
    if mode == 'test':
        # Start with the base filter for trace category
        filtered_eq = chunk_eq[chunk_eq.trace_category == eq_filters['trace_category']]
        
        # Apply additional filters if they exist
        if 'source_distance_km' in eq_filters and eq_filters['source_distance_km'] is not None:
            filtered_eq = filtered_eq[filtered_eq.source_distance_km <= eq_filters['source_distance_km']]
            print(f"Distance range: 0-{eq_filters['source_distance_km']} km")
            
        if 'source_magnitude' in eq_filters and eq_filters['source_magnitude'] is not None:
            mag_min, mag_max = eq_filters['source_magnitude']
            filtered_eq = filtered_eq[
                (filtered_eq.source_magnitude > mag_min) & 
                (filtered_eq.source_magnitude < mag_max)
            ]
            print(f"Magnitude range: {mag_min}-{mag_max}")
            
        print(f"Filtered earthquake records: {len(filtered_eq)}")
    else:
        filtered_eq = chunk_eq[chunk_eq.trace_category == eq_filters['trace_category']]
    
    # Get all noise data for aggregated PSD
    all_noise_data = chunk_noise[chunk_noise.trace_category == 'noise']
    print(f"Total noise records after filtering: {len(all_noise_data)}")
    
    # Sample noise data to match earthquake data size for individual comparisons
    filtered_noise = all_noise_data.sample(n=len(filtered_eq), random_state=42) if not filtered_eq.empty else all_noise_data
    print(f"Sampled noise records for comparison: {len(filtered_noise)}")
    
    return filtered_eq, filtered_noise, all_noise_data

def compute_psd(data, fs):
    """Compute PSD using classical periodogram method
    
    Parameters:
    -----------
    data : numpy.ndarray
        Seismic data array with shape (n_samples, 3) for E, N, Z components
    fs : float
        Sampling frequency
        
    Returns:
    --------
    tuple
        (frequencies, PSD_E, PSD_N, PSD_Z)
    """
    n = data.shape[0]
    
    # Compute FFT for each component
    fft_e = np.fft.rfft(data[:, 0])
    fft_n = np.fft.rfft(data[:, 1])
    fft_z = np.fft.rfft(data[:, 2])
    
    # Compute frequencies
    f = np.fft.rfftfreq(n, 1/fs)
    
    # Compute PSD (periodogram)
    Pxx_e = np.abs(fft_e)**2 / (fs * n)
    Pxx_n = np.abs(fft_n)**2 / (fs * n)
    Pxx_z = np.abs(fft_z)**2 / (fs * n)
    
    return f, Pxx_e, Pxx_n, Pxx_z

def extract_psd_features(data, fs):
    """Extract PSD features from seismic data for machine learning
    
    Parameters:
    -----------
    data : numpy.ndarray
        Seismic data array (n_samples, 3)
    fs : float
        Sampling frequency
        
    Returns:
    --------
    tuple
        (features, freqs) where features is the processed PSD data
    """
    # Compute PSD using classical periodogram method
    f, Pxx_e, Pxx_n, Pxx_z = compute_psd(data, fs)
    
    # Combine features from all components
    features = np.concatenate([Pxx_e, Pxx_n, Pxx_z])
    
    return features, f

def plot_psd_comparison(data, fs, title, figs_dir, trace_name):
    """Plot PSD using periodogram method
    
    Parameters:
    -----------
    data : numpy.ndarray
        Seismic data array
    fs : float
        Sampling frequency
    title : str
        Plot title
    figs_dir : str
        Directory to save figures
    trace_name : str
        Name of the trace for the filename
    """
    # Compute PSD using periodogram
    f, Pxx_e, Pxx_n, Pxx_z = compute_psd(data, fs)
    
    # Process PSD for LDA
    features, _ = extract_psd_features(data, fs)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Periodogram - Linear scale
    ax1.plot(f, Pxx_e, label='E component', color='C0')
    ax1.plot(f, Pxx_n, label='N component', color='C1')
    ax1.plot(f, Pxx_z, label='Z component', color='C2')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('PSD (counts²/Hz)')
    ax1.set_title('Periodogram - Linear Scale')
    ax1.grid(True, which='both', ls='--', lw=0.5)
    ax1.legend()
    
    # Periodogram - Log scale
    ax2.loglog(f, Pxx_e, label='E component', color='C0')
    ax2.loglog(f, Pxx_n, label='N component', color='C1')
    ax2.loglog(f, Pxx_z, label='Z component', color='C2')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('PSD (counts²/Hz)')
    ax2.set_title('Periodogram - Log Scale')
    ax2.grid(True, which='both', ls='--', lw=0.5)
    ax2.legend()
    
    # Add overall title
    fig.suptitle(f'{title}\nSignal length: {len(data)} samples, Sampling rate: {fs} Hz\nFeature vector length: {len(features)}', y=1.02)
    plt.tight_layout()
    
    # Save the figure
    fig_path = os.path.join(figs_dir, f'psd_periodogram_{trace_name}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Print statistics
    print(f"\nPSD Statistics for {trace_name}:")
    print(f"Frequency points: {len(f)}, Resolution: {f[1]-f[0]:.2f} Hz")
    print(f"Feature vector length for LDA: {len(features)}")



# %%
# =============================================================================
# 11. Main Processing Loop
# =============================================================================
print_timestamp("Starting main processing loop...")
# Initialize lists for features, labels, trace names, and regression targets
all_features = []
all_labels = []
all_trace_names = []
all_targets = []  # For regression: Y = 0 for noise, Y = source_magnitude for earthquake

# Create progress bar for chunks
with tqdm(total=total_chunks, desc=f"Processing chunks ({MODE} mode)") as pbar:
    for chunk_eq, chunk_noise in zip(chunks_eq, chunks_noise):
        # Filter data
        filtered_eq, filtered_noise, all_noise_data = filter_data(
            chunk_eq, chunk_noise, mode=MODE, eq_filters=EQ_FILTERS
        )
        
        if filtered_eq.empty:
            print("No events found in this chunk")
            pbar.update(1)
            continue

        ev_list_eq = filtered_eq['trace_name'].tolist()
        ev_list_noise = filtered_noise['trace_name'].tolist()

        # Process both earthquake and noise data
        with h5py.File(file_name_eq, 'r') as dtfl_eq, h5py.File(file_name_noise, 'r') as dtfl_noise:
            # Process earthquake data
            print("\nProcessing earthquake data:")
            for trace_name_eq in tqdm(ev_list_eq, desc="Earthquake traces", leave=False):
                ds_eq = dtfl_eq.get(f"data/{trace_name_eq}")
                
                if ds_eq is None:
                    continue

                # Process earthquake data
                data_eq = np.array(ds_eq)
                fs_eq = float(ds_eq.attrs.get('sampling_rate', 100.0))
                
                # Process PSD for regression
                features, _ = extract_psd_features(data_eq, fs_eq)
                all_features.append(features)
                all_labels.append(1)  # 1 for earthquake
                all_trace_names.append(trace_name_eq)
                # Regression target: source_magnitude
                mag = filtered_eq[filtered_eq.trace_name == trace_name_eq].source_magnitude.iloc[0]
                all_targets.append(mag)
                
                # Store PSD data for aggregated plots
                f, Pxx_e, Pxx_n, Pxx_z = compute_psd(data_eq, fs_eq)
         
                # Plot PSD comparison
                if not quiet:
                    plot_psd_comparison(data_eq, fs_eq, 
                                      f'Earthquake PSD Comparison\n{trace_name_eq}', 
                                      figs_dir, trace_name_eq)

            # Process noise data
            print("\nProcessing noise data:")
            for trace_name_noise in tqdm(ev_list_noise, desc="Noise traces", leave=False):
                ds_noise = dtfl_noise.get(f"data/{trace_name_noise}")
                
                if ds_noise is None:
                    continue

                # Process noise data
                data_noise = np.array(ds_noise)
                fs_noise = float(ds_noise.attrs.get('sampling_rate', 100.0))
                
                # Process PSD for regression
                features, _ = extract_psd_features(data_noise, fs_noise)
                all_features.append(features)
                all_labels.append(0)  # 0 for noise
                all_trace_names.append(trace_name_noise)
                # Regression target: 0 for noise
                all_targets.append(0)
                
                # Store PSD data for aggregated plots
                f, Pxx_e, Pxx_n, Pxx_z = compute_psd(data_noise, fs_noise)

                
                # Plot PSD comparison
                if not quiet:
                    plot_psd_comparison(data_noise, fs_noise, 
                                      f'Noise PSD Comparison\n{trace_name_noise}', 
                                      figs_dir, trace_name_noise)

        # Update chunk progress bar
        pbar.update(1)

print(f"\nCollected {len(all_features)} total features:")
print(f"- {sum(all_labels)} earthquake features")
print(f"- {len(all_labels) - sum(all_labels)} noise features")

# Classification on PSD features using logistic regression with train/test split
if len(all_features) > 0:
    print_timestamp("Starting classification...")
    X = np.array(all_features)
    Y = np.array(all_targets)

    # Binary class labels: 1 for earthquake, 0 for noise
    Y_class = (Y > 0).astype(int)

    # Print data dimensions
    print_timestamp("\nData dimensions:")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Class distribution: {np.bincount(Y_class)}")

    # Split into train and test sets (70% train, 30% test)
    print_timestamp("Splitting into train and test sets...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_class, test_size=0.3, random_state=42, stratify=Y_class
    )

    from sklearn.linear_model import LogisticRegression

    # Start timing
    print_timestamp("Training model...")
    start_time = time.time()
    
    # Initialize and train the model with increased max_iter and better solver
    clf = LogisticRegression(
        max_iter=5000,  # Increased from default 1000
        solver='saga',  # More robust solver
        n_jobs=-1,      # Use all available cores
        random_state=42
    )
    clf.fit(X_train, Y_train)

    # End training time
    training_time = time.time() - start_time

    # Start prediction timing
    print_timestamp("Predicting on test data...")
    pred_start_time = time.time()
    
    # Predict on test data
    Y_pred_class = clf.predict(X_test)
    
    # End prediction time
    prediction_time = time.time() - pred_start_time

    # Classification metrics
    precision = precision_score(Y_test, Y_pred_class)
    accuracy = accuracy_score(Y_test, Y_pred_class)
    f1 = f1_score(Y_test, Y_pred_class)
    print_timestamp("\nLogistic Regression Classification Results (Test Set):")
    print(f"Test Precision: {precision}")
    print(f"Test Accuracy: {accuracy}")
    print(f"Test F1 Score: {f1}")
    print_timestamp("\nTiming Results:")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Prediction time: {prediction_time:.2f} seconds")
    print(f"Total time: {training_time + prediction_time:.2f} seconds")
    print_timestamp("Classification completed.")
else:
    print_timestamp("No features collected for classification analysis. Check if data filtering is too restrictive.")



