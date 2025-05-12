import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD
import os
from tqdm import tqdm

# Create figs directory if it doesn't exist
figs_dir = "figs"
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)

# %% File paths and parameters
file_name_eq = r"/users/230442014/archive/STEAD_dataset/chunk2.hdf5"
csv_file_eq = r"/users/230442014/archive/STEAD_dataset/chunk2.csv"

file_name_noise = r"/users/230442014/archive/STEAD_dataset/chunk1.hdf5"
csv_file_noise = r"/users/230442014/archive/STEAD_dataset/chunk1.csv"

# Processing mode
MODE = 'test'  # 'test' or 'prod'
if MODE == 'test':
    chunksize = 1000  # Smaller chunk size for testing
    nrows = 1000     # Limit number of rows for testing
    # Test mode filters
    EQ_FILTERS = {
        'trace_category': 'earthquake_local',
        'source_distance_km': 20,  # <= 20 km
        'source_magnitude': (1, 3)  # between 1 and 3
    }
else:  # prod mode
    chunksize = 100000  # Larger chunk size for production
    nrows = None       # No row limit
    # Production mode filters (minimal or none)
    EQ_FILTERS = {
        'trace_category': 'earthquake_local',
        'source_distance_km': None,  # No distance limit
        'source_magnitude': None     # No magnitude limit
    }

plotting = True     # Toggle all plotting on/off
quiet = False    # If True, skip waveform+IMF plots

# %% Storage
all_eq_imfs = []  # Store IMFs for earthquakes
all_noise_imfs = []  # Store IMFs for noise
all_eq_traces = []  # Store trace names for earthquakes
all_noise_traces = []  # Store trace names for noise
all_eq_info = []  # Store magnitude and distance info

def plot_emd_comparison(t_eq, data_eq, imfs_eq, t_noise, data_noise, imfs_noise, 
                       trace_name_eq, trace_name_noise, comp_name, eq_info=None):
    """Plot EMD results for earthquake and noise side by side"""
    n_imfs = max(len(imfs_eq), len(imfs_noise))
    fig, axes = plt.subplots(n_imfs + 1, 2, figsize=(20, 2*(n_imfs + 1)), dpi=150)
    
    # Plot original signals
    axes[0, 0].plot(t_eq, data_eq, 'k', lw=0.5)
    eq_title = f'Earthquake - {comp_name} component\n{trace_name_eq}'
    if eq_info:
        eq_title += f'\nM{eq_info["magnitude"]:.1f} @ {eq_info["distance"]:.1f}km'
    axes[0, 0].set_title(eq_title)
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(t_noise, data_noise, 'k', lw=0.5)
    axes[0, 1].set_title(f'Noise - {comp_name} component\n{trace_name_noise}')
    axes[0, 1].grid(True)
    
    # Plot IMFs
    for i in range(n_imfs):
        # Earthquake IMFs
        if i < len(imfs_eq):
            axes[i+1, 0].plot(t_eq, imfs_eq[i], 'k', lw=0.5)
        axes[i+1, 0].set_title(f'IMF {i+1}')
        axes[i+1, 0].grid(True)
        
        # Noise IMFs
        if i < len(imfs_noise):
            axes[i+1, 1].plot(t_noise, imfs_noise[i], 'k', lw=0.5)
        axes[i+1, 1].set_title(f'IMF {i+1}')
        axes[i+1, 1].grid(True)
    
    plt.tight_layout()
    return fig

def process_chunk(chunk_eq, chunk_noise, dtfl_eq, dtfl_noise):
    """Process a chunk of data for EMD analysis"""
    # Filter earthquake data based on mode
    if MODE == 'test':
        chunk_eq = chunk_eq[
            (chunk_eq.trace_category == EQ_FILTERS['trace_category']) &
            (chunk_eq.source_distance_km <= EQ_FILTERS['source_distance_km']) &
            (chunk_eq.source_magnitude > EQ_FILTERS['source_magnitude'][0]) &
            (chunk_eq.source_magnitude < EQ_FILTERS['source_magnitude'][1])
        ]
    else:
        chunk_eq = chunk_eq[chunk_eq.trace_category == EQ_FILTERS['trace_category']]
    
    # Get all noise data
    all_noise_data = chunk_noise[chunk_noise.trace_category == 'noise']
    
    # Sample noise data to match earthquake data size for individual comparisons
    chunk_noise = all_noise_data.sample(n=len(chunk_eq), random_state=42) if not chunk_eq.empty else all_noise_data
    
    if chunk_eq.empty:
        print("No events found in this chunk")
        return

    ev_list_eq = chunk_eq['trace_name'].tolist()
    ev_list_noise = chunk_noise['trace_name'].tolist()

    # Initialize EMD with default settings
    emd = EMD()
    
    # Process both earthquake and noise data
    for trace_name_eq, trace_name_noise in tqdm(zip(ev_list_eq, ev_list_noise), 
                                              total=len(ev_list_eq),
                                              desc="Processing data",
                                              leave=False):
        # Process earthquake data
        ds_eq = dtfl_eq.get(f"data/{trace_name_eq}")
        if ds_eq is None:
            continue
            
        data_eq = np.array(ds_eq)
        fs_eq = float(ds_eq.attrs.get('sampling_rate', 100.0))
        n_eq = data_eq.shape[0]
        t_eq = np.arange(n_eq) / fs_eq
        
        # Process noise data
        ds_noise = dtfl_noise.get(f"data/{trace_name_noise}")
        if ds_noise is None:
            continue
            
        data_noise = np.array(ds_noise)
        fs_noise = float(ds_noise.attrs.get('sampling_rate', 100.0))
        n_noise = data_noise.shape[0]
        t_noise = np.arange(n_noise) / fs_noise
        
        # Calculate IMFs for each component
        for comp_idx, comp_name in enumerate(['E', 'N', 'Z']):
            # Earthquake IMFs
            imfs_eq = emd(data_eq[:, comp_idx])
            all_eq_imfs.append(imfs_eq)
            all_eq_traces.append(trace_name_eq)
            eq_info = {
                'magnitude': chunk_eq[chunk_eq['trace_name'] == trace_name_eq]['source_magnitude'].iloc[0],
                'distance': chunk_eq[chunk_eq['trace_name'] == trace_name_eq]['source_distance_km'].iloc[0],
                'component': comp_name
            }
            all_eq_info.append(eq_info)
            
            # Noise IMFs
            imfs_noise = emd(data_noise[:, comp_idx])
            all_noise_imfs.append(imfs_noise)
            all_noise_traces.append(trace_name_noise)
            
            if not quiet and plotting:
                # Create side-by-side comparison plot
                fig = plot_emd_comparison(t_eq, data_eq[:, comp_idx], imfs_eq,
                                        t_noise, data_noise[:, comp_idx], imfs_noise,
                                        trace_name_eq, trace_name_noise, comp_name, eq_info)
                
                # Save the figure
                save_path = os.path.join(figs_dir, f'emd_comparison_{trace_name_eq}_{comp_name}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                # Show the plot and wait for user input
                plt.show()
                user_input = input("Press Enter to continue to next comparison (or 'q' to quit): ")
                if user_input.lower() == 'q':
                    print("Exiting program...")
                    exit()

# %% Main processing loop
chunks_eq = pd.read_csv(csv_file_eq, chunksize=chunksize, nrows=nrows)
chunks_noise = pd.read_csv(csv_file_noise, chunksize=chunksize, nrows=nrows)

# Calculate total number of chunks
if MODE == 'test':
    total_chunks = min(nrows // chunksize + (1 if nrows % chunksize else 0), 
                      len(list(pd.read_csv(csv_file_eq, chunksize=chunksize, nrows=nrows))))
else:
    # For production, estimate total chunks from file size
    file_size = os.path.getsize(csv_file_eq)
    estimated_rows = file_size / 1000  # Rough estimate: 1KB per row
    total_chunks = (estimated_rows + chunksize - 1) // chunksize

# Create progress bar for chunks
with tqdm(total=total_chunks, desc=f"Processing chunks ({MODE} mode)") as pbar:
    with h5py.File(file_name_eq, 'r') as dtfl_eq, h5py.File(file_name_noise, 'r') as dtfl_noise:
        for chunk_eq, chunk_noise in zip(chunks_eq, chunks_noise):
            process_chunk(chunk_eq, chunk_noise, dtfl_eq, dtfl_noise)
            pbar.update(1)

print(f"Processed {len(all_eq_imfs)} earthquake records and {len(all_noise_imfs)} noise records")
print(f"Results saved in {figs_dir} directory") 