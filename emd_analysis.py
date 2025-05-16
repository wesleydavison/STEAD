import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD
import os
from tqdm import tqdm
from scipy.signal import welch
from scipy.stats import pearsonr
from matplotlib.patches import Patch

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
MODE = 'prod'  # 'test' or 'prod'
if MODE == 'test':
    chunksize = 100  # Smaller chunk size for testing
    nrows = 100     # Limit number of rows for testing
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
quiet = True    # If True, skip waveform+IMF plots

# %% Storage
all_eq_imfs = []  # Store IMFs for earthquakes
all_noise_imfs = []  # Store IMFs for noise
all_eq_traces = []  # Store trace names for earthquakes
all_noise_traces = []  # Store trace names for noise
all_eq_info = []  # Store magnitude and distance info

def calculate_imf_energy(imfs):
    """Calculate energy distribution across IMFs"""
    energies = np.array([np.sum(imf**2) for imf in imfs])
    total_energy = np.sum(energies)
    return energies / total_energy  # Normalized energy distribution

def calculate_imf_frequency(imf, fs):
    """Calculate dominant frequency of an IMF using Welch's method"""
    f, Pxx = welch(imf, fs=fs, nperseg=min(256, len(imf)))
    return f[np.argmax(Pxx)]  # Return frequency with maximum power

def calculate_imf_correlation(imfs_eq, imfs_noise):
    """Calculate correlation between corresponding IMFs"""
    n_imfs = min(len(imfs_eq), len(imfs_noise))
    correlations = []
    for i in range(n_imfs):
        # Ensure same length for correlation
        min_len = min(len(imfs_eq[i]), len(imfs_noise[i]))
        corr, _ = pearsonr(imfs_eq[i][:min_len], imfs_noise[i][:min_len])
        correlations.append(corr)
    return correlations

def calculate_reconstruction_error(original, imfs):
    """Calculate reconstruction error"""
    reconstructed = np.sum(imfs, axis=0)
    return np.mean((original - reconstructed)**2)

def plot_imf_analysis(t_eq, data_eq, imfs_eq, t_noise, data_noise, imfs_noise, 
                     trace_name_eq, trace_name_noise, comp_name, eq_info=None):
    """Plot comprehensive IMF analysis with EMD decomposition"""
    # Create figure with subplots
    n_imfs = max(len(imfs_eq), len(imfs_noise))
    fig = plt.figure(figsize=(20, 4*(n_imfs + 2)), dpi=150)
    gs = fig.add_gridspec(n_imfs + 2, 4)  # Extra row for analysis plots
    
    # 1. Original signals
    ax1_eq = fig.add_subplot(gs[0, 0:2])
    ax1_noise = fig.add_subplot(gs[0, 2:4])
    
    # Plot original signals
    ax1_eq.plot(t_eq, data_eq, 'k', lw=0.5)
    eq_title = f'Earthquake - {comp_name} component\n{trace_name_eq}'
    if eq_info:
        eq_title += f'\nM{eq_info["magnitude"]:.1f} @ {eq_info["distance"]:.1f}km'
    ax1_eq.set_title(eq_title)
    ax1_eq.grid(True)
    
    ax1_noise.plot(t_noise, data_noise, 'k', lw=0.5)
    ax1_noise.set_title(f'Noise - {comp_name} component\n{trace_name_noise}')
    ax1_noise.grid(True)
    
    # 2. Plot IMFs
    for i in range(n_imfs):
        # Earthquake IMFs
        ax_imf_eq = fig.add_subplot(gs[i+1, 0:2])
        if i < len(imfs_eq):
            ax_imf_eq.plot(t_eq, imfs_eq[i], 'k', lw=0.5)
        ax_imf_eq.set_title(f'Earthquake IMF {i+1}')
        ax_imf_eq.grid(True)
        
        # Noise IMFs
        ax_imf_noise = fig.add_subplot(gs[i+1, 2:4])
        if i < len(imfs_noise):
            ax_imf_noise.plot(t_noise, imfs_noise[i], 'k', lw=0.5)
        ax_imf_noise.set_title(f'Noise IMF {i+1}')
        ax_imf_noise.grid(True)
    
    # 3. Analysis plots at the bottom
    # Energy distribution
    ax2 = fig.add_subplot(gs[-1, 0])
    energy_eq = calculate_imf_energy(imfs_eq)
    energy_noise = calculate_imf_energy(imfs_noise)
    max_imfs = max(len(energy_eq), len(energy_noise))
    energy_eq_padded = np.zeros(max_imfs)
    energy_noise_padded = np.zeros(max_imfs)
    energy_eq_padded[:len(energy_eq)] = energy_eq
    energy_noise_padded[:len(energy_noise)] = energy_noise
    
    x = np.arange(max_imfs)
    width = 0.35
    ax2.bar(x - width/2, energy_eq_padded, width, label='Earthquake')
    ax2.bar(x + width/2, energy_noise_padded, width, label='Noise')
    ax2.set_title('Energy Distribution')
    ax2.set_xlabel('IMF Index')
    ax2.set_ylabel('Normalized Energy')
    ax2.legend()
    ax2.grid(True)
    
    # Frequency content
    ax3 = fig.add_subplot(gs[-1, 1])
    fs_eq = 1/(t_eq[1] - t_eq[0])
    fs_noise = 1/(t_noise[1] - t_noise[0])
    
    freq_eq = [calculate_imf_frequency(imf, fs_eq) for imf in imfs_eq]
    freq_noise = [calculate_imf_frequency(imf, fs_noise) for imf in imfs_noise]
    
    max_imfs = max(len(freq_eq), len(freq_noise))
    freq_eq_padded = np.zeros(max_imfs)
    freq_noise_padded = np.zeros(max_imfs)
    freq_eq_padded[:len(freq_eq)] = freq_eq
    freq_noise_padded[:len(freq_noise)] = freq_noise
    
    # Only use log scale if all values are positive
    if np.all(freq_eq_padded > 0) and np.all(freq_noise_padded > 0):
        ax3.semilogy(range(max_imfs), freq_eq_padded, 'o-', label='Earthquake')
        ax3.semilogy(range(max_imfs), freq_noise_padded, 'o-', label='Noise')
    else:
        ax3.plot(range(max_imfs), freq_eq_padded, 'o-', label='Earthquake')
        ax3.plot(range(max_imfs), freq_noise_padded, 'o-', label='Noise')
    ax3.set_title('Dominant Frequency')
    ax3.set_xlabel('IMF Index')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.legend()
    ax3.grid(True)
    
    # IMF correlations
    ax4 = fig.add_subplot(gs[-1, 2])
    correlations = calculate_imf_correlation(imfs_eq, imfs_noise)
    ax4.plot(range(len(correlations)), correlations, 'o-')
    ax4.set_title('IMF Correlations')
    ax4.set_xlabel('IMF Index')
    ax4.set_ylabel('Pearson Correlation')
    ax4.grid(True)
    
    # Reconstruction error
    ax5 = fig.add_subplot(gs[-1, 3])
    error_eq = calculate_reconstruction_error(data_eq, imfs_eq)
    error_noise = calculate_reconstruction_error(data_noise, imfs_noise)
    ax5.bar(['Earthquake', 'Noise'], [error_eq, error_noise])
    ax5.set_title('Reconstruction Error')
    ax5.grid(True)
    
    # Add overall title
    fig.suptitle(f'EMD Analysis - {comp_name} Component\n' +
                 f'Earthquake: {trace_name_eq} | Noise: {trace_name_noise}', y=0.95)
    
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
                # Create comprehensive analysis plot
                fig = plot_imf_analysis(t_eq, data_eq[:, comp_idx], imfs_eq,
                                      t_noise, data_noise[:, comp_idx], imfs_noise,
                                      trace_name_eq, trace_name_noise, comp_name, eq_info)
                
                # Save the figure
                save_path = os.path.join(figs_dir, f'imf_analysis_{trace_name_eq}_{comp_name}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                # Show the plot and wait for user input
                plt.show()
                user_input = input("Press Enter to continue to next comparison (or 'q' to quit): ")
                if user_input.lower() == 'q':
                    print("Exiting program...")
                    exit()

def plot_analysis_boxplots():
    """Create boxplots for each analysis metric across all processed data"""
    # Create four separate figures
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    
    # Common parameters
    max_imfs = max(len(imfs) for imfs in all_eq_imfs + all_noise_imfs)
    colors = {'eq': '#1f77b4', 'noise': '#ff7f0e'}  # Blue for earthquake, Orange for noise
    
    # Create legend elements
    legend_elements = [
        Patch(facecolor=colors['eq'], label='Earthquake'),
        Patch(facecolor=colors['noise'], label='Noise')
    ]
    
    # 1. Energy Distribution Boxplot
    energy_data = []
    labels = []
    positions = []
    pos = 1
    energy_samples = []  # Track number of samples
    
    for i in range(max_imfs):
        eq_energies = []
        noise_energies = []
        for imfs in all_eq_imfs:
            if i < len(imfs):
                energy = calculate_imf_energy(imfs)[i]
                eq_energies.append(energy)
        for imfs in all_noise_imfs:
            if i < len(imfs):
                energy = calculate_imf_energy(imfs)[i]
                noise_energies.append(energy)
        if eq_energies or noise_energies:
            energy_data.extend([eq_energies, noise_energies])
            energy_samples.extend([len(eq_energies), len(noise_energies)])
            labels.extend([f'{i+1}', f'{i+1}'])
            positions.extend([pos, pos+0.3])
            pos += 1.5
    
    bp1 = ax1.boxplot(energy_data, positions=positions, labels=labels, 
                     patch_artist=True, widths=0.25)
    # Color the boxes
    for i, box in enumerate(bp1['boxes']):
        box.set(facecolor=colors['eq'] if i % 2 == 0 else colors['noise'])
    ax1.set_title('Energy Distribution')
    ax1.set_xlabel('IMF')
    ax1.set_ylabel('Energy')
    ax1.legend(handles=legend_elements, loc='upper right')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
    
    # 2. Frequency Content Boxplot
    freq_data = []
    labels = []
    positions = []
    pos = 1
    freq_samples = []  # Track number of samples
    
    for i in range(max_imfs):
        eq_freqs = []
        noise_freqs = []
        for imfs, t in zip(all_eq_imfs, [np.arange(len(imfs[0]))/100 for imfs in all_eq_imfs]):
            if i < len(imfs):
                freq = calculate_imf_frequency(imfs[i], 1/(t[1]-t[0]))
                eq_freqs.append(freq)
        for imfs, t in zip(all_noise_imfs, [np.arange(len(imfs[0]))/100 for imfs in all_noise_imfs]):
            if i < len(imfs):
                freq = calculate_imf_frequency(imfs[i], 1/(t[1]-t[0]))
                noise_freqs.append(freq)
        if eq_freqs or noise_freqs:
            freq_data.extend([eq_freqs, noise_freqs])
            freq_samples.extend([len(eq_freqs), len(noise_freqs)])
            labels.extend([f'{i+1}', f'{i+1}'])
            positions.extend([pos, pos+0.3])
            pos += 1.5
    
    bp2 = ax2.boxplot(freq_data, positions=positions, labels=labels,
                     patch_artist=True, widths=0.25)
    # Color the boxes
    for i, box in enumerate(bp2['boxes']):
        box.set(facecolor=colors['eq'] if i % 2 == 0 else colors['noise'])
    ax2.set_title('Dominant Frequency')
    ax2.set_xlabel('IMF')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_yscale('log')
    ax2.legend(handles=legend_elements, loc='upper right')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)
    
    # 3. IMF Correlations Boxplot
    corr_data = []
    labels = []
    positions = []
    pos = 1
    corr_samples = []  # Track number of samples
    
    for i in range(max_imfs):
        correlations = []
        for imfs_eq, imfs_noise in zip(all_eq_imfs, all_noise_imfs):
            if i < min(len(imfs_eq), len(imfs_noise)):
                min_len = min(len(imfs_eq[i]), len(imfs_noise[i]))
                corr, _ = pearsonr(imfs_eq[i][:min_len], imfs_noise[i][:min_len])
                correlations.append(corr)
        if correlations:
            corr_data.append(correlations)
            corr_samples.append(len(correlations))
            labels.append(f'{i+1}')
            positions.append(pos)
            pos += 1
    
    bp3 = ax3.boxplot(corr_data, positions=positions, labels=labels,
                     patch_artist=True, widths=0.5)
    # Color the boxes
    for box in bp3['boxes']:
        box.set(facecolor='#2ca02c')  # Green for correlations
    ax3.set_title('IMF Correlations')
    ax3.set_xlabel('IMF')
    ax3.set_ylabel('Correlation')
    ax3.legend(handles=[Patch(facecolor='#2ca02c', label='Correlation')], loc='upper right')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)
    
    # 4. Reconstruction Error Boxplot
    error_data = []
    labels = []
    positions = []
    pos = 1
    error_samples = []  # Track number of samples
    
    for i in range(max_imfs):
        eq_errors = []
        noise_errors = []
        for imfs, data in zip(all_eq_imfs, [np.array(imfs[0]) for imfs in all_eq_imfs]):
            if i < len(imfs):
                error = calculate_reconstruction_error(data, imfs)
                eq_errors.append(error)
        for imfs, data in zip(all_noise_imfs, [np.array(imfs[0]) for imfs in all_noise_imfs]):
            if i < len(imfs):
                error = calculate_reconstruction_error(data, imfs)
                noise_errors.append(error)
        if eq_errors or noise_errors:
            error_data.extend([eq_errors, noise_errors])
            error_samples.extend([len(eq_errors), len(noise_errors)])
            labels.extend([f'{i+1}', f'{i+1}'])
            positions.extend([pos, pos+0.3])
            pos += 1.5
    
    bp4 = ax4.boxplot(error_data, positions=positions, labels=labels,
                     patch_artist=True, widths=0.25)
    # Color the boxes
    for i, box in enumerate(bp4['boxes']):
        box.set(facecolor=colors['eq'] if i % 2 == 0 else colors['noise'])
    ax4.set_title('Reconstruction Error')
    ax4.set_xlabel('IMF')
    ax4.set_ylabel('Error')
    ax4.set_yscale('log')
    ax4.legend(handles=legend_elements, loc='upper right')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0)
    
    # Adjust layouts and save figures
    for fig, name in zip([fig1, fig2, fig3, fig4], 
                        ['energy_boxplot', 'frequency_boxplot', 
                         'correlation_boxplot', 'error_boxplot']):
        fig.tight_layout()
        save_path = os.path.join(figs_dir, f'{name}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Print sample information
    print("\nSample counts for boxplots:")
    print("Energy Distribution:")
    for i in range(0, len(energy_samples), 2):
        print(f"  IMF{i//2 + 1}: EQ={energy_samples[i]}, Noise={energy_samples[i+1]}")
    
    print("\nFrequency Content:")
    for i in range(0, len(freq_samples), 2):
        print(f"  IMF{i//2 + 1}: EQ={freq_samples[i]}, Noise={freq_samples[i+1]}")
    
    print("\nIMF Correlations:")
    for i, n in enumerate(corr_samples):
        print(f"  IMF{i+1}: {n} samples")
    
    print("\nReconstruction Error:")
    for i in range(0, len(error_samples), 2):
        print(f"  IMF{i//2 + 1}: EQ={error_samples[i]}, Noise={error_samples[i+1]}")
    
    return fig1, fig2, fig3, fig4

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

# Create and save boxplot analysis
if len(all_eq_imfs) > 0 and len(all_noise_imfs) > 0:
    boxplot_figs = plot_analysis_boxplots()
    print("Analysis boxplots saved to figs directory:")
    print("- energy_boxplot.png")
    print("- frequency_boxplot.png")
    print("- correlation_boxplot.png")
    print("- error_boxplot.png") 