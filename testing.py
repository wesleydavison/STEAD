import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import matplotlib.ticker as ticker
import os
from matplotlib.colors import LogNorm
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
MODE = 'prod'  # 'test' or 'prod'
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
quiet = False    # If True, skip waveform+PSD plots
plot_aggregated_only = True  # If True, only plot aggregated PSD, skip individual waveform plots

# %% Storage
max_amp_e_eq = []
max_amp_n_eq = []
max_amp_z_eq = []
max_amp_e_noise = []
max_amp_n_noise = []
max_amp_z_noise = []
psd_data_eq = []  # [(f, Pxx_e, Pxx_n, Pxx_z), ...]
psd_data_noise = []  # [(f, Pxx_e, Pxx_n, Pxx_z), ...]

# Store all PSDs for aggregation
all_noise_psd_e = []
all_noise_psd_n = []
all_noise_psd_z = []
all_noise_freqs = []
all_noise_traces = []  # Store trace names for noise
all_eq_psd_e = []
all_eq_psd_n = []
all_eq_psd_z = []
all_eq_freqs = []
all_eq_traces = []  # Store trace names and info for earthquakes
all_eq_info = []  # Store magnitude and distance info

def create_aggregated_psd():
    """Create and save aggregated PSD plots in both log-log and linear scales"""
    if len(all_noise_psd_e) > 0 and len(all_eq_psd_e) > 0:
        # Convert lists to arrays
        all_noise_psd_e_array = np.array(all_noise_psd_e)
        all_noise_psd_n_array = np.array(all_noise_psd_n)
        all_noise_psd_z_array = np.array(all_noise_psd_z)
        all_eq_psd_e_array = np.array(all_eq_psd_e)
        all_eq_psd_n_array = np.array(all_eq_psd_n)
        all_eq_psd_z_array = np.array(all_eq_psd_z)
        
        # Use the frequency arrays (they should all be the same within each dataset)
        freqs_noise = all_noise_freqs[0]
        freqs_eq = all_eq_freqs[0]
        
        # Calculate mean PSDs
        mean_noise_psd_e = np.mean(all_noise_psd_e_array, axis=0)
        mean_noise_psd_n = np.mean(all_noise_psd_n_array, axis=0)
        mean_noise_psd_z = np.mean(all_noise_psd_z_array, axis=0)
        mean_eq_psd_e = np.mean(all_eq_psd_e_array, axis=0)
        mean_eq_psd_n = np.mean(all_eq_psd_n_array, axis=0)
        mean_eq_psd_z = np.mean(all_eq_psd_z_array, axis=0)
        
        # Create two figures - one for log-log and one for linear scales
        fig_log, (ax1_log, ax2_log) = plt.subplots(1, 2, figsize=(20, 8), dpi=150)
        fig_lin, (ax1_lin, ax2_lin) = plt.subplots(1, 2, figsize=(20, 8), dpi=150)
        
        # Function to plot PSDs with given axes
        def plot_psds(ax1, ax2, is_log=False):
            # Plot noise PSDs
            if is_log:
                ax1.loglog(freqs_noise, mean_noise_psd_e, label='E component', color='C0')
                ax1.loglog(freqs_noise, mean_noise_psd_n, label='N component', color='C1')
                ax1.loglog(freqs_noise, mean_noise_psd_z, label='Z component', color='C2')
                ax2.loglog(freqs_eq, mean_eq_psd_e, label='E component', color='C0')
                ax2.loglog(freqs_eq, mean_eq_psd_n, label='N component', color='C1')
                ax2.loglog(freqs_eq, mean_eq_psd_z, label='Z component', color='C2')
            else:
                ax1.plot(freqs_noise, mean_noise_psd_e, label='E component', color='C0')
                ax1.plot(freqs_noise, mean_noise_psd_n, label='N component', color='C1')
                ax1.plot(freqs_noise, mean_noise_psd_z, label='Z component', color='C2')
                ax2.plot(freqs_eq, mean_eq_psd_e, label='E component', color='C0')
                ax2.plot(freqs_eq, mean_eq_psd_n, label='N component', color='C1')
                ax2.plot(freqs_eq, mean_eq_psd_z, label='Z component', color='C2')
            
            # Add shaded regions for standard deviation
            std_noise_psd_e = np.std(all_noise_psd_e_array, axis=0)
            std_noise_psd_n = np.std(all_noise_psd_n_array, axis=0)
            std_noise_psd_z = np.std(all_noise_psd_z_array, axis=0)
            std_eq_psd_e = np.std(all_eq_psd_e_array, axis=0)
            std_eq_psd_n = np.std(all_eq_psd_n_array, axis=0)
            std_eq_psd_z = np.std(all_eq_psd_z_array, axis=0)
            
            # Plot shaded regions
            for ax, mean_psds, std_psds, freqs, title, traces, info in zip(
                [ax1, ax2],
                [[mean_noise_psd_e, mean_noise_psd_n, mean_noise_psd_z],
                 [mean_eq_psd_e, mean_eq_psd_n, mean_eq_psd_z]],
                [[std_noise_psd_e, std_noise_psd_n, std_noise_psd_z],
                 [std_eq_psd_e, std_eq_psd_n, std_eq_psd_z]],
                [freqs_noise, freqs_eq],
                ['Noise', 'Earthquake'],
                [all_noise_traces, all_eq_traces],
                [None, all_eq_info]
            ):
                # Plot mean and std for each component
                for i, (mean, std) in enumerate(zip(mean_psds, std_psds)):
                    ax.fill_between(freqs, mean - std, mean + std, 
                                  color=f'C{i}', alpha=0.2)
                
                # Set plot properties
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('PSD (counts²/Hz)')
                
                # Create detailed title
                if title == 'Noise':
                    title_text = f'Aggregated PSD - {title}\nTraces: {", ".join(traces[:3])}...'
                else:
                    mag_dist_info = "\n".join([f"M{info['magnitude']:.1f} @ {info['distance']:.1f}km" 
                                             for info in info[:3]])
                    title_text = f'Aggregated PSD - {title}\nTraces: {", ".join(traces[:3])}...\n{mag_dist_info}'
                
                ax.set_title(title_text)
                ax.grid(True, which='both', ls='--', lw=0.5)
                ax.legend()
        
        # Create both plots
        plot_psds(ax1_log, ax2_log, is_log=True)
        plot_psds(ax1_lin, ax2_lin, is_log=False)
        
        # Save both figures
        plt.figure(fig_log.number)
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, 'aggregated_psd_comparison_loglog.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_log)
        
        plt.figure(fig_lin.number)
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, 'aggregated_psd_comparison_linear.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_lin)
        
        print(f"Created aggregated PSD comparison plots with {len(all_noise_psd_e)} noise records and {len(all_eq_psd_e)} earthquake records")

# %% Loop over CSV in chunks
chunks_eq = pd.read_csv(csv_file_eq, chunksize=chunksize, nrows=nrows)
chunks_noise = pd.read_csv(csv_file_noise, chunksize=chunksize, nrows=nrows)

# Calculate total number of chunks
if MODE == 'test':
    total_chunks = min(nrows // chunksize + (1 if nrows % chunksize else 0), 
                      len(list(pd.read_csv(csv_file_eq, chunksize=chunksize, nrows=nrows))))
else:
    # For production, estimate total chunks from file size
    import os
    file_size = os.path.getsize(csv_file_eq)
    estimated_rows = file_size / 1000  # Rough estimate: 1KB per row
    total_chunks = (estimated_rows + chunksize - 1) // chunksize

# Create progress bar for chunks
with tqdm(total=total_chunks, desc=f"Processing chunks ({MODE} mode)") as pbar:
    for chunk_eq, chunk_noise in zip(chunks_eq, chunks_noise):
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
        
        # Get all noise data for aggregated PSD
        all_noise_data = chunk_noise[chunk_noise.trace_category == 'noise']
        
        # Sample noise data to match earthquake data size for individual comparisons
        chunk_noise = all_noise_data.sample(n=len(chunk_eq), random_state=42) if not chunk_eq.empty else all_noise_data
        
        if chunk_eq.empty:
            print("No events found in this chunk")
            pbar.update(1)
            continue

        ev_list_eq = chunk_eq['trace_name'].tolist()
        ev_list_noise = chunk_noise['trace_name'].tolist()
        all_noise_list = all_noise_data['trace_name'].tolist()

        # Process both earthquake and noise data
        with h5py.File(file_name_eq, 'r') as dtfl_eq, h5py.File(file_name_noise, 'r') as dtfl_noise:
            # First process all noise data for aggregated PSD
            for trace_name_noise in tqdm(all_noise_list, 
                                       desc="Processing all noise data",
                                       leave=False):
                ds_noise = dtfl_noise.get(f"data/{trace_name_noise}")
                
                if ds_noise is None:
                    continue

                # Process noise data
                data_noise = np.array(ds_noise)
                fs_noise = float(ds_noise.attrs.get('sampling_rate', 100.0))
                n_noise = data_noise.shape[0]
                
                # Compute PSDs for noise
                nperseg_noise = min(int(fs_noise*4), n_noise)
                noverlap_noise = nperseg_noise // 2
                
                # Noise PSDs
                f_noise, Pxx_e_noise = welch(data_noise[:, 0], fs=fs_noise, nperseg=nperseg_noise, noverlap=noverlap_noise)
                _, Pxx_n_noise = welch(data_noise[:, 1], fs=fs_noise, nperseg=nperseg_noise, noverlap=noverlap_noise)
                _, Pxx_z_noise = welch(data_noise[:, 2], fs=fs_noise, nperseg=nperseg_noise, noverlap=noverlap_noise)

                # Store PSDs for aggregation
                all_noise_psd_e.append(Pxx_e_noise)
                all_noise_psd_n.append(Pxx_n_noise)
                all_noise_psd_z.append(Pxx_z_noise)
                all_noise_freqs.append(f_noise)
                all_noise_traces.append(trace_name_noise)

            # Then process earthquake and sampled noise data for comparison
            for trace_name_eq, trace_name_noise in tqdm(zip(ev_list_eq, ev_list_noise), 
                                                      total=len(ev_list_eq),
                                                      desc="Processing comparison data",
                                                      leave=False):
                ds_eq = dtfl_eq.get(f"data/{trace_name_eq}")
                ds_noise = dtfl_noise.get(f"data/{trace_name_noise}")
                
                if ds_eq is None or ds_noise is None:
                    continue

                # Process earthquake data
                data_eq = np.array(ds_eq)
                fs_eq = float(ds_eq.attrs.get('sampling_rate', 100.0))
                n_eq = data_eq.shape[0]
                t_eq = np.arange(n_eq) / fs_eq

                # Process noise data
                data_noise = np.array(ds_noise)
                fs_noise = float(ds_noise.attrs.get('sampling_rate', 100.0))
                n_noise = data_noise.shape[0]
                t_noise = np.arange(n_noise) / fs_noise

                # Compute max amplitudes
                max_amp_e_eq.append(np.max(np.abs(data_eq[:, 0])))
                max_amp_n_eq.append(np.max(np.abs(data_eq[:, 1])))
                max_amp_z_eq.append(np.max(np.abs(data_eq[:, 2])))
                max_amp_e_noise.append(np.max(np.abs(data_noise[:, 0])))
                max_amp_n_noise.append(np.max(np.abs(data_noise[:, 1])))
                max_amp_z_noise.append(np.max(np.abs(data_noise[:, 2])))

                # Compute PSDs
                nperseg_eq = min(int(fs_eq*4), n_eq)
                nperseg_noise = min(int(fs_noise*4), n_noise)
                noverlap_eq = nperseg_eq // 2
                noverlap_noise = nperseg_noise // 2

                # Earthquake PSDs
                f_eq, Pxx_e_eq = welch(data_eq[:, 0], fs=fs_eq, nperseg=nperseg_eq, noverlap=noverlap_eq)
                _, Pxx_n_eq = welch(data_eq[:, 1], fs=fs_eq, nperseg=nperseg_eq, noverlap=noverlap_eq)
                _, Pxx_z_eq = welch(data_eq[:, 2], fs=fs_eq, nperseg=nperseg_eq, noverlap=noverlap_eq)
                
                # Store earthquake PSDs for aggregation
                all_eq_psd_e.append(Pxx_e_eq)
                all_eq_psd_n.append(Pxx_n_eq)
                all_eq_psd_z.append(Pxx_z_eq)
                all_eq_freqs.append(f_eq)
                all_eq_traces.append(trace_name_eq)
                all_eq_info.append({
                    'magnitude': chunk_eq[chunk_eq['trace_name'] == trace_name_eq]['source_magnitude'].iloc[0],
                    'distance': chunk_eq[chunk_eq['trace_name'] == trace_name_eq]['source_distance_km'].iloc[0]
                })

                if quiet or plot_aggregated_only:
                    continue

                # Plotting
                if plotting:
                    fig, axes = plt.subplots(5, 2, figsize=(16, 15), dpi=150)
                    
                    # Get earthquake info for title
                    eq_info = chunk_eq[chunk_eq['trace_name'] == trace_name_eq].iloc[0]
                    eq_title = f'Earthquake\n{trace_name_eq}\nM{eq_info.source_magnitude:.1f} @ {eq_info.source_distance_km:.1f}km'
                    noise_title = f'Noise\n{trace_name_noise}'
                    
                    # Earthquake waveforms
                    labels = ['E', 'N', 'Z']
                    for i, ax in enumerate(axes[:3, 0]):
                        ax.plot(t_eq, data_eq[:, i], 'k', lw=0.5)
                        ax.set_ylabel(f'{labels[i]} counts')
                        # arrival lines if valid
                        for name, color in (
                            ('p_arrival_sample', 'b'),
                            ('s_arrival_sample', 'r'),
                            ('coda_end_sample', 'aqua')
                        ):
                            idx = ds_eq.attrs.get(name, None)
                            if isinstance(idx, (int, float)) and 0 < idx < n_eq:
                                ax.axvline(idx/fs_eq, color=color, lw=1)
                        ax.grid(True)
                        if i == 0:
                            ax.set_title(eq_title)

                    # Noise waveforms
                    for i, ax in enumerate(axes[:3, 1]):
                        ax.plot(t_noise, data_noise[:, i], 'k', lw=0.5)
                        ax.set_ylabel(f'{labels[i]} counts')
                        ax.grid(True)
                        if i == 0:
                            ax.set_title(noise_title)

                    # Combined traces
                    ax4_eq = axes[3, 0]
                    ax4_eq.plot(t_eq, data_eq[:, 0], label='E', color='C0', lw=0.5)
                    ax4_eq.plot(t_eq, data_eq[:, 1], label='N', color='C1', lw=0.5)
                    ax4_eq.plot(t_eq, data_eq[:, 2], label='Z', color='C2', lw=0.5)
                    ax4_eq.set_ylabel('Counts')
                    ax4_eq.legend()
                    ax4_eq.grid(True)
                    ax4_eq.set_title(eq_title)

                    ax4_noise = axes[3, 1]
                    ax4_noise.plot(t_noise, data_noise[:, 0], label='E', color='C0', lw=0.5)
                    ax4_noise.plot(t_noise, data_noise[:, 1], label='N', color='C1', lw=0.5)
                    ax4_noise.plot(t_noise, data_noise[:, 2], label='Z', color='C2', lw=0.5)
                    ax4_noise.set_ylabel('Counts')
                    ax4_noise.legend()
                    ax4_noise.grid(True)
                    ax4_noise.set_title(noise_title)

                    # PSDs
                    ax5_eq = axes[4, 0]
                    ax5_eq.plot(f_eq, Pxx_e_eq, label='E')
                    ax5_eq.plot(f_eq, Pxx_n_eq, label='N')
                    ax5_eq.plot(f_eq, Pxx_z_eq, label='Z')
                    ax5_eq.set_xlabel('Frequency (Hz)')
                    ax5_eq.set_ylabel('PSD (counts²/Hz)')
                    ax5_eq.legend()
                    ax5_eq.grid(True, which='both', ls='--', lw=0.5)
                    ax5_eq.set_title(eq_title)

                    ax5_noise = axes[4, 1]
                    ax5_noise.plot(f_noise, Pxx_e_noise, label='E')
                    ax5_noise.plot(f_noise, Pxx_n_noise, label='N')
                    ax5_noise.plot(f_noise, Pxx_z_noise, label='Z')
                    ax5_noise.set_xlabel('Frequency (Hz)')
                    ax5_noise.set_ylabel('PSD (counts²/Hz)')
                    ax5_noise.legend()
                    ax5_noise.grid(True, which='both', ls='--', lw=0.5)
                    ax5_noise.set_title(noise_title)

                    plt.tight_layout()
                    
                    # Save the figure
                    fig_path = os.path.join(figs_dir, f'comparison_{trace_name_eq}.png')
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)  # Close the figure to free memory
                    
                    # Ask for continue
                    user_input = input("Press Enter to continue to next comparison (or 'q' to create heatmap): ")
                    if user_input.lower() == 'q':
                        print("Creating aggregated PSD plot...")
                        create_aggregated_psd()
                        print("Aggregated PSD plot created. Exiting program...")
                        exit()

        # Update chunk progress bar
        pbar.update(1)

# %% Convert amplitude lists to arrays for further stats
max_amp_e_eq = np.array(max_amp_e_eq)
max_amp_n_eq = np.array(max_amp_n_eq)
max_amp_z_eq = np.array(max_amp_z_eq)
max_amp_e_noise = np.array(max_amp_e_noise)
max_amp_n_noise = np.array(max_amp_n_noise)
max_amp_z_noise = np.array(max_amp_z_noise)

# Create final aggregated PSD plot with all collected data
create_aggregated_psd()

# Now max_amp_* and psd_data are ready for your downstream analysis.
