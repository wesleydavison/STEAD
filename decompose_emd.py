import numpy as np
import pandas as pd
import h5py
import os
from PyEMD import EMD
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import numpy as np
from scipy.signal import hilbert



def plot_emd_features(data, fs, imfs, n_imfs, features=None, title="EMD Features", stats=None):
    """Plot EMD features for a single trace"""
    n_imfs_to_plot = min(n_imfs, max(imf.shape[0] for imf in imfs))
    n_features_per_comp = n_imfs_to_plot * 2 + 1  # energies + frequencies + error
    fig = plt.figure(figsize=(15, 4*(n_imfs_to_plot + 2)))
    gs = fig.add_gridspec(n_imfs_to_plot + 2, 3)
    fig.suptitle(title)
    t = np.arange(len(data)) / fs
    components = ['e', 'n', 'z']
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
                # Annotate with stats if provided
                if stats is not None and len(stats) > comp_idx and i < len(stats[comp_idx]['mean']):
                    stat_text = (
                        f"mean={stats[comp_idx]['mean'][i]:.2g}\n"
                        f"std={stats[comp_idx]['std'][i]:.2g}\n"
                        f"skew={stats[comp_idx]['skewness'][i]:.2g}\n"
                        f"kurt={stats[comp_idx]['kurtosis'][i]:.2g}\n"
                        f"zcross={int(stats[comp_idx]['zero_crossings'][i])}\n"
                        f"ptp={stats[comp_idx]['peak_to_peak'][i]:.2g}\n"
                        f"centroid={stats[comp_idx]['spectral_centroid'][i]:.2g}\n"
                        f"bandwidth={stats[comp_idx]['spectral_bandwidth'][i]:.2g}\n"
                        f"entropy={stats[comp_idx]['spectral_entropy'][i]:.2g}\n"
                        f"flatness={stats[comp_idx]['spectral_flatness'][i]:.2g}\n"
                        f"fm_index={stats[comp_idx]['freq_modulation_index'][i]:.2g}\n"

                    )
                    ax.text(0.98, 0.98, stat_text, transform=ax.transAxes,
                            fontsize=8, va='top', ha='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.tight_layout()
    plt.show()
    plt.savefig(f"emd_features_{title}.png")
    return fig
    
    
def imf_freq_domain_stats(imfs, fs=100):
    """
    imfs: np.ndarray of shape (n_imfs, n_samples)
    fs: sampling frequency
    Returns: dict of lists, one per frequency metric
    """
    eps = np.finfo(float).eps
    stats = {
        'spectral_centroid': [],
        'spectral_bandwidth': [],
        'spectral_entropy': [],
        'spectral_flatness': [],
        'freq_modulation_index': []
    }

    for imf in imfs:
        N = len(imf)
        # FFT power spectrum
        freqs = np.fft.rfftfreq(N, d=1/fs)
        P = np.abs(np.fft.rfft(imf))**2
        P_sum = P.sum() + eps
        P_norm = P / P_sum

        # Spectral centroid
        centroid = np.sum(freqs * P_norm)

        # Spectral bandwidth
        bandwidth = np.sqrt(np.sum(((freqs - centroid)**2) * P_norm))

        # Spectral entropy
        entropy = -np.sum(P_norm * np.log(P_norm + eps))

        # Spectral flatness (geometric mean / arithmetic mean)
        geo_mean = np.exp(np.mean(np.log(P + eps)))
        arith_mean = np.mean(P + eps)
        flatness = geo_mean / arith_mean

        # Instantaneous frequency via Hilbert transform
        analytic_sig = hilbert(imf)
        inst_phase = np.unwrap(np.angle(analytic_sig))
        inst_freq = np.diff(inst_phase) * fs / (2 * np.pi)
        # avoid division by zero
        if np.mean(inst_freq) != 0:
            fm_index = np.std(inst_freq) / np.mean(inst_freq)
        else:
            fm_index = 0.0

        # append
        stats['spectral_centroid'].append(centroid)
        stats['spectral_bandwidth'].append(bandwidth)
        stats['spectral_entropy'].append(entropy)
        stats['spectral_flatness'].append(flatness)
        stats['freq_modulation_index'].append(fm_index)

    return stats
    
    
def imf_freq_domain_stats(imfs, fs=100):
    """
    imfs: np.ndarray of shape (n_imfs, n_samples)
    fs: sampling frequency
    Returns: dict of lists, one per frequency metric
    """
    eps = np.finfo(float).eps
    stats = {
        'spectral_centroid': [],
        'spectral_bandwidth': [],
        'spectral_entropy': [],
        'spectral_flatness': [],
        'freq_modulation_index': []
    }

    for imf in imfs:
        N = len(imf)
        # FFT power spectrum
        freqs = np.fft.rfftfreq(N, d=1/fs)
        P = np.abs(np.fft.rfft(imf))**2
        P_sum = P.sum() + eps
        P_norm = P / P_sum

        # Spectral centroid
        centroid = np.sum(freqs * P_norm)

        # Spectral bandwidth
        bandwidth = np.sqrt(np.sum(((freqs - centroid)**2) * P_norm))

        # Spectral entropy
        entropy = -np.sum(P_norm * np.log(P_norm + eps))

        # Spectral flatness (geometric mean / arithmetic mean)
        geo_mean = np.exp(np.mean(np.log(P + eps)))
        arith_mean = np.mean(P + eps)
        flatness = geo_mean / arith_mean

        # Instantaneous frequency via Hilbert transform
        analytic_sig = hilbert(imf)
        inst_phase = np.unwrap(np.angle(analytic_sig))
        inst_freq = np.diff(inst_phase) * fs / (2 * np.pi)
        # avoid division by zero
        if np.mean(inst_freq) != 0:
            fm_index = np.std(inst_freq) / np.mean(inst_freq)
        else:
            fm_index = 0.0

        # append
        stats['spectral_centroid'].append(centroid)
        stats['spectral_bandwidth'].append(bandwidth)
        stats['spectral_entropy'].append(entropy)
        stats['spectral_flatness'].append(flatness)
        stats['freq_modulation_index'].append(fm_index)

    return stats

def imf_time_domain_stats(imfs):
    """
    imfs: np.ndarray of shape (n_imfs, n_samples)
    Returns: dict of lists, one per statistic
    """
    stats = {
        'mean': [],
        'std': [],
        'skewness': [],
        'kurtosis': [],
        'zero_crossings': [],
        'peak_to_peak': []
    }
    for imf in imfs:
        stats['mean'].append(np.mean(imf))
        stats['std'].append(np.std(imf))
        stats['skewness'].append(skew(imf))
        stats['kurtosis'].append(kurtosis(imf))
        stats['zero_crossings'].append(np.sum(np.diff(np.sign(imf)) != 0))
        stats['peak_to_peak'].append(np.ptp(imf))
    return stats

def decompose_and_save_emd(
    file_name_hdf5, 
    csv_file, 
    output_file, 
    n_samples=10, 
    components=['e', 'n', 'z'],
    plot_random_traces=5
):
    print(f"Processing: {csv_file}")
    dtype_dict = {
        'trace_name': 'category',
        'trace_category': 'category',
        'source_magnitude': 'float32'
    }
    df = pd.read_csv(csv_file, usecols=['trace_name', 'trace_category', 'source_magnitude'], dtype=dtype_dict)
    all_indices = np.arange(len(df))
    np.random.seed(42)
    sampled_indices = np.random.choice(all_indices, size=min(n_samples, len(df)), replace=False)
    sampled_df = df.iloc[sampled_indices]
    eq_data = {}
    noise_data = {}
    with h5py.File(output_file, 'w') as f_out:
        f_out.attrs['creation_date'] = np.string_(pd.Timestamp.now().isoformat())
        f_out.attrs['n_samples'] = n_samples
        with h5py.File(file_name_hdf5, 'r') as h5f:
            for _, row in sampled_df.iterrows():
                trace_name = row['trace_name']
                trace_category = row['trace_category']
                mag = row.get('source_magnitude', 0)
                ds = h5f.get(f"data/{trace_name}")
                if ds is None:
                    continue
                data = np.array(ds)
                fs = float(ds.attrs.get('sampling_rate', 100.0))
                if trace_category == 'earthquake_local':
                    eq_data[trace_name] = (data, fs, mag)
                else:
                    noise_data[trace_name] = (data, fs)
                # EMD decomposition for each component
                imfs = []
                imf_stats = []
                for comp_idx in range(data.shape[1]):
                    emd = EMD()
                    imfs_comp = emd.emd(data[:, comp_idx])
                    imfs.append(imfs_comp)
                    # Calculate stats for this component's IMFs
                    stats = imf_time_domain_stats(imfs_comp)
                    # print(f"Stats for component {comp_idx} of trace, {trace_name} : {stats}")
                    
                    stats_fd = imf_freq_domain_stats(imfs_comp)
                    stats.update(stats_fd)
                    imf_stats.append(stats)

                    
                plot_emd_features(data, fs, imfs, n_imfs=10, title=f"EMD_Decomposition_{trace_name}", stats=imf_stats)
                # Save decomposition results
                trace_group = f_out.create_group(trace_name)
                trace_group.attrs['sampling_rate'] = fs
                trace_group.attrs['magnitude'] = mag
                trace_group.attrs['category'] = trace_category
                trace_group.create_dataset('original_data', data=data)
                for comp_idx, (comp_name, imf_data, stats) in enumerate(zip(components, imfs, imf_stats)):
                    comp_group = trace_group.create_group(f'component_{comp_name}')
                    comp_group.create_dataset('imfs', data=imf_data)
                    # Save stats as attributes (as JSON string for easy reading)
                    for stat_name, stat_list in stats.items():
                        comp_group.attrs[stat_name] = np.array(stat_list)
                print(f"Saved decomposition for trace: {trace_name}")
        print(f"All decompositions saved to: {output_file}")
        # Optionally plot random traces
        print("Checking if the file saved correctly...")
        print(f"File contains the following groups: {list(f_out.keys())}")
        for group in f_out.keys():
            print(f"Group {group} contains the following datasets: {list(f_out[group].keys())}")
        print("Choosing random traces to plot...")
        random_trace_names = np.random.choice(list(f_out.keys()), size=min(plot_random_traces, len(f_out.keys())), replace=False)
        print(f"Plotting EMD decompositions for the following traces: {random_trace_names}")
        for trace_name in random_trace_names:
            trace_group = f_out[trace_name]
            imfs = []
            for comp in components:
                imfs.append(trace_group[f'component_{comp}']['imfs'][:])
            plot_emd_features(trace_group['original_data'][:], 
                              trace_group.attrs['sampling_rate'], 
                              imfs, 
                              n_imfs=10, 
                              title=f"EMD_test_Decomposition-{trace_name}",
                              stats=imf_stats)

# --- Configuration for earthquake and noise data ---
file_name_merged = r"/users/230442014/archive/STEAD_dataset/chunk2.hdf5"
csv_file_merged = r"/users/230442014/archive/STEAD_dataset/chunk2.csv"
output_file_merged = 'emd_decompositions_chunk2.h5'

file_name_noise = r"/users/230442014/archive/STEAD_dataset/chunk1.hdf5"
csv_file_noise = r"/users/230442014/archive/STEAD_dataset/chunk1.csv"
output_file_noise = 'emd_decompositions_chunk1.h5'

# Ensure model directory exists
os.makedirs("saved_models", exist_ok=True)

# Run for earthquake data
decompose_and_save_emd(
    file_name_hdf5=file_name_merged,
    csv_file=csv_file_merged,
    output_file=output_file_merged,
    n_samples=3
)

# Run for noise data
decompose_and_save_emd(
    file_name_hdf5=file_name_noise,
    csv_file=csv_file_noise,
    output_file=output_file_noise,
    n_samples=3
)





