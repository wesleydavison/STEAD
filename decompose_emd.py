# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import h5py
import os
from PyEMD import EMD
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import numpy as np
from scipy.signal import hilbert
import math
import matplotlib.patches as mpatches
from tqdm import tqdm




def plot_emd_features(data, fs, imfs, n_imfs, title="EMD Features", stats=None):
    ###
    #Draw original 3-component trace, energy-ratio bars, chosen IMFs with
    #stats, correlation-matrix heat-map and mode-mixing-index stem plot.
    #------------------------------------------------------------------
    # `imfs`  – list [imfs_E, imfs_N, imfs_Z] with shape
    #            (n_imfs_comp, n_samples) for each component.
    # `stats` – list of 3 dicts (one per component) produced in
    #            decompose_and_save_emd.  Must contain keys:
    #            mean, std, ..., energy_ratio, corr_matrix, mode_mixing.
    #
    # How many IMFs do we actually have to show?
    n_imfs_to_plot = min(n_imfs, max(imf.shape[0] for imf in imfs))

    # Rows: 0  = original trace
    #       1  = energy ratio
    #    2..(1+N)  = IMFs
    #   N+2        = correlation matrix
    #   N+3        = mode-mixing index
    rows = n_imfs_to_plot + 4
    fig  = plt.figure(figsize=(15, 4 * rows))
    gs   = fig.add_gridspec(rows, 3, hspace=0.6)
    fig.suptitle(title, fontsize=16, y=0.99)

    t = np.arange(len(data)) / fs
    components = ['e', 'n', 'z']

    for comp_idx, comp in enumerate(components):
        # ----------- row 0: original component -----------
        ax0 = fig.add_subplot(gs[0, comp_idx])
        ax0.plot(t, data[:, comp_idx], 'k', lw=0.5)
        ax0.set_title(f'{comp.upper()} component')
        ax0.set_xlabel('Time (s)'); ax0.set_ylabel('Amplitude'); ax0.grid(True)

        # ----------- row 1: energy-ratio bars -----------
        if stats is not None:
            er = stats[comp_idx]['energy_ratio']
            ax_er = fig.add_subplot(gs[1, comp_idx])
            ax_er.bar(np.arange(len(er)) + 1, er, color='slategray')
            ax_er.set_title('Energy ratio')
            ax_er.set_xlabel('IMF #'); ax_er.set_ylabel('Fraction')
            ax_er.set_ylim(0, 1.05)

        # ----------- rows 2 … N: IMFs -----------
        for i in range(n_imfs_to_plot):
            if i >= imfs[comp_idx].shape[0]:
                continue
            ax = fig.add_subplot(gs[i + 2, comp_idx])
            ax.plot(t, imfs[comp_idx][i], 'b', lw=0.5)
            ax.set_title(f'IMF {i + 1}')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude'); ax.grid(True)

            # ----- per-IMF stat overlay -----
            if stats is not None and i < len(stats[comp_idx]['mean']):
                st = stats[comp_idx]  # shorthand
                stat_text = (
                    f"mean={st['mean'][i]:.2g}\n"
                    f"std={st['std'][i]:.2g}\n"
                    f"skew={st['skewness'][i]:.2g}\n"
                    f"kurt={st['kurtosis'][i]:.2g}\n"
                    f"zcross={int(st['zero_crossings'][i])}\n"
                    f"ptp={st['peak_to_peak'][i]:.2g}\n"
                    f"centroid={st['spectral_centroid'][i]:.2g}\n"
                    f"bw={st['spectral_bandwidth'][i]:.2g}\n"
                    f"entropy={st['spectral_entropy'][i]:.2g}\n"
                    f"flatness={st['spectral_flatness'][i]:.2g}\n"
                    f"fm_idx={st['freq_modulation_index'][i]:.2g}"
                )
                ax.text(0.98, 0.98, stat_text, transform=ax.transAxes,
                        fontsize=7, va='top', ha='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # ----------- row N+1: correlation heat-map -----------
        if stats is not None:
            ax_cm = fig.add_subplot(gs[n_imfs_to_plot + 2, comp_idx])
            cmat  = stats[comp_idx]['corr_matrix']
            im    = ax_cm.imshow(cmat, vmin=-1, vmax=1,
                                 cmap='seismic', aspect='auto')
            ax_cm.set_title('IMF correlation')
            ax_cm.set_xlabel('IMF'); ax_cm.set_ylabel('IMF')
            fig.colorbar(im, ax=ax_cm, shrink=0.7)

        # ----------- row N+2: mode-mixing index -----------
        if stats is not None:
            mmi  = stats[comp_idx]['mode_mixing']
            ax_m = fig.add_subplot(gs[n_imfs_to_plot + 3, comp_idx])
            ax_m.stem(np.arange(1, len(mmi) + 1), mmi, basefmt=' ')
            ax_m.set_title('Mode-mixing index')
            ax_m.set_xlabel('Between IMF i & i+1'); ax_m.set_ylabel('MMI')
            ax_m.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(f"emd_features_{title}.png", dpi=150)
    #plt.show()
    plt.close(fig)
    return fig


# -------------------------  NON-LINEAR IMF METRICS  --------------------------


# -- 1.a  Fractal dimension  (Higuchi’s method – robust for short 1-D data) --
def higuchi_fd(x, kmax=10):
    """
    Higuchi fractal dimension of 1-D signal x.
    kmax controls scale. 8-10 is typical for a few-hundred-sample IMF.
    """
    N = len(x)
    L = []
    k_vals = range(1, kmax + 1)
    for k in k_vals:
        Lk = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:      # nothing to measure
                continue
            dist = np.abs(np.diff(x[idx])).sum()
            norm = (N - 1) / (len(idx) * k)
            Lk.append(dist * norm)
        L.append(np.mean(Lk))
    # Linear fit in log–log domain
    logL = np.log(L)
    logk = np.log(1. / np.array(list(k_vals)))
    # slope = –D  ?  FD = –slope
    coeffs = np.polyfit(logk, logL, 1)
    return -coeffs[0]


# -- 1.b  Sample entropy  (m=2, r=0.2·std is common in literature) -----------
def sampen(x, m=2, r_ratio=0.2):
    """
    Sample entropy (Richman & Moorman).  Returns np.nan if signal too short.
    """
    N = len(x)
    if N <= m + 1:
        return np.nan
    r = r_ratio * np.std(x)
    def _phi(order):
        count = 0
        for i in range(N - order):
            template = x[i:i+order]
            for j in range(i+1, N - order):
                if np.all(np.abs(template - x[j:j+order]) <= r):
                    count += 1
        return count
    B = _phi(m)
    A = _phi(m + 1)
    if B == 0:
        return np.inf
    return -np.log(A / B) if A else np.inf


# -- 1.c  Teager–Kaiser energy operator statistics ---------------------------
def tkeo_mean_var(x):
    """
    Returns mean and variance of the discrete Teager-Kaiser operator
    """
    if len(x) < 3:
        return (np.nan, np.nan)
    psi = x[1:-1]**2 - x[:-2] * x[2:]
    return np.mean(psi), np.var(psi)

    
def imf_nonlinear_stats(imfs):
    """
    Returns dict with lists of nonlinear features for each IMF.
    """
    stats = {
        'fractal_dim': [],
        'sample_entropy': [],
        'tkeo_mean': [],
        'tkeo_var': []
    }
    for imf in imfs:
        stats['fractal_dim'].append(higuchi_fd(imf))
        stats['sample_entropy'].append(sampen(imf))
        mu, var = tkeo_mean_var(imf)
        stats['tkeo_mean'].append(mu)
        stats['tkeo_var'].append(var)
    return stats
    
    
# -------------------------  CROSS-IMF METRICS  ------------------------------
def imf_energy_ratio(imfs):
    """
    Return vector r_k = E_k / SE of shape (n_imfs,).
    """
    energies = np.sum(imfs**2, axis=1)          # energy per IMF
    tot = energies.sum()
    return energies / tot if tot else energies * np.nan


def imf_corr_matrix(imfs):
    """
    Pearson correlation matrix of shape (n_imfs, n_imfs).
    """
    return np.corrcoef(imfs)


def imf_mode_mixing_index(imfs, fs):
    """
    Returns array of length n_imfs-1 with MMI between successive IMFs.
    """
    mmi = []
    for i in range(len(imfs) - 1):
        # power spectra
        P_i = np.abs(np.fft.rfft(imfs[i]))**2
        P_j = np.abs(np.fft.rfft(imfs[i + 1]))**2
        # make sure arrays have same length (they will)
        overlap = np.minimum(P_i, P_j).sum()
        mmi.append((2 * overlap) / (P_i.sum() + P_j.sum()))
    return np.array(mmi)


    
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
            for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc=f"Processing {os.path.basename(csv_file)}"):
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
                    stats_fd = imf_freq_domain_stats(imfs_comp)
                    stats.update(stats_fd)
                    # stats_nl = imf_nonlinear_stats(imfs_comp)
                    # stats.update(stats_nl)
                    
                    # -------- new cross-IMF stats -------------
                    xstats = {
                        'energy_ratio': imf_energy_ratio(imfs_comp),             # 1-D
                        'corr_matrix':  imf_corr_matrix(imfs_comp),              # 2-D
                        'mode_mixing':  imf_mode_mixing_index(imfs_comp, fs)     # 1-D
                    }
                    stats.update(xstats) 
                    
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
                    for stat_name, stat_val in stats.items():
                      # If it’s a list or an ndarray (1-D or 2-D), store it as a dataset
                      if isinstance(stat_val, (list, np.ndarray)):
                          comp_group.create_dataset(stat_name, data=np.asarray(stat_val))
                      # Otherwise (single number, string, etc.) keep it as an attribute
                      else:
                          comp_group.attrs[stat_name] = stat_val
                          
                # --- NEW: collect into global_records ---
                for comp_idx, stats in enumerate(imf_stats):
                    comp = components[comp_idx]
                    for metric in metrics:
                        values = stats[metric]
                        for imf_idx, val in enumerate(values, start=1):
                            global_records.append({
                                'trace': trace_name,
                                'component': comp,
                                'imf': imf_idx,
                                'metric': metric,
                                'value': val,
                                'category': trace_category
                            })
                            
                print(f"Saved decomposition for trace: {trace_name}")
        print(f"All decompositions saved to: {output_file}")
        
        # Optionally plot random traces
        
        #print("Checking if the file saved correctly...")
        #print(f"File contains the following groups: {list(f_out.keys())}")
        #for group in f_out.keys():
        #    print(f"Group {group} contains the following datasets: {list(f_out[group].keys())}")
        #print("Choosing random traces to plot...")
        
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

n_random_traces = 2
n_samples = 1000


# this will accumulate every (trace,component,imf,metric,value)
global_records = []

# define the exact metrics you want to summarize
metrics = [
    'mean', 'std', 'skewness', 'kurtosis',
    'zero_crossings', 'peak_to_peak',
    'spectral_centroid', 'spectral_bandwidth',
    'spectral_entropy', 'spectral_flatness',
    'freq_modulation_index'
]

# Run for earthquake data
decompose_and_save_emd(
    file_name_hdf5=file_name_merged,
    csv_file=csv_file_merged,
    output_file=output_file_merged,
    n_samples=n_samples,
    plot_random_traces=n_random_traces
)

# Run for noise data
decompose_and_save_emd(
    file_name_hdf5=file_name_noise,
    csv_file=csv_file_noise,
    output_file=output_file_noise,
    n_samples=n_samples,
    plot_random_traces=n_random_traces
)

# =============================================================================
# 5) Build DataFrame & grouped boxplots
# =============================================================================
df = pd.DataFrame(global_records)
cats    = df.category.unique().tolist()
colors = ['C0', 'C1']  # add more if you have more than two categories

n_cats = len(cats)
width  = 0.8 / n_cats
offsets = np.linspace(-0.4 + width/2, 0.4 - width/2, n_cats)

for metric in metrics:
    sub  = df[df.metric == metric]
    imfs = sorted(sub.imf.unique())
    base = np.arange(1, len(imfs) + 1)

    # 1 row, 2 cols: left=linear y, right=log y
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    panel_info = [
        (axes[0], False, 'Linear scale'),
        (axes[1], True,  'Log scale')
    ]

    for ax, use_log, suffix in panel_info:
        # draw each category side-by-side
        for idx, (cat, off) in enumerate(zip(cats, offsets)):
            data = [
                sub[(sub.imf==k) & (sub.category==cat)].value.values
                for k in imfs
            ]
            bp = ax.boxplot(
                data,
                positions=base + off,
                widths=width,
                patch_artist=True,
                showfliers=False,
                notch=False
            )
            # style boxes/whiskers/caps/medians
            for box in bp['boxes']:
                box.set(facecolor=colors[idx], edgecolor='black', linewidth=1)
            for w in bp['whiskers']:
                w.set(color='black', linewidth=1)
            for c in bp['caps']:
                c.set(color='black', linewidth=1)
            for m in bp['medians']:
                m.set(color='black', linewidth=0.5)

        ax.set_xticks(base)
        ax.set_xticklabels(imfs)
        ax.set_xlabel('IMF #')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} ({suffix})')
        ax.grid(alpha=0.3)

        if use_log:
            ax.set_yscale('log')
            # avoid zero or negative values if any:
            ax.set_ylim(bottom=max(sub.value.min(), 1e-3))

    # add legend to the left panel
    handles = [
        mpatches.Patch(facecolor=colors[i],
                       edgecolor='black',
                       label=cats[i])
        for i in range(n_cats)
    ]
    axes[0].legend(handles=handles, loc='upper right')

    plt.tight_layout()
    plt.savefig(f'boxplot_{metric}_linear_vs_log.png', dpi=150)
    plt.close(fig)


