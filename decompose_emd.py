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
import argparse
import time  # Add at the top with other imports
import signal
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
import seaborn as sns

# Suppress the specific NumPy deprecation warning about product
warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*product.*')

def parse_arguments():
    parser = argparse.ArgumentParser(description='EMD Decomposition of Seismic Data')
    parser.add_argument('--eq-hdf5', type=str, required=True,
                      help='Path to the earthquake HDF5 file (chunk2.hdf5)')
    parser.add_argument('--eq-csv', type=str, required=True,
                      help='Path to the earthquake CSV file (chunk2.csv)')
    parser.add_argument('--noise-hdf5', type=str, required=True,
                      help='Path to the noise HDF5 file (chunk1.hdf5)')
    parser.add_argument('--noise-csv', type=str, required=True,
                      help='Path to the noise CSV file (chunk1.csv)')
    parser.add_argument('--output-eq', type=str, default='emd_decompositions_chunk2.h5',
                      help='Output file for earthquake data (default: emd_decompositions_chunk2.h5)')
    parser.add_argument('--output-noise', type=str, default='emd_decompositions_chunk1.h5',
                      help='Output file for noise data (default: emd_decompositions_chunk1.h5)')
    parser.add_argument('--n-samples', type=int, default=None,
                      help='Number of samples to process (default: process all traces)')
    parser.add_argument('--n-random-traces', type=int, default=2,
                      help='Number of random traces to plot (default: 2)')
    parser.add_argument('--trace-number', type=int, default=None,
                      help='Specific trace number to analyze (if not provided, will use random sampling)')
    parser.add_argument('--eq-specific-lines', type=str, default=None,
                      help='Comma-separated list of specific line numbers to process for earthquake data (e.g., "1,5,10")')
    parser.add_argument('--noise-specific-lines', type=str, default=None,
                      help='Comma-separated list of specific line numbers to process for noise data (e.g., "1,5,10")')
    parser.add_argument('--plots-dir', type=str, default='plots',
                      help='Directory to save plots (default: plots)')
    parser.add_argument('--no-plots', action='store_true',
                      help='Disable trace plot generation')
    parser.add_argument('--no-boxplots', action='store_true',
                      help='Disable boxplot generation')
    return parser.parse_args()

def plot_emd_features(data, fs, imfs, n_imfs, title="EMD Features", stats=None):
    ###
    #Draw original 3-component trace, energy-ratio bars, chosen IMFs with
    #stats, correlation-matrix heat-map and mode-mixing-index stem plot.
    #------------------------------------------------------------------
    # `imfs`  list [imfs_E, imfs_N, imfs_Z] with shape
    #            (n_imfs_comp, n_samples) for each component.
    # `stats` list of 3 dicts (one per component) produced in
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
    
    # Create figure with enough space for all subplots
    fig = plt.figure(figsize=(24, 6 * rows))
    gs = fig.add_gridspec(rows, 3, height_ratios=[1.2] + [1] * (rows-1))
    fig.suptitle(title, fontsize=16, y=0.99)

    t = np.arange(len(data)) / fs
    components = ['e', 'n', 'z']

    for comp_idx, comp in enumerate(components):
        # ----------- row 0: original component -----------
        ax0 = fig.add_subplot(gs[0, comp_idx])
        ax0.plot(t, data[:, comp_idx], 'k', lw=0.5)
        ax0.set_title(f'{comp.upper()} component', pad=20, fontsize=12)
        ax0.set_xlabel('Time (s)', fontsize=10)
        ax0.set_ylabel('Amplitude', fontsize=10)
        ax0.grid(True)
        ax0.tick_params(labelsize=9)

        # ----------- row 1: energy-ratio bars -----------
        if stats is not None:
            er = stats[comp_idx]['energy_ratio']
            ax_er = fig.add_subplot(gs[1, comp_idx])
            ax_er.bar(np.arange(len(er)) + 1, er, color='slategray')
            ax_er.set_title('Energy ratio', pad=20, fontsize=12)
            ax_er.set_xlabel('IMF #', fontsize=10)
            ax_er.set_ylabel('Fraction', fontsize=10)
            ax_er.set_ylim(0, 1.05)
            ax_er.tick_params(labelsize=9)

        # ----------- rows 2 N: IMFs -----------
        for i in range(n_imfs_to_plot):
            if i >= imfs[comp_idx].shape[0]:
                continue
            ax = fig.add_subplot(gs[i + 2, comp_idx])
            ax.plot(t, imfs[comp_idx][i], 'b', lw=0.5)
            ax.set_title(f'IMF {i + 1}', pad=20, fontsize=12)
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.grid(True)
            ax.tick_params(labelsize=9)

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
                        fontsize=8, va='top', ha='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # ----------- row N+1: correlation heat-map -----------
        if stats is not None:
            ax_cm = fig.add_subplot(gs[n_imfs_to_plot + 2, comp_idx])
            cmat  = stats[comp_idx]['corr_matrix']
            im    = ax_cm.imshow(cmat, vmin=-1, vmax=1,
                                 cmap='seismic', aspect='auto')
            ax_cm.set_title('IMF correlation', pad=20, fontsize=12)
            ax_cm.set_xlabel('IMF', fontsize=10)
            ax_cm.set_ylabel('IMF', fontsize=10)
            ax_cm.tick_params(labelsize=9)
            fig.colorbar(im, ax=ax_cm, shrink=0.7)

        # ----------- row N+2: mode-mixing index -----------
        if stats is not None:
            mmi  = stats[comp_idx]['mode_mixing']
            ax_m = fig.add_subplot(gs[n_imfs_to_plot + 3, comp_idx])
            ax_m.stem(np.arange(1, len(mmi) + 1), mmi, basefmt=' ')
            ax_m.set_title('Mode-mixing index', pad=20, fontsize=12)
            ax_m.set_xlabel('Between IMF i & i+1', fontsize=10)
            ax_m.set_ylabel('MMI', fontsize=10)
            ax_m.set_ylim(0, 1.05)
            ax_m.tick_params(labelsize=9)

    # Adjust layout manually
    plt.subplots_adjust(
        top=0.95,
        bottom=0.05,
        left=0.05,
        right=0.95,
        hspace=0.4,
        wspace=0.3
    )
    
    # Save figure with tight bounding box
    plt.savefig(f"emd_features_{title}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


# -------------------------  NON-LINEAR IMF METRICS  --------------------------


# -- 1.a  Fractal dimension  (Higuchi's method robust for short 1-D data) --
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
    # Linear fit in log?log domain
    logL = np.log(L)
    logk = np.log(1. / np.array(list(k_vals)))
    # slope = ?D  ?  FD = ?slope
    coeffs = np.polyfit(logk, logL, 1)
    return -coeffs[0]


# -- 1.b  Sample entropy  (m=2, r=0.2?std is common in literature) -----------
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


# -- 1.c  Teager?Kaiser energy operator statistics ---------------------------
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

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("EMD processing timed out")
    
    # Set the signal handler and a timer
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

def process_component(data_comp, fs, comp_idx, trace_name, trace_category, mag, timeout_seconds=1):
    """Process a single component of a trace"""
    my_emd = EMD(max_imf=10)
    try:
        with timeout(timeout_seconds):
            imfs_comp = my_emd.emd(data_comp)
        imfs_comp = imfs_comp[:10]  # Enforce the limit
    except TimeoutError:
        print(f"    EMD processing timed out for trace {trace_name}, component {comp_idx+1}")
        return None, None
    
    # Stats calculation
    stats = imf_time_domain_stats(imfs_comp)
    stats_fd = imf_freq_domain_stats(imfs_comp)
    stats.update(stats_fd)
    
    # Cross-IMF stats
    xstats = {
        'energy_ratio': imf_energy_ratio(imfs_comp),
        'corr_matrix': imf_corr_matrix(imfs_comp),
        'mode_mixing': imf_mode_mixing_index(imfs_comp, fs)
    }
    stats.update(xstats)
    
    return imfs_comp, stats

def create_emd_feature_vector(imfs_list, n_imfs=10):
    """
    Create a feature vector from EMD decomposition results.
    
    Args:
        imfs_list: List of IMFs for each component (E, N, Z)
        n_imfs: Number of IMFs to use (default: 10)
        
    Returns:
        numpy array of shape (n_imfs * 2 * 3,) containing:
        - For each component (E, N, Z)
        - For each IMF (1 to n_imfs)
        - Mean and kurtosis of the signal
    """
    features = []
    
    for comp_imfs in imfs_list:
        if comp_imfs is None:
            # If component failed, pad with zeros
            features.extend([0.0] * (n_imfs * 2))
            continue
            
        for i in range(min(n_imfs, len(comp_imfs))):
            imf = comp_imfs[i]
            # Calculate mean
            mean_val = np.mean(imf)
            # Calculate kurtosis
            kurt_val = kurtosis(imf)
            features.extend([mean_val, kurt_val])
            
        # Pad with zeros if we have fewer IMFs than n_imfs
        if len(comp_imfs) < n_imfs:
            features.extend([0.0] * ((n_imfs - len(comp_imfs)) * 2))
    
    return np.array(features)

def process_trace(args):
    """Process a single trace with all its components sequentially"""
    trace_name, trace_category, mag, file_path = args
    
    # Load data
    with h5py.File(file_path, 'r') as h5f:
        ds = h5f.get(f"data/{trace_name}")
        if ds is None:
            return None
        
        data = np.array(ds)
        fs = float(ds.attrs.get('sampling_rate', 100.0))
    
    # Process components sequentially
    imfs = []
    imf_stats = []
    component_success = []  # Track which components were processed successfully
    component_failures = []  # Track which components failed
    
    for comp_idx in range(data.shape[1]):
        result = process_component(data[:, comp_idx], fs, comp_idx, trace_name, trace_category, mag)
        if result is None:
            imfs.append(None)
            imf_stats.append(None)
            component_success.append(False)
            component_failures.append(comp_idx)
        else:
            imfs_comp, stats = result
            imfs.append(imfs_comp)
            imf_stats.append(stats)
            component_success.append(True)
    
    # Create feature vector
    feature_vector = create_emd_feature_vector(imfs)
    
    return {
        'trace_name': trace_name,
        'data': data,
        'fs': fs,
        'mag': mag,
        'trace_category': trace_category,
        'imfs': imfs,
        'imf_stats': imf_stats,
        'component_success': component_success,
        'component_failures': component_failures,
        'feature_vector': feature_vector
    }

def plot_trace(args):
    """Generate plots for a single trace"""
    trace_name, data, fs, imfs, imf_stats, plots_dir = args
    try:
        # Check if we have any valid components
        valid_components = [i for i, imf in enumerate(imfs) if imf is not None]
        if not valid_components:
            print(f"Skipping plot for {trace_name}: All components failed EMD processing")
            return False
            
        # Create figure with enough space for all subplots
        n_imfs_to_plot = min(10, max(imf.shape[0] for imf in imfs if imf is not None))
        rows = n_imfs_to_plot + 4
        fig = plt.figure(figsize=(24, 6 * rows))
        gs = fig.add_gridspec(rows, 3, height_ratios=[1.2] + [1] * (rows-1))
        fig.suptitle(f"EMD_Decomposition_{trace_name}", fontsize=16, y=0.99)

        t = np.arange(len(data)) / fs
        components = ['e', 'n', 'z']

        for comp_idx, comp in enumerate(components):
            # ----------- row 0: original component -----------
            ax0 = fig.add_subplot(gs[0, comp_idx])
            ax0.plot(t, data[:, comp_idx], 'k', lw=0.5)
            ax0.set_title(f'{comp.upper()} component', pad=20, fontsize=12)
            ax0.set_xlabel('Time (s)', fontsize=10)
            ax0.set_ylabel('Amplitude', fontsize=10)
            ax0.grid(True)
            ax0.tick_params(labelsize=9)

            # Skip this component if it failed
            if imfs[comp_idx] is None:
                ax0.text(0.5, 0.5, 'EMD Processing Failed', 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax0.transAxes,
                        fontsize=12,
                        color='red')
                continue

            # ----------- row 1: energy-ratio bars -----------
            if imf_stats[comp_idx] is not None and 'energy_ratio' in imf_stats[comp_idx]:
                er = imf_stats[comp_idx]['energy_ratio']
                if len(er) > 0:
                    ax_er = fig.add_subplot(gs[1, comp_idx])
                    ax_er.bar(np.arange(len(er)) + 1, er, color='slategray')
                    ax_er.set_title('Energy ratio', pad=20, fontsize=12)
                    ax_er.set_xlabel('IMF #', fontsize=10)
                    ax_er.set_ylabel('Fraction', fontsize=10)
                    ax_er.set_ylim(0, 1.05)
                    ax_er.tick_params(labelsize=9)

            # ----------- rows 2 N: IMFs -----------
            for i in range(n_imfs_to_plot):
                if i >= imfs[comp_idx].shape[0]:
                    continue
                ax = fig.add_subplot(gs[i + 2, comp_idx])
                ax.plot(t, imfs[comp_idx][i], 'b', lw=0.5)
                ax.set_title(f'IMF {i + 1}', pad=20, fontsize=12)
                ax.set_xlabel('Time (s)', fontsize=10)
                ax.set_ylabel('Amplitude', fontsize=10)
                ax.grid(True)
                ax.tick_params(labelsize=9)

                # ----- per-IMF stat overlay -----
                if imf_stats[comp_idx] is not None and i < len(imf_stats[comp_idx]['mean']):
                    st = imf_stats[comp_idx]
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
                            fontsize=8, va='top', ha='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # ----------- row N+1: correlation heat-map -----------
            if imf_stats[comp_idx] is not None and 'corr_matrix' in imf_stats[comp_idx]:
                cmat = imf_stats[comp_idx]['corr_matrix']
                if cmat.size > 0 and not np.all(cmat == cmat[0,0]):  # Check if matrix is not constant
                    ax_cm = fig.add_subplot(gs[n_imfs_to_plot + 2, comp_idx])
                    im = ax_cm.imshow(cmat, vmin=-1, vmax=1,
                                     cmap='seismic', aspect='auto')
                    ax_cm.set_title('IMF correlation', pad=20, fontsize=12)
                    ax_cm.set_xlabel('IMF', fontsize=10)
                    ax_cm.set_ylabel('IMF', fontsize=10)
                    ax_cm.tick_params(labelsize=9)
                    fig.colorbar(im, ax=ax_cm, shrink=0.7)

            # ----------- row N+2: mode-mixing index -----------
            if imf_stats[comp_idx] is not None and 'mode_mixing' in imf_stats[comp_idx]:
                mmi = imf_stats[comp_idx]['mode_mixing']
                if len(mmi) > 0:
                    ax_m = fig.add_subplot(gs[n_imfs_to_plot + 3, comp_idx])
                    ax_m.stem(np.arange(1, len(mmi) + 1), mmi, basefmt=' ')
                    ax_m.set_title('Mode-mixing index', pad=20, fontsize=12)
                    ax_m.set_xlabel('Between IMF i & i+1', fontsize=10)
                    ax_m.set_ylabel('MMI', fontsize=10)
                    ax_m.set_ylim(0, 1.05)
                    ax_m.tick_params(labelsize=9)

        # Adjust layout manually
        plt.subplots_adjust(
            top=0.95,
            bottom=0.05,
            left=0.05,
            right=0.95,
            hspace=0.4,
            wspace=0.3
        )
        
        # Save figure with tight bounding box
        plot_path = os.path.join(plots_dir, f"emd_features_{trace_name}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"Error generating plot for {trace_name}: {str(e)}")
        if 'fig' in locals():
            plt.close(fig)
        return False

def create_boxplots(global_records, plots_dir):
    """Generate boxplots for all metrics with both linear and log scales, showing all components in subplots"""
    if not global_records:
        print("No data available for boxplots")
        return

    # Convert records to DataFrame
    df = pd.DataFrame(global_records)
    
    # Get total number of unique traces by category
    n_eq_traces = df[df['category'] == 'earthquake_local']['trace'].nunique()
    n_noise_traces = df[df['category'] == 'noise']['trace'].nunique()
    n_total_traces = df['trace'].nunique()
    
    # Create a figure for each metric
    for metric in df['metric'].unique():
        # Create figure with 1 row (log scale only) and 3 columns (E, N, Z)
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        # Plot each component
        for comp_idx, component in enumerate(['e', 'n', 'z']):
            # Filter data for this metric and component
            metric_data = df[(df['metric'] == metric) & (df['component'] == component)]
            if len(metric_data) == 0:
                continue
            n_eq_comp = metric_data[metric_data['category'] == 'earthquake_local']['trace'].nunique()
            n_noise_comp = metric_data[metric_data['category'] == 'noise']['trace'].nunique()
            offset = 0.15  # adjust as needed
            category_offsets = {'earthquake_local': -offset, 'noise': offset}
            markers = {'earthquake_local': 'o', 'noise': 's'}
            grouped = metric_data.groupby(['imf', 'category'])['value'].agg(['mean', 'std']).reset_index()
            # Log scale only
            for cat in grouped['category'].unique():
                cat_data = grouped[grouped['category'] == cat].sort_values('imf')
                x = cat_data['imf'] + category_offsets.get(cat, 0)
                mask = cat_data['mean'] > 0
                x_log = x[mask]
                mean_log = cat_data['mean'][mask]
                yerr_log = (cat_data['std'] / np.sqrt(cat_data['mean'].count()))[mask]
                axes[comp_idx].errorbar(
                    x_log, mean_log, yerr=yerr_log,
                    fmt=markers.get(cat, 'o')+'-', capsize=3, label=cat
                )
            axes[comp_idx].set_title(
                f'{component.upper()} Component (Log Scale)\n'
                f'EQ: {n_eq_comp} traces, Noise: {n_noise_comp} traces'
            )
            axes[comp_idx].set_xlabel('IMF Number')
            axes[comp_idx].set_ylabel(f'{metric} (log scale)')
            axes[comp_idx].tick_params(axis='x', rotation=45)
            axes[comp_idx].set_yscale('log')
            axes[comp_idx].legend()
            # Ensure all IMF numbers are shown on the x-axis
            axes[comp_idx].set_xticks(range(1, 11))
            axes[comp_idx].set_xticklabels([str(i) for i in range(1, 11)])
        # Add overall title with total number of traces by category
        fig.suptitle(
            f'Errorbar of {metric} by Component (Log Scale Only)\n'
            f'Total Traces: {n_total_traces} (EQ: {n_eq_traces}, Noise: {n_noise_traces})',
            fontsize=16,
            y=0.95
        )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(plots_dir, f'errorbar_{metric}_all_components.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved errorbar for {metric} with all components (EQ: {n_eq_traces}, Noise: {n_noise_traces} traces)")

def decompose_and_save_emd(
    file_name_hdf5, 
    csv_file, 
    output_file, 
    n_samples=None, 
    components=['e', 'n', 'z'],
    plot_random_traces=5,
    trace_number=None,
    specific_lines=None,
    timeout_log=None,
    plots_dir='plots',
    generate_plots=True,
    generate_boxplots=True,
    return_features=False
):
    print(f"Processing: {csv_file}")
    dtype_dict = {
        'trace_name': 'category',
        'trace_category': 'category',
        'source_magnitude': 'float32'
    }
    df = pd.read_csv(csv_file, usecols=['trace_name', 'trace_category', 'source_magnitude'], dtype=dtype_dict)
    
    if timeout_log is None:
        timeout_log = []
    
    if trace_number is not None:
        if trace_number >= len(df):
            raise ValueError(f"Trace number {trace_number} is out of range. Total traces available: {len(df)}")
        sampled_indices = [trace_number]
        print(f"Analyzing specific trace number: {trace_number}")
    elif specific_lines is not None:
        try:
            line_numbers = [int(x.strip()) for x in specific_lines.split(',')]
            # Validate line numbers
            invalid_lines = [x for x in line_numbers if x < 0 or x >= len(df)]
            if invalid_lines:
                raise ValueError(f"Invalid line numbers found: {invalid_lines}. Valid range is 0 to {len(df)-1}")
            sampled_indices = line_numbers
            print(f"Analyzing specific lines: {line_numbers}")
        except ValueError as e:
            raise ValueError(f"Error parsing specific lines: {e}")
    else:
        # Process all traces if n_samples is None, otherwise process first n_samples
        sampled_indices = list(range(len(df))) if n_samples is None else list(range(min(n_samples, len(df))))
        print(f"Processing {'all' if n_samples is None else n_samples} traces")
    
    sampled_df = df.iloc[sampled_indices]
    
    # Initialize lists for features if return_features is True
    if return_features:
        features = []
        labels = []
        targets = []
    
    with h5py.File(output_file, 'w') as f_out:
        f_out.attrs['creation_date'] = np.bytes_(pd.Timestamp.now().isoformat())
        # Store n_samples as -1 if None (indicating all traces)
        f_out.attrs['n_samples'] = -1 if n_samples is None else n_samples
        f_out.attrs['total_traces_available'] = len(df)
        f_out.attrs['traces_processed'] = len(sampled_indices)
        
        # Process traces in parallel
        n_processes = min(cpu_count(), len(sampled_df))
        print(f"Using {n_processes} processes for parallel processing")
        
        with Pool(processes=n_processes) as pool:
            process_args = [(row['trace_name'], row['trace_category'], 
                           row.get('source_magnitude', 0), file_name_hdf5) 
                          for _, row in sampled_df.iterrows()]
            
            results = list(tqdm(
                pool.imap(process_trace, process_args),
                total=len(process_args),
                desc=f"Processing {os.path.basename(csv_file)}"
            ))
        
        # Save results and prepare plot arguments
        plot_args = []
        for result in results:
            if result is None:
                continue
                
            trace_name = result['trace_name']
            data = result['data']
            fs = result['fs']
            mag = result['mag']
            trace_category = result['trace_category']
            imfs = result['imfs']
            imf_stats = result['imf_stats']
            component_success = result['component_success']
            component_failures = result['component_failures']
            feature_vector = result['feature_vector']
            
            # Store features if return_features is True
            if return_features:
                features.append(feature_vector)
                labels.append(1 if trace_category == 'earthquake_local' else 0)
                targets.append(mag)
            
            # Prepare plot arguments with plots_dir
            if generate_plots:
                plot_args.append((trace_name, data, fs, imfs, imf_stats, plots_dir))
            
            # Saving
            # trace_group = f_out.create_group(trace_name)
            # trace_group.attrs['sampling_rate'] = fs
            # trace_group.attrs['magnitude'] = mag
            # trace_group.attrs['category'] = trace_category
            # trace_group.create_dataset('original_data', data=data)
            
            # for comp_idx, (comp_name, imf_data, stats, success) in enumerate(zip(components, imfs, imf_stats, component_success)):
            #     if not success or imf_data is None or stats is None:
            #         print(f"Skipping failed component {comp_name} for trace {trace_name}")
            #         continue
                    
            #     try:
            #         comp_group = trace_group.create_group(f'component_{comp_name}')
            #         if imf_data is not None and len(imf_data) > 0:
            #             comp_group.create_dataset('imfs', data=imf_data)
            #             for stat_name, stat_val in stats.items():
            #                 if isinstance(stat_val, (list, np.ndarray)):
            #                     comp_group.create_dataset(stat_name, data=np.asarray(stat_val))
            #                 else:
            #                     comp_group.attrs[stat_name] = stat_val
            #     except Exception as e:
            #         print(f"Error saving component {comp_name} for trace {trace_name}: {str(e)}")
            #         continue
            
            # Global records - only collect if boxplots are enabled
            if generate_boxplots:
                for comp_idx, (stats, success) in enumerate(zip(imf_stats, component_success)):
                    if not success or stats is None:
                        continue
                    comp = components[comp_idx]
                    for metric in metrics:
                        if metric not in stats:
                            continue
                        values = stats[metric]
                        if values is None:
                            continue
                        
                        for imf_idx, val in enumerate(values, start=1):
                            global_records.append({
                                'trace': trace_name,
                                'component': comp,
                                'imf': imf_idx,
                                'metric': metric,
                                'value': val,
                                'category': trace_category
                            })
        
        # Generate plots in parallel if enabled
        if generate_plots and plot_args:
            print("\nGenerating plots in parallel...")
            n_plot_processes = min(cpu_count(), len(plot_args))
            with Pool(processes=n_plot_processes) as pool:
                list(tqdm(
                    pool.imap(plot_trace, plot_args),
                    total=len(plot_args),
                    desc="Generating plots"
                ))
    
    print(f"All decompositions saved to: {output_file}")
    
    # Plot random traces if requested and plots are enabled
    if generate_plots and plot_random_traces > 0:
        random_trace_names = np.random.choice(list(f_out.keys()), 
                                            size=min(plot_random_traces, len(f_out.keys())), 
                                            replace=False)
        print(f"Plotting EMD decompositions for the following traces: {random_trace_names}")
        
        # Prepare random trace plot arguments
        random_plot_args = []
        for trace_name in random_trace_names:
            trace_group = f_out[trace_name]
            imfs = []
            for comp in components:
                imfs.append(trace_group[f'component_{comp}']['imfs'][:])
            random_plot_args.append((
                trace_name,
                trace_group['original_data'][:],
                trace_group.attrs['sampling_rate'],
                imfs,
                imf_stats,
                plots_dir
            ))
        
        # Generate random trace plots in parallel
        if random_plot_args:
            n_random_plot_processes = min(cpu_count(), len(random_plot_args))
            with Pool(processes=n_random_plot_processes) as pool:
                list(tqdm(
                    pool.imap(plot_trace, random_plot_args),
                    total=len(random_plot_args),
                    desc="Generating random trace plots"
                ))

    # Generate boxplots if enabled
    if generate_boxplots and global_records:
        print("\nGenerating boxplots...")
        create_boxplots(global_records, plots_dir)
    
    # Return features if requested
    if return_features:
        return np.array(features), np.array(labels), np.array(targets)

# --- Configuration for earthquake and noise data ---
if __name__ == "__main__":
    args = parse_arguments()
    
    # Ensure model directory exists
    os.makedirs("saved_models", exist_ok=True)
    
    # Create plots directory if it doesn't exist
    os.makedirs(args.plots_dir, exist_ok=True)

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

    # Initialize timeout logs
    eq_timeout_log = []
    noise_timeout_log = []

    # Run for earthquake data
    print("-----------------  Running for EARTHQUAKE data...-------------")
    eq_output_file = os.path.join(args.plots_dir, args.output_eq)
    decompose_and_save_emd(
        file_name_hdf5=args.eq_hdf5,
        csv_file=args.eq_csv,
        output_file=eq_output_file,
        n_samples=args.n_samples,
        plot_random_traces=args.n_random_traces,
        trace_number=args.trace_number,
        specific_lines=args.eq_specific_lines,
        timeout_log=eq_timeout_log,
        plots_dir=args.plots_dir,
        generate_plots=not args.no_plots,
        generate_boxplots=not args.no_boxplots
    )

    # Run for noise data
    print("********************  Running for NOISE data...********************")
    noise_output_file = os.path.join(args.plots_dir, args.output_noise)
    decompose_and_save_emd(
        file_name_hdf5=args.noise_hdf5,
        csv_file=args.noise_csv,
        output_file=noise_output_file,
        n_samples=args.n_samples,
        plot_random_traces=args.n_random_traces,
        trace_number=args.trace_number,
        specific_lines=args.noise_specific_lines,
        timeout_log=noise_timeout_log,
        plots_dir=args.plots_dir,
        generate_plots=not args.no_plots,
        generate_boxplots=not args.no_boxplots
    )

    if eq_timeout_log or noise_timeout_log:
        print("\nFinal Timeout Summary:")
        print("=====================")
        if eq_timeout_log:
            print("\nEarthquake Data Timeouts:")
            for entry in eq_timeout_log:
                print(f"Trace: {entry['trace_name']}, Component: {entry['component']}, Magnitude: {entry['magnitude']}")
            print(f"Total earthquake timeouts: {len(eq_timeout_log)}")
        
        if noise_timeout_log:
            print("\nNoise Data Timeouts:")
            for entry in noise_timeout_log:
                print(f"Trace: {entry['trace_name']}, Component: {entry['component']}")
            print(f"Total noise timeouts: {len(noise_timeout_log)}")
        
        print(f"\nTotal timeouts across all data: {len(eq_timeout_log) + len(noise_timeout_log)}")


