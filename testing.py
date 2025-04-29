import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import matplotlib.ticker as ticker

# %% File paths and parameters

file_name = r"C:\Users\cadas\Box\ECSTATIC - General\02 datasets (public)\STanford EArthquake Dataset (STEAD)\chunk2.hdf5"
csv_file  = r"C:\Users\cadas\Box\ECSTATIC - General\02 datasets (public)\STanford EArthquake Dataset (STEAD)\chunk2.csv"

chunksize = 10000
nrows      = 10000
plotting   = True     # Toggle all plotting on/off
quiet      = False    # If True, skip waveform+PSD plots

# %% Storage
max_amp_e = []
max_amp_n = []
max_amp_z = []
psd_data  = []  # [(f, Pxx_e, Pxx_n, Pxx_z), ...]

# %% Loop over CSV in chunks
for chunk in pd.read_csv(csv_file, chunksize=chunksize, nrows=nrows):
    # --- filter to local earthquakes within 20 km and mag > 3, for example
    chunk = chunk[
        (chunk.trace_category == 'earthquake_local')   &
        (chunk.source_distance_km <= 20) &
        (chunk.source_magnitude > 1)
    ]
    if chunk.empty:
        print("No events found in this chunk")
        print(chunk.trace_category)
        continue

    ev_list = chunk['trace_name'].tolist()
    with h5py.File(file_name, 'r') as dtfl:
        for trace_name in ev_list:
            ds = dtfl.get(f"data/{trace_name}")
            if ds is None:
                continue

            data = np.array(ds)   # shape (n_samples, 3)

            # --- sampling rate (Hz)
            fs = float(ds.attrs.get('sampling_rate', 100.0))
            n = data.shape[0]
            t = np.arange(n) / fs  # time axis (s)

            # --- max absolute amplitudes
            max_amp_e.append(np.max(np.abs(data[:, 0])))
            max_amp_n.append(np.max(np.abs(data[:, 1])))
            max_amp_z.append(np.max(np.abs(data[:, 2])))

            # --- compute Welch PSD with at least two segments
            #    use a 4 s window (or shorter if record is shorter)
            nperseg = min(int(fs*4), n)
            noverlap = nperseg // 2
            f_e, Pxx_e = welch(data[:, 0], fs=fs,
                               nperseg=nperseg, noverlap=noverlap)
            f_n, Pxx_n = welch(data[:, 1], fs=fs,
                               nperseg=nperseg, noverlap=noverlap)
            f_z, Pxx_z = welch(data[:, 2], fs=fs,
                               nperseg=nperseg, noverlap=noverlap)
            psd_data.append((f_z, Pxx_e, Pxx_n, Pxx_z))

            if quiet:
                continue

            # --- Plotting
            if plotting:
                fig, axes = plt.subplots(5, 1,
                                         figsize=(8, 15),
                                         dpi=150,
                                         sharex=True)
                # Waveforms E, N, Z
                labels = ['E', 'N', 'Z']
                for i, ax in enumerate(axes[:3]):
                    ax.plot(t, data[:, i], 'k', lw=0.5)
                    ax.set_ylabel(f'{labels[i]} counts')
                    # arrival lines if valid
                    for name, color in (
                        ('p_arrival_sample', 'b'),
                        ('s_arrival_sample', 'r'),
                        ('coda_end_sample', 'aqua')
                    ):
                        idx = ds.attrs.get(name, None)
                        if isinstance(idx, (int, float)) and 0 < idx < n:
                            ax.axvline(idx/fs, color=color, lw=1)
                    ax.grid(True)

                # Combined trace
                ax4 = axes[3]
                ax4.plot(t, data[:, 0], label='E', color='C0', lw=0.5)
                ax4.plot(t, data[:, 1], label='N', color='C1', lw=0.5)
                ax4.plot(t, data[:, 2], label='Z', color='C2', lw=0.5)
                ax4.set_ylabel('Counts')
                ax4.legend()
                ax4.grid(True)

                # PSD (log–log)
                ax5 = axes[4]
                ax5.loglog(f_e, Pxx_e, label='E')
                ax5.loglog(f_n, Pxx_n, label='N')
                ax5.loglog(f_z, Pxx_z, label='Z')
                ax5.set_xlabel('Frequency (Hz)')
                ax5.set_ylabel('PSD (counts²/Hz)')
                ax5.legend()
                ax5.grid(True, which='both', ls='--', lw=0.5)

                plt.tight_layout()
                plt.show()

# %% Convert amplitude lists to arrays for further stats
max_amp_e = np.array(max_amp_e)
max_amp_n = np.array(max_amp_n)
max_amp_z = np.array(max_amp_z)

# Now max_amp_* and psd_data are ready for your downstream analysis.
