import os
import numpy as np
import matplotlib.pyplot as plt
import mne

def main():
    os.makedirs("results", exist_ok=True)

    data_path = mne.datasets.sample.data_path(verbose=False)
    raw_fname = os.path.join(data_path, "MEG", "sample", "sample_audvis_raw.fif")
    event_fname = os.path.join(data_path, "MEG", "sample", "sample_audvis_raw-eve.fif")

    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    raw.pick("eeg")
    raw.filter(0.5, 40.0, verbose=False)

    events = mne.read_events(event_fname)
    event_id = {"Auditory/Left": 1, "Auditory/Right": 2}

    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=-0.2, tmax=0.8,
        baseline=(-0.2, 0.0),
        preload=True, verbose=False,
        reject_by_annotation=True,
    ).resample(250)

    # Use a stable channel if available
    preferred = "EEG 014"
    ch = preferred if preferred in epochs.ch_names else epochs.ch_names[0]

    # --- ERP (Evoked) ---
    epochs_L = epochs["Auditory/Left"][:120]
    epochs_R = epochs["Auditory/Right"][:120]

    evk_L = epochs_L.average()
    evk_R = epochs_R.average()

    # Extract channel waveform (Volts -> microvolts)
    t = evk_L.times
    yL = evk_L.copy().pick(ch).data[0] * 1e6
    yR = evk_R.copy().pick(ch).data[0] * 1e6

    # --- TFR Difference (Right - Left), baseline logratio ---
    freqs = np.linspace(4, 40, 50)
    n_cycles = freqs / 2.0

    tfr_L = epochs_L.compute_tfr(
        method="morlet", freqs=freqs, n_cycles=n_cycles,
        return_itc=False, average=True, decim=1
    )
    tfr_R = epochs_R.compute_tfr(
        method="morlet", freqs=freqs, n_cycles=n_cycles,
        return_itc=False, average=True, decim=1
    )

    baseline = (-0.2, 0.0)
    tfr_L.apply_baseline(baseline=baseline, mode="logratio")
    tfr_R.apply_baseline(baseline=baseline, mode="logratio")

    # pick channel and compute diff map
    tfr_L_ch = tfr_L.copy().pick(ch)
    tfr_R_ch = tfr_R.copy().pick(ch)
    diff = (tfr_R_ch.data[0] - tfr_L_ch.data[0])  # (freq, time)

    tf_times = tfr_L.times
    tf_freqs = tfr_L.freqs

    vmax = np.nanpercentile(np.abs(diff), 98)
    vmin = -vmax

    # --- Figure layout: ERP (top) + TFR diff (bottom) ---
    fig = plt.figure(figsize=(10, 7))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t, yL, label="Auditory Left")
    ax1.plot(t, yR, label="Auditory Right")
    ax1.axvline(0, linewidth=1)
    ax1.axhline(0, linewidth=0.8)
    ax1.set_title(f"ERP at {ch} (baseline corrected)")
    ax1.set_ylabel("Amplitude (µV)")
    ax1.legend(loc="best")
    ax1.set_xlim(t[0], t[-1])

    ax2 = fig.add_subplot(2, 1, 2)
    im = ax2.imshow(
        diff,
        origin="lower",
        aspect="auto",
        extent=[tf_times[0], tf_times[-1], tf_freqs[0], tf_freqs[-1]],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="bicubic",
    )
    ax2.axvline(0, linewidth=1)
    ax2.set_title("TFR Difference (Right − Left), logratio baseline")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")

    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label("Power difference (logratio)")

    plt.tight_layout()
    out_png = os.path.join("results", "erp_tfr_figure.png")
    fig.savefig(out_png, dpi=400)
    plt.close(fig)

    # mini report
    with open(os.path.join("results", "erp_tfr_report.txt"), "w", encoding="utf-8") as f:
        f.write("ERP + TFR figure\n")
        f.write(f"Channel: {ch}\n")
        f.write(f"Epochs L/R: {len(epochs_L)} / {len(epochs_R)}\n")
        f.write("Outputs: results/erp_tfr_figure.png, results/erp_tfr_report.txt\n")

    print("Saved:", out_png)

if __name__ == "__main__":
    main()