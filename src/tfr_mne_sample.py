import os
import numpy as np
import matplotlib.pyplot as plt
import mne

np.random.seed(42)

def main():
    os.makedirs("results", exist_ok=True)

    # Load dataset
    data_path = mne.datasets.sample.data_path(verbose=True)
    raw_fname = os.path.join(data_path, "MEG", "sample", "sample_audvis_raw.fif")
    event_fname = os.path.join(data_path, "MEG", "sample", "sample_audvis_raw-eve.fif")

    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    raw.pick("eeg")
    raw.filter(1.0, 40.0, verbose=False)

    events = mne.read_events(event_fname)

    event_id = {"Auditory/Left": 1}
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=-0.2,
        tmax=0.8,
        baseline=(-0.2, 0.0),
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )

    epochs = epochs[:40].resample(200)

    # Time–frequency using new API
    freqs = np.linspace(4, 30, 40)
    n_cycles = freqs / 2.0

    power = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=True,
        decim=2,
    )

    ch_name = power.ch_names[0]
    power_ch = power.copy().pick(ch_name)

    figs = power_ch.plot(
        baseline=(-0.2, 0.0),
        mode="logratio",
        show=False,
        title=f"Time–Frequency (Morlet) — {ch_name}",
    )

    fig = figs[0] if isinstance(figs, list) else figs

    out_png = os.path.join("results", "time_frequency_eeg.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Report
    report_path = os.path.join("results", "run_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("EEG Time–Frequency mini-pipeline (MNE sample)\n")
        f.write(f"Epochs used: {len(epochs)}\n")
        f.write(f"Channel: {ch_name}\n")
        f.write(f"Freq range: {freqs[0]:.1f}-{freqs[-1]:.1f} Hz\n")
        f.write(f"Saved figure: {out_png}\n")

    print("Done. Saved:", out_png)


if __name__ == "__main__":
    main()