import numpy as np
from matplotlib import pyplot as plt
from mne.filter import filter_data
from mne.preprocessing import find_eog_events
from mne.viz import plot_raw

from utils import *

if __name__ == '__main__':
    raw = read_bdf(path='data_original/', subject='s01.bdf')
    raw.resample(sfreq=128, verbose=False)

    indices = get_indices_where_video_start(raw)

    n_trial = 21
    index_trial = indices[n_trial - 1]

    eeg = ['Fp1', 'Fp2']
    raw.pick_channels(eeg)  # We will focus on EEG channels only

    raw = crop(raw, index_trial)

    l_freq = 1.0
    h_freq = 10.0
    filtered_data = filter_data(raw.get_data(), sfreq=get_sample_rate(raw), l_freq=l_freq, h_freq=h_freq)
    raw_filtered = mne.io.RawArray(filtered_data, raw.info)
    plot_raw(raw_filtered, duration=60, scalings=20e-5)
    # save(plt, 'plot.png')
    plt.show()

    thresh = (np.max(filtered_data) - np.min(filtered_data)) / 2
    events = find_eog_events(raw, ch_name=eeg, thresh=thresh)  # On the raw signal!
    events_in_seconds = (events[:, 0] - index_trial) / get_sample_rate(raw)
    print('Events', events, 'at second', events_in_seconds)
