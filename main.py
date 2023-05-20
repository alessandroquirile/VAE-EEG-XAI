from mne.filter import filter_data
from mne.preprocessing import find_eog_events

from utils import *

if __name__ == '__main__':
    path = 'data_original/'
    eeg = ['Fp1', 'Fp2']
    l_freq = 1
    h_freq = 10

    subjects = get_subjects(path)

    for subject in subjects:
        raw = read_bdf(path='data_original/', subject=subject)
        raw.resample(sfreq=128)

        indices = get_indices_where_video_start(raw)
        for trial, sample_index in enumerate(indices, start=1):
            raw_copy = raw.copy().pick_channels(eeg)
            cropped_raw = crop(raw_copy, sample_index)

            data = raw_copy.get_data()
            sample_rate = get_sample_rate(raw_copy)
            filtered_data = filter_data(data, sample_rate, l_freq, h_freq)

            info = raw_copy.info

            # If this gives problems, disable PyCharm's python SciView
            save_plot('plots', subject, trial, filtered_data, info)

            thresh = calculate_threshold(filtered_data)
            events = find_eog_events(raw_copy, ch_name=eeg, thresh=thresh)
            save_events('events', subject, trial, events, sample_index, sample_rate)
