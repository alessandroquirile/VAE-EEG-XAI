from mne.preprocessing import find_eog_events

from savers import save_plot, save_events
from utils import *

if __name__ == '__main__':
    path = 'data_original/'
    sample_rate = 128
    l_freq = 1
    h_freq = 10

    subjects = get_subjects(path)
    for subject in subjects:
        raw = read_bdf(path, subject)
        # raw.resample(sample_rate)

        # Computing the extrema for each subject (instead of for each trial) handles the scenario in which
        # a subject does not blink at all watching a trial:
        # their extrema would be too close to the baseline and the number of FP blinks would be too large
        lowest_peak, highest_peak = get_extrema(raw, l_freq, h_freq)

        indices = get_indices_where_video_start(raw)
        for trial, index in enumerate(indices, start=1):
            # Crop the raw signal based on the 60-seconds-trial-length at given index
            cropped_raw = crop(raw, index)
            save_plot('plots', l_freq, h_freq, cropped_raw, subject, trial)
            thresh = calculate_threshold(lowest_peak, highest_peak)
            # We shall focus on EEG since Fp1, Fp2 are the closest electrodes to the eyes
            events = find_eog_events(cropped_raw, ch_name=EEG, thresh=thresh)
            save_events('events', subject, trial, events, index, sample_rate)
