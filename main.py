from mne.preprocessing import find_eog_events

from savers import save_plot, save_events
from utils import *


def plots_and_events():
    path = 'data_original/'
    l_freq = 1
    h_freq = 10

    subjects = get_subjects(path)
    subjects = ['s01.bdf']  # Todo: for each subjects
    for subject in subjects:
        raw = read_bdf(path, subject)

        # Computing the extrema for each subject (instead of for each trial) handles the scenario in which
        # a subject does not blink at all watching a trial:
        # their extrema would be too close to the baseline and the number of FP blinks would be too large
        subject_lowest_peak, subject_highest_peak = get_extrema(raw, l_freq, h_freq)

        indices = get_indices_where_video_start(raw)
        for trial, index in enumerate(indices, start=1):
            # Crop the raw signal based on the 60-seconds-trial-length at given index
            cropped_raw = crop(raw, index)
            save_plot('plots', l_freq, h_freq, cropped_raw, subject, trial)

            data = cropped_raw.get_data()
            sample_rate = get_sample_rate(cropped_raw)
            filtered_data = filter_data(data, sample_rate, l_freq, h_freq)
            trial_lowest_peak = np.min(filtered_data)
            trial_highest_peak = np.max(filtered_data)

            # s01: 105, s02: 160, s03: 100, s04: 150, s05: 150, s06: 150, s07: 110 ...
            magic_number = 105  # Todo: Depends on the subject
            thresh = calculate_thresh(filtered_data, magic_number)

            # We shall focus on EEG since Fp1, Fp2 are the closest electrodes to the eyes
            events = find_eog_events(cropped_raw, ch_name=EEG, thresh=thresh)

            save_events('events', subject, trial, events, index, sample_rate,
                        subject_lowest_peak, subject_highest_peak, magic_number,
                        trial_lowest_peak, trial_highest_peak,
                        thresh)


if __name__ == '__main__':
    plots_and_events()
