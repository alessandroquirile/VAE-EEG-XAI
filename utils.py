import os

import mne.io
import numpy as np
from matplotlib import pyplot as plt
from mne.filter import filter_data
from mne.io import read_raw_bdf
from mne.viz import plot_raw

from deap_constants import *
from eeg_constants import EEG, EOG, MISC, STIM_CHANNEL


def read_bdf(path: str, subject: str) -> mne.io.Raw:
    raw = read_raw_bdf(path + subject, eog=EOG, misc=MISC, stim_channel=STIM_CHANNEL, preload=True, verbose=False)
    return raw


def get_indices_where_video_start(raw: mne.io.Raw) -> list[int]:
    sample_rate = get_sample_rate(raw)
    n_samples_in_1_minute = _get_number_of_samples_in_1_minute(sample_rate)
    status_channel = raw.get_data()[STATUS_CHANNEL_ID]
    idx_starts = []
    i = 0
    while i <= (status_channel.shape[0] - n_samples_in_1_minute):
        if status_channel[i] == 4:
            idx_starts.append(i)
            i += n_samples_in_1_minute
        else:
            i += 1
    return idx_starts


def get_sample_rate(raw: mne.io.Raw) -> int:
    return int(raw.info['sfreq'])


def get_subjects(path: str) -> list[str]:
    subjects = []
    if os.path.exists(path):
        # s01-s22 are the only who have videos
        for i in range(1, 23):
            subject = f's{i:02d}.bdf'
            subject_path = os.path.join(path, subject)
            if os.path.exists(subject_path):
                subjects.append(subject)
    return subjects


def calculate_threshold(lowest_peak, highest_peak):
    magic_number = 2
    return (highest_peak - lowest_peak) / magic_number


def save_events(folder_name: str, subject: str, trial: int, events, sample_index, sample_rate):
    filename = _create_filename(folder_name, subject, trial, 'txt')
    os.makedirs(folder_name, exist_ok=True)
    events_in_seconds = _convert_in_seconds(events, sample_index, sample_rate)
    with open(filename, 'w') as f:
        f.write('Events detected: {}\n'.format(len(events_in_seconds)))
        f.write('Events: {}\n'.format(events))
        f.write('Events in seconds: {}\n'.format(events_in_seconds))


def save_plot(folder_name: str, l_freq, h_freq, raw, subject, trial):
    data = raw.get_data()
    sample_rate = get_sample_rate(raw)
    filtered_data = filter_data(data, sample_rate, l_freq, h_freq)
    _save_plot(folder_name, subject, trial, filtered_data, raw.info)


def crop(raw, sample_index):
    raw_copy = raw.copy().pick_channels(EEG)
    cropped_raw = _crop(raw_copy, sample_index)
    return cropped_raw


def get_extrema(raw, l_freq, h_freq):
    indices = get_indices_where_video_start(raw)
    return _get_extrema(raw, indices, l_freq, h_freq)


def _save(plot, file_name: str):
    fig = plot.gcf()
    fig.canvas.draw()
    fig.savefig(file_name)


def _crop(raw: mne.io.Raw, sample_index: int) -> mne.io.Raw:
    sample_rate = get_sample_rate(raw)
    n_samples_in_1_minute = _get_number_of_samples_in_1_minute(sample_rate)
    tmin = sample_index / sample_rate
    tmax = (sample_index + n_samples_in_1_minute) / sample_rate
    cropped_raw = raw.crop(tmin=tmin, tmax=tmax)
    return cropped_raw


def _get_extrema(raw, indices, l_freq, h_freq):
    lowest_peak = float('inf')
    highest_peak = float('-inf')
    for trial, sample_index in enumerate(indices, start=1):
        raw_copy = raw.copy().pick_channels(EEG)
        cropped_raw = _crop(raw_copy, sample_index)
        data = raw_copy.get_data()
        sample_rate = get_sample_rate(raw_copy)
        filtered_data = filter_data(data, sample_rate, l_freq, h_freq)

        if np.min(filtered_data) < lowest_peak:
            lowest_peak = np.min(filtered_data)
        if np.max(filtered_data) > highest_peak:
            highest_peak = np.max(filtered_data)

    return lowest_peak, highest_peak


def _save_plot(folder_name: str, subject: str, trial: int, data, info):
    raw_filtered = mne.io.RawArray(data, info)
    plot_raw(raw_filtered, duration=60, scalings=20e-5)
    # plt.show()
    filename = _create_filename(folder_name, subject, trial, 'png')
    os.makedirs(folder_name, exist_ok=True)
    _save(plt, filename)
    plt.close()


def _create_filename(folder_name: str, subject: str, trial: int, extension: str):
    subject_no_extension = subject.split(".")[0]
    two_digits_trial_count = str(trial).zfill(2)
    filename = f"{folder_name}/{subject_no_extension}_trial{two_digits_trial_count}.{extension}"
    return filename


def _get_number_of_samples_in_1_minute(sample_rate: int):
    seconds_in_1_minute = 60
    n_samples_in_1_minute = sample_rate * seconds_in_1_minute
    return n_samples_in_1_minute


def _convert_in_seconds(events, sample_index, sample_rate: int):
    events_indices = events[:, 0]
    return (events_indices - sample_index) / sample_rate
