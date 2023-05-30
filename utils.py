import os

import mne.io
import numpy as np
from mne.filter import filter_data
from mne.io import read_raw_bdf

from deap_constants import *
from eeg_constants import EEG, EOG, MISC, STIM_CHANNEL


def read_bdf(path: str, subject: str) -> mne.io.Raw:
    raw = read_raw_bdf(path + subject, eog=EOG, misc=MISC, stim_channel=STIM_CHANNEL, preload=True, verbose=False)
    return raw


def get_indices_where_video_start(raw: mne.io.Raw) -> list[int]:
    sample_rate = get_sample_rate(raw)
    n_samples_in_1_minute = _get_number_of_samples_in_1_minute(sample_rate)
    # https://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html
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
        # S01-S22 are the only ones who have videos of their trials
        # range(1, 23) means [1;23) mathematically
        for i in range(1, 23):
            subject = f's{i:02d}.bdf'
            subject_path = os.path.join(path, subject)
            if os.path.exists(subject_path):
                subjects.append(subject)
    return subjects


def calculate_thresh(subject, subject_highest_peak, subject_lowest_peak, trial_highest_peak, trial_lowest_peak,
                     magic_number, filtered_data):
    if _blinks_rarely(subject):
        return (subject_highest_peak - subject_lowest_peak) / magic_number
    else:
        return (trial_highest_peak - trial_lowest_peak) - (magic_number * np.std(filtered_data))


def crop(raw, sample_index):
    raw_copy = raw.copy().pick_channels(EEG)
    cropped_raw = _crop(raw_copy, sample_index)
    return cropped_raw


def get_extrema(raw, l_freq, h_freq):
    indices = get_indices_where_video_start(raw)
    return _get_extrema(raw, indices, l_freq, h_freq)


def _blinks_rarely(subject):
    subjects_who_blink_rarely = ['s01.bdf', 's03.bdf', 's16.bdf', 's18.bdf', 's21.bdf']
    return subject in subjects_who_blink_rarely


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


def _get_number_of_samples_in_1_minute(sample_rate: int):
    seconds_in_1_minute = 60
    n_samples_in_1_minute = sample_rate * seconds_in_1_minute
    return n_samples_in_1_minute
