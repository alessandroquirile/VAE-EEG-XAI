import os

import mne.io
import numpy as np
from mne.filter import filter_data
from mne.io import read_raw_bdf

from eeg_constants import FP1_FP2, EOG, MISC, STIM_CHANNEL
from label_enum import *


def read_bdf(path: str, subject: str) -> mne.io.Raw:
    raw = read_raw_bdf(path + subject, eog=EOG, misc=MISC, stim_channel=STIM_CHANNEL, preload=True, verbose=False)
    return raw


def get_indices_where_video_start(raw: mne.io.Raw) -> list[int]:
    STATUS_CHANNEL_ID = 47
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


def calculate_thresh(filtered_data, magic_number):
    # Detection of EEG-Based Eye-Blinks Using A Thresholding Algorithm, Dang-Khoa Tran, Thanh-Hai Nguyen, Thanh-Nghia
    # Nguyen
    data_max = np.max(np.sqrt(np.abs(filtered_data)))
    return (data_max - np.std(filtered_data)) / magic_number


def crop(raw, sample_index, channels):
    raw_copy = raw.copy().pick_channels(channels, verbose=False)
    cropped_raw = _crop(raw_copy, sample_index)
    return cropped_raw


def get_extrema(raw, l_freq, h_freq):
    indices = get_indices_where_video_start(raw)
    return _get_extrema(raw, indices, l_freq, h_freq)


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
    """lowest_peak_trial = 0
    highest_peak_trial = 0"""
    for trial, sample_index in enumerate(indices, start=1):
        raw_copy = raw.copy().pick_channels(FP1_FP2, verbose=False)
        cropped_raw = _crop(raw_copy, sample_index)
        data = raw_copy.get_data()
        sample_rate = get_sample_rate(raw_copy)
        filtered_data = filter_data(data, sample_rate, l_freq, h_freq, verbose=False)

        if np.min(filtered_data) < lowest_peak:
            lowest_peak = np.min(filtered_data)
            # lowest_peak_trial = trial
        if np.max(filtered_data) > highest_peak:
            highest_peak = np.max(filtered_data)
            # highest_peak_trial = trial

    """print("min found at", lowest_peak_trial)
    print("max found at", highest_peak_trial)"""
    return lowest_peak, highest_peak


def _get_number_of_samples_in_1_minute(sample_rate: int):
    seconds_in_1_minute = 60
    n_samples_in_1_minute = sample_rate * seconds_in_1_minute
    return n_samples_in_1_minute


def correct_labels():
    print("\nCorrecting labels...")

    # Correcting s01
    subject = "s01"
    mark_as_no_blinks(f"labels/{subject}_trial06.npy")
    mark_as_transition(f"labels/{subject}_trial30.npy", 79, 90)

    # Correcting s02
    subject = "s02"
    mark_as_transition(f"labels/{subject}_trial07.npy", 591, 619)
    mark_as_transition(f"labels/{subject}_trial07.npy", 719, 742)
    mark_as_transition(f"labels/{subject}_trial07.npy", 1743, 1745)
    mark_as_transition(f"labels/{subject}_trial07.npy", 1871, 1878)
    mark_as_transition(f"labels/{subject}_trial40.npy", 335, 340)
    mark_as_transition(f"labels/{subject}_trial40.npy", 463, 480)

    # Correcting s07
    subject = "s07"
    mark_as_transition(f"labels/{subject}_trial01.npy", 978, 991)
    mark_as_transition(f"labels/{subject}_trial39.npy", 209, 229)

    # Correcting s15
    subject = "s15"
    mark_as_no_blinks(f"labels/{subject}_trial40.npy", 693, 699)
    mark_as_no_blinks(f"labels/{subject}_trial40.npy", 717, 720)
    mark_as_transition(f"labels/{subject}_trial40.npy", 768, 787)
    mark_as_blink(f"labels/{subject}_trial40.npy", 788, 790)
    mark_as_transition(f"labels/{subject}_trial40.npy", 791, 792)

    # Correcting s16
    subject = "s16"
    mark_as_no_blinks(f"labels/{subject}_trial01.npy", 70, 80)
    mark_as_no_blinks(f"labels/{subject}_trial01.npy", 181, 186)
    mark_as_no_blinks(f"labels/{subject}_trial01.npy", 309, 336)

    # Correcting s17
    subject = "s17"
    mark_as_no_blinks(f"labels/{subject}_trial01.npy", 38, 65)
    mark_as_no_blinks(f"labels/{subject}_trial01.npy", 166, 193)
    mark_as_transition(f"labels/{subject}_trial01.npy", 259, 294)
    mark_as_transition(f"labels/{subject}_trial01.npy", 321, 368)

    print("Labels corrected")


def mark_as_blink(file_path, start_index, end_index):
    """
    Correct labels marking blinks
    :param file_path:  (str) Path to the file
    :param start_index: Start index to be marked as transition
    :param end_index: End index to be marked as transition (esclusive)
    :return: None
    """
    labels = np.load(file_path)
    labels[start_index:end_index] = BLINK  # [start_index; end_index)
    np.save(file_path, labels)


def mark_as_no_blinks(file_path, start_index=None, end_index=None):
    """
    Correct labels marking no blinks
    :param file_path:  (str) Path to the file
    :param start_index: Start index to be marked as transition
    :param end_index: End index to be marked as transition (esclusive)
    :return: None
    """
    labels = np.load(file_path)
    if start_index is None and end_index is None:
        labels.fill(NO_BLINK)
    else:
        labels[start_index:end_index] = NO_BLINK
    np.save(file_path, labels)


def mark_as_transition(file_path, start_index, end_index):
    """
    Correct labels marking transitions
    :param file_path:  (str) Path to the file
    :param start_index: Start index to be marked as transition
    :param end_index: End index to be marked as transition (esclusive)
    :return: None
    """
    labels = np.load(file_path)
    labels[start_index:end_index] = TRANSITION  # [start_index; end_index)
    np.save(file_path, labels)
