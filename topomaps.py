import math as m
import warnings

import matplotlib.pyplot as plt
from mne.preprocessing import find_eog_events
from scipy.interpolate import griddata
from tqdm import tqdm

from eeg_constants import *
from utils import *


def getChannellInfoForSample(channelNames, channelValues, onlyValues=False):
    i = 0
    channelValuesforCurrentSample = []
    for ch in channelNames:
        chValue = channelValues[i]
        if onlyValues:
            channelValuesforCurrentSample.append(chValue)
        else:
            channelValuesforCurrentSample.append((ch, chValue))
        i += 1

    return channelValuesforCurrentSample


def convert3DTo2D(pos_3d):
    pos_2d = []
    for e in pos_3d:
        pos_2d.append(azim_proj(e))

    return (pos_2d)


def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x ** 2 + y ** 2
    r = m.sqrt(x2_y2 + z ** 2)  # r     tant^(-1)(y/x)
    elev = m.atan2(z, m.sqrt(x2_y2))  # Elevation
    az = m.atan2(y, x)  # Azimuth

    return r, elev, az


def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates    [x, y, z]
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])

    return pol2cart(az, m.pi / 2 - elev)


def pol2cart(theta, rho):
    """
    Transform polar coordinates to Cartesian
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    # print ('----------')
    # print (rho * m.cos(theta))
    # print (rho * m.sin(theta))
    return rho * m.cos(theta), rho * m.sin(theta)


def get3DCoordinates(MontageChannelLocation, EEGChannels):
    MontageChannelLocation = MontageChannelLocation[-EEGChannels:]
    location = []
    for i in range(0, 32):
        v = MontageChannelLocation[i].values()
        values = list(v)
        a = (values[1]) * 1000
        location.append(a)
    MontageLocation = np.array(location)
    # MontageLocation=trunc(MontageLocation,decs=3)
    MontageLocation = np.round(MontageLocation, 1)
    MontageLocation = MontageLocation.tolist()
    return MontageLocation


def createTopographicMapFromChannelValues(channelValues, interpolationMethod="cubic", verbose=False):
    channelNames = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7',
                    'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2',
                    'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']
    listOfChannelValues = getChannellInfoForSample(channelNames, channelValues, onlyValues=True)

    rawDatasetForMontageLocation = rawDatasetReReferenced.copy()
    MontageChannelLocation = rawDatasetForMontageLocation.info['dig']
    lengthOfTopographicMap = 32
    emptyTopographicMap = np.array(np.zeros([lengthOfTopographicMap, lengthOfTopographicMap]))

    if verbose:
        plt.imshow(emptyTopographicMap)
        plt.show()

    NumberOfEEGChannel = 32
    pos2D = np.array(convert3DTo2D(get3DCoordinates(MontageChannelLocation, NumberOfEEGChannel)))

    grid_x, grid_y = np.mgrid[
                     min(pos2D[:, 0]):max(pos2D[:, 0]):lengthOfTopographicMap * 1j,
                     min(pos2D[:, 1]):max(pos2D[:, 1]):lengthOfTopographicMap * 1j
                     ]

    # Generate edgeless images

    min_x, min_y = np.min(pos2D, axis=0)
    max_x, max_y = np.max(pos2D, axis=0)
    locations = np.append(pos2D, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)

    interpolatedTopographicMap = griddata(pos2D, channelValues, (grid_x, grid_y), method=interpolationMethod,
                                          fill_value=0)

    CordinateYellowRegion = np.argwhere(interpolatedTopographicMap == 0.)

    if verbose:
        i = 0
        for chVal in channelValues:
            for x in range(32):
                for y in range(32):
                    print("Trying to find value {} in pixel ({},{}-{})".format(chVal, x, y,
                                                                               interpolatedTopographicMap[x][y]))
                    if (chVal == interpolatedTopographicMap[x][y]):
                        print("Value found at pixel ({}{}) for channel: {}".format(x, y, channelNames[i]))
            i = i + 1

    if (verbose):
        plt.imshow(interpolatedTopographicMap)
        plt.show()

    return interpolatedTopographicMap, CordinateYellowRegion


if __name__ == '__main__':
    path = 'data_original/'
    l_freq = 0.1
    h_freq = 45

    # This dictionary has to be updated including missing subjects
    magic_numbers = {
        "s01.bdf": 105,
        "s02.bdf": 160,
        "s03.bdf": 100,
        "s04.bdf": 150,
        "s05.bdf": 150,
        "s06.bdf": 150,
        "s07.bdf": 110
    }

    subjects = get_subjects(path)
    # subjects = list(magic_numbers.keys())  # Only the subjects in the dictionary
    subjects = ["s02.bdf"]
    for subject in tqdm(subjects, desc="Processing subjects", unit="subject"):
        raw = read_bdf(path, subject)

        montage = mne.channels.make_standard_montage('biosemi32')
        raw.set_montage(montage, verbose=False)

        # Filtering and then resampling avoids aliasing
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
        warnings.filterwarnings("ignore", message="Resampling of the stim channels caused event information to become unreliable. Consider finding events on the original data and passing the event matrix as a parameter.")

        raw.resample(sfreq=128, verbose=False)

        # Computing the extrema for each subject (instead of for each trial) handles the scenario in which
        # a subject does not blink at all watching a trial:
        # their extrema would be too close to the baseline and the number of false positive blinks would be too large
        subject_lowest_peak, subject_highest_peak = get_extrema(raw, l_freq, h_freq)

        indices = get_indices_where_video_start(raw)
        for trial, index in tqdm(enumerate(indices, start=1), desc=f"Processing {subject} trials", total=len(indices), unit="trials"):
            # Crop the raw signal based on the 60-seconds-trial-length at given index
            cropped_raw_fp1_fp2 = crop(raw, index, FP1_FP2)
            """save_plot('plots', 1, 10, cropped_raw_fp1_fp2, subject, trial)"""

            data = cropped_raw_fp1_fp2.get_data()
            sample_rate = get_sample_rate(cropped_raw_fp1_fp2)
            filtered_data = filter_data(data, sample_rate, l_freq=1, h_freq=10, verbose=False)
            trial_lowest_peak = np.min(filtered_data)
            trial_highest_peak = np.max(filtered_data)

            magic_number = magic_numbers[subject]
            thresh = calculate_thresh(filtered_data, magic_number)

            # We shall focus on Fp1 and Fp2 only since they are the closest electrodes wrt the eyes
            events = find_eog_events(cropped_raw_fp1_fp2, ch_name=FP1_FP2, thresh=thresh, l_freq=1, h_freq=10,
                                     verbose=False)

            # Let's save some useful information on disk
            """save_events('events', subject, trial, events, index, sample_rate,
                        subject_lowest_peak, subject_highest_peak, magic_number,
                        trial_lowest_peak, trial_highest_peak,
                        thresh)"""

            ##########SALVATORE##########
            cropped_raw_eeg = crop(raw, index, EEG)

            idx_blinks = events[:, 0] - index
            # durata del blink prima e dopo il picco (in secondi). Può variare per soggetto/trial, meglio rimanere larghi
            blinkTime_pre = 0.09
            blinkTime_post = 0.12
            blink_pre = int(blinkTime_pre * sample_rate)
            blink_post = int(blinkTime_post * sample_rate)

            rawDataset = raw.copy()
            rawEEGall = rawDataset.pick_channels(EEG, verbose=False)
            min1 = sample_rate * 60
            rawEEGall_trialTest = rawEEGall.crop(tmin=index / sample_rate,
                                                 tmax=(index + min1) / sample_rate)
            dataTrial = rawEEGall_trialTest.get_data()

            # crea un intorno prima e dopo il picco del blink (per l'etichettatura)
            idx_blinks_about = []
            for i in idx_blinks:
                # intorno prima del picco
                for j in range(max(0, i - blink_pre), i):  # max evita le posizioni negative
                    idx_blinks_about.append(j)

                # intorno dopo
                for j in range(i, min(i + blink_post, dataTrial.shape[1])):  # min evita le posizioni inesistenti
                    idx_blinks_about.append(j)

            # ad ogni id del picco del blink, crea un intorno con l'id precedente e successivo
            idx_blinks_near = []
            for i in idx_blinks:
                idx_blinks_near.append(i - 1)  # sample precedente
                idx_blinks_near.append(i)  # sample del picco
                idx_blinks_near.append(i + 1)  # sample successivo

            sec = 0.5
            rawDatasetReReferenced = rawEEGall_trialTest.copy().set_eeg_reference(ref_channels='average', verbose=False)
            transposedDataset = np.transpose(rawDatasetReReferenced.get_data())

            trial_topomaps = []  # for given subject and given trial
            trial_labels = []
            for j in idx_blinks:
                start_index = max(j - int(sec * sample_rate), 0)
                end_index = min(j + int(sec * sample_rate), dataTrial.shape[1])
                for i in range(start_index, end_index):
                    channelValuesForCurrentSample = list(transposedDataset[i, :])
                    interpolatedTopographicMap, CordinateYellowRegion = createTopographicMapFromChannelValues(
                        channelValuesForCurrentSample, interpolationMethod="cubic", verbose=False)
                    trial_topomaps.append(interpolatedTopographicMap)

                    ### LABELING
                    # etichetta i blink in questo modo: 0, 1, 2
                    # 0 - non blink
                    # 1 - picco blink (e sample immediadamente prima e dopo)
                    # 2 - transizione da non blink a picco blink, e da picco blink a non blink
                    if i in idx_blinks_about:  # se i fa parte di un intorno di blink
                        if i in idx_blinks_near:  # se i è il picco del blink (o sample immediadamente prima o dopo)
                            label = BLINK  # da non usare per l'anomaly detection
                        else:  # transizione
                            label = TRANSITION  # da non usare per l'anomaly detection
                    else:  # non blink
                        label = NO_BLINK

                    trial_labels.append(label)

            if len(trial_topomaps) != 0:
                trial_topomaps = np.array(trial_topomaps)
                trial_labels = np.array(trial_labels)

                topomap_folder = 'topomaps'
                labels_folder = 'labels'

                os.makedirs(topomap_folder, exist_ok=True)
                os.makedirs(labels_folder, exist_ok=True)

                subject_without_extension = subject.rsplit(".", 1)[0]
                trial_with_leading_zero = str(trial).zfill(2)
                file_name = f"{subject_without_extension}_trial{trial_with_leading_zero}.npy"
                # print("Saving", file_name)
                np.save(os.path.join(topomap_folder, file_name), trial_topomaps)
                np.save(os.path.join(labels_folder, file_name), trial_labels)

                # Check con il notebook di Sabatina
                """if trial == 40:
                    plt.imshow(topomaps[0], cmap='gray')
                    plt.show()
                    print(topomaps[0].shape)"""

    correct_labels()

    # Check visuale, crea i png delle topomap specificate
    subject = "s02"
    trial = "01"
    file_name = subject + "_trial" + str(trial) + ".npy"
    topomaps = np.load(f"topomaps/{file_name}")
    labels = np.load(f"labels/{file_name}")
    output_folder = os.path.join("images", subject, trial)
    os.makedirs(output_folder, exist_ok=True)
    file_name_without_extension = os.path.splitext(file_name)[0]
    for i in tqdm(range(topomaps.shape[0]), desc=f"Saving {file_name_without_extension} topomaps", unit="topomap"):
        plt.imshow(topomaps[i], cmap="gray")
        plt.title(f"{file_name}[{i}] label = {labels[i]}")
        output_file = os.path.join(output_folder, f"{file_name_without_extension}_topomap{i + 1}.png")
        plt.savefig(output_file)
        plt.clf()
