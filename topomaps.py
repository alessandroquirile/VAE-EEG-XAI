import math as m

import matplotlib.pyplot as plt
from mne.preprocessing import find_eog_events
from scipy.interpolate import griddata

from eeg_constants import *
from utils import *


# monkey patching, per non creare naso e orecchie
def _make_head_outlines_new(sphere, pos, outlines, clip_origin):
    """Check or create outlines for topoplot."""
    assert isinstance(sphere, np.ndarray)
    x, y, _, radius = sphere
    del sphere

    if outlines in ("head", None):
        ll = np.linspace(0, 2 * np.pi, 101)
        head_x = np.cos(ll) * radius + x
        head_y = np.sin(ll) * radius + y
        dx = np.exp(np.arccos(np.deg2rad(12)) * 1j)
        dx, dy = dx.real, dx.imag
        # nose_x = np.array([-dx, 0, dx]) * radius + x
        # nose_y = np.array([dy, 1.15, dy]) * radius + y
        # ear_x = np.array(
        #     [0.497, 0.510, 0.518, 0.5299, 0.5419, 0.54, 0.547, 0.532, 0.510, 0.489]
        # ) * (radius * 2)
        # ear_y = (
        #     np.array(
        #         [
        #             0.0555,
        #             0.0775,
        #             0.0783,
        #             0.0746,
        #             0.0555,
        #             -0.0055,
        #             -0.0932,
        #             -0.1313,
        #             -0.1384,
        #             -0.1199,
        #         ]
        #     )
        #     * (radius * 2)
        #     + y
        # )

        if outlines is not None:
            # Define the outline of the head, ears and nose
            outlines_dict = dict(
                head=(head_x, head_y),
                # nose=(nose_x, nose_y),
                # ear_left=(-ear_x + x, ear_y),
                # ear_right=(ear_x + x, ear_y),
            )
        else:
            outlines_dict = dict()

        # Make the figure encompass slightly more than all points
        # We probably want to ensure it always contains our most
        # extremely positioned channels, so we do:
        mask_scale = max(1.0, np.linalg.norm(pos, axis=1).max() * 1.01 / radius)
        outlines_dict["mask_pos"] = (mask_scale * head_x, mask_scale * head_y)
        clip_radius = radius * mask_scale
        outlines_dict["clip_radius"] = (clip_radius,) * 2
        outlines_dict["clip_origin"] = clip_origin
        outlines = outlines_dict

    elif isinstance(outlines, dict):
        if "mask_pos" not in outlines:
            raise ValueError("You must specify the coordinates of the image " "mask.")
    else:
        raise ValueError("Invalid value for `outlines`.")

    return outlines


# monkey patching, per non far disegnare l'outline
def _draw_outlines_new(ax, outlines):
    """Draw the outlines for a topomap."""
    from matplotlib import rcParams

    outlines_ = {k: v for k, v in outlines.items() if k not in ["patch"]}
    for key, (x_coord, y_coord) in outlines_.items():
        if "mask" in key or key in ("clip_radius", "clip_origin"):
            continue
        ax.plot(
            x_coord,
            y_coord,
            color=rcParams["axes.edgecolor"],
            linewidth=0,  # 1, cambiato
            clip_on=False,
        )
    return outlines_


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
    subjects = ["s01.bdf"]
    for subject in subjects:
        raw = read_bdf(path, subject)

        montage = mne.channels.make_standard_montage('biosemi32')
        raw.set_montage(montage)

        # Filtering and then resampling avoids aliasing
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
        raw.resample(sfreq=128)  # così si avranno 128 MAPPE PER SECONDO

        # Computing the extrema for each subject (instead of for each trial) handles the scenario in which
        # a subject does not blink at all watching a trial:
        # their extrema would be too close to the baseline and the number of false positive blinks would be too large
        subject_lowest_peak, subject_highest_peak = get_extrema(raw, l_freq, h_freq)

        indices = get_indices_where_video_start(raw)
        for trial, index in enumerate(indices, start=1):
            # Crop the raw signal based on the 60-seconds-trial-length at given index
            cropped_raw_fp1_fp2 = crop(raw, index, FP1_FP2)
            """save_plot('plots', 1, 10, cropped_raw_fp1_fp2, subject, trial)"""

            data = cropped_raw_fp1_fp2.get_data()
            sample_rate = get_sample_rate(cropped_raw_fp1_fp2)
            filtered_data = filter_data(data, sample_rate, l_freq=1, h_freq=10)
            trial_lowest_peak = np.min(filtered_data)
            trial_highest_peak = np.max(filtered_data)

            magic_number = magic_numbers[subject]
            thresh = calculate_thresh(filtered_data, magic_number)

            # We shall focus on Fp1 and Fp2 only since they are the closest electrodes wrt the eyes
            events = find_eog_events(cropped_raw_fp1_fp2, ch_name=FP1_FP2, thresh=thresh, l_freq=1, h_freq=10)

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

            # dataTrial = cropped_raw_eeg_all.get_data()
            # dataTrial = raw.copy().pick_channels(EEG).get_data()
            rawDataset = raw.copy()
            rawEEGall = rawDataset.pick_channels(EEG)
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

            # plot topomaps
            """# Monkey patches
            mne.viz.topomap._make_head_outlines = _make_head_outlines_new
            mne.viz.topomap._draw_outlines = _draw_outlines_new"""

            sec = 0.5
            rawDatasetReReferenced = rawEEGall_trialTest.copy().set_eeg_reference(ref_channels='average')
            transposedDataset = np.transpose(rawDatasetReReferenced._data)

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
                            label = 1  # da non usare per l'anomaly detection
                        else:  # transizione
                            label = 2  # da non usare per l'anomaly detection
                    else:  # non blink
                        label = 0

                    trial_labels.append(label)

            if len(trial_topomaps) != 0:
                trial_topomaps = np.array(trial_topomaps)
                trial_labels = np.array(trial_labels)

                topomap_folder = 'topomaps'
                labels_folder = 'labels'

                os.makedirs(topomap_folder, exist_ok=True)
                os.makedirs(labels_folder, exist_ok=True)

                subject_without_extension = subject.rsplit(".", 1)[0]
                file_name = f"{subject_without_extension}_trial{str(trial).zfill(2)}.npy"
                print("Saving", file_name)
                np.save(os.path.join(topomap_folder, file_name), trial_topomaps)
                np.save(os.path.join(labels_folder, file_name), trial_labels)

                # Check con il notebook di Sabatina
                """if trial == 40:
                    plt.imshow(topomaps[0], cmap='gray')
                    plt.show()
                    print(topomaps[0].shape)"""
