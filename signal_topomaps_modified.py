import warnings

from mne.preprocessing import find_eog_events
from tqdm import tqdm

from eeg_constants import *
from indian_functions import *
from savers import save_events
from utils import *


def extract_idx_blinks_about(idx_blinks, blink_pre, blink_post, dataTrial):
    idx_blinks_about = []
    for i in idx_blinks:
        # intorno prima del picco
        for j in range(max(0, i - blink_pre), i):  # max evita le posizioni negative
            idx_blinks_about.append(j)

        # intorno dopo
        for j in range(i, min(i + blink_post, dataTrial.shape[1])):  # min evita le posizioni inesistenti
            idx_blinks_about.append(j)

    return idx_blinks_about


def extract_idx_blinks_near(idx_blinks):
    idx_blinks_near = []
    for i in idx_blinks:
        idx_blinks_near.append(i - 1)  # sample precedente
        idx_blinks_near.append(i)  # sample del picco
        idx_blinks_near.append(i + 1)  # sample successivo
    return idx_blinks_near


def labeling(idx_blinks_about, idx_blinks_near):
    if i in idx_blinks_about:
        if i in idx_blinks_near:
            label = BLINK  # da non usare per l'anomaly detection
        else:
            label = TRANSITION  # da non usare per l'anomaly detection
    else:
        label = NO_BLINK
    return label


if __name__ == '__main__':
    path = 'data_original/'
    l_freq = 0.1
    h_freq = 45

    # For participants 3, 5, 11 and 14, one or several of the last trials are missing due to technical issues
    # s01-s22 are the only participants who do have videos
    magic_numbers = {
        "s01.bdf": 105,
        # "s02.bdf": 160,
        # "s03.bdf": 100,
        # "s04.bdf": 150,
        # "s05.bdf": 150,
        # "s06.bdf": 150,
        # "s07.bdf": 110,
        # "s08.bdf": 150,
        # "s09.bdf": 150,
        # "s10.bdf": 110,
        # "s11.bdf": 110,
        # "s12.bdf": 150,
        # "s13.bdf": 110,
        # "s14.bdf": 110,
        # "s15.bdf": 150,
        # "s16.bdf": 105,
        # "s17.bdf": 150,
        # "s18.bdf": 105,
        # "s19.bdf": 150,
        # "s20.bdf": 150,
        # "s21.bdf": 150,
        # "s22.bdf": 160
    }

    subjects = list(magic_numbers.keys())
    montage = mne.channels.make_standard_montage('biosemi32')

    print("\n>>> QUESTO SCRIPT MOSTRA I GRAFICI DEI SEGNALI CHANNEL, INTERPOLATED E RECONSTRUCTED <<<")

    for subject in tqdm(subjects, desc="Processing subjects", unit="subject"):

        pos2D = []

        raw = read_bdf(path, subject)
        raw.set_montage(montage, verbose=False)

        # Filtering and then resampling avoids aliasing
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
        warnings.filterwarnings("ignore",
                                message="Resampling of the stim channels caused event information to become unreliable. Consider finding events on the original data and passing the event matrix as a parameter.")

        raw.resample(128, verbose=False)

        # Computing the extrema for each subject (instead of for each trial) handles the scenario in which
        # a subject does not blink at all watching a trial:
        # their extrema would be too close to the baseline and the number of false positive blinks would be too large
        subject_lowest_peak, subject_highest_peak = get_extrema(raw, l_freq, h_freq)

        indices = get_indices_where_video_start(raw)
        for trial, index in tqdm(enumerate(indices, start=1), desc=f"Processing {subject} trials", total=len(indices),
                                 unit="trials"):
            # Crop the raw signal based on the 60-seconds-trial-length at given index
            cropped_raw_fp1_fp2 = crop(raw, index, FP1_FP2)

            data = cropped_raw_fp1_fp2.get_data()
            sample_rate = get_sample_rate(cropped_raw_fp1_fp2)
            filtered_data = filter_data(data, sample_rate, l_freq=1, h_freq=10, verbose=False)

            magic_number = magic_numbers[subject]
            thresh = calculate_thresh(filtered_data, magic_number)

            # We shall focus on Fp1 and Fp2 only since they are the closest electrodes wrt the eyes
            events = find_eog_events(cropped_raw_fp1_fp2, ch_name=FP1_FP2, thresh=thresh, l_freq=1, h_freq=10,
                                     verbose=False)

            trial_lowest_peak = np.min(filtered_data)
            trial_highest_peak = np.max(filtered_data)

            cropped_raw_eeg = crop(raw, index, EEG)

            idx_blinks = events[:, 0] - index
            blinkTime_pre = 0.09
            blinkTime_post = 0.13
            blink_pre = int(blinkTime_pre * sample_rate)
            blink_post = int(blinkTime_post * sample_rate)

            rawDataset = raw.copy()
            rawEEGall = rawDataset.pick_channels(EEG, verbose=False)
            min1 = sample_rate * 60
            rawEEGall_trialTest = rawEEGall.crop(tmin=index / sample_rate,
                                                 tmax=(index + min1) / sample_rate, include_tmax=False)
            dataTrial = rawEEGall_trialTest.get_data()

            # crea un intorno prima e dopo il picco del blink (per l'etichettatura)
            idx_blinks_about = extract_idx_blinks_about(idx_blinks, blink_pre, blink_post, dataTrial)

            # ad ogni id del picco del blink, crea un intorno con l'id precedente e successivo
            idx_blinks_near = extract_idx_blinks_near(idx_blinks)

            sec = 0.5
            rawDatasetReReferenced = rawEEGall_trialTest.copy().set_eeg_reference(ref_channels='average', verbose=False)
            transposedDataset = np.transpose(rawDatasetReReferenced.get_data())

            trial_topomaps = []  # for given subject and given trial
            trial_labels = []

            # Solo nell'intorno
            channel_values = []
            for j in idx_blinks:
                start_index = max(j - int(sec * sample_rate), 0)
                end_index = min(j + int(sec * sample_rate), dataTrial.shape[1])
                for i in range(start_index, end_index):
                    channelValuesForCurrentSample = list(transposedDataset[i, :])
                    interpolatedTopographicMap, CordinateYellowRegion, pos2D = createTopographicMapFromChannelValues(
                        channelValuesForCurrentSample, rawDatasetReReferenced, interpolationMethod="cubic",
                        verbose=False)
                    channel_values.append(channelValuesForCurrentSample)
                    trial_topomaps.append(interpolatedTopographicMap)
                    label = labeling(idx_blinks_about, idx_blinks_near)
                    trial_labels.append(label)

            if len(trial_topomaps) == 0:
                continue

            trial_topomaps = np.array(trial_topomaps)
            trial_labels = np.array(trial_labels)

            rawDatasetForMontageLocation = rawDatasetReReferenced.copy()
            montage_ch_location = rawDatasetForMontageLocation.info['dig']
            channelNames = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7',
                            'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2',
                            'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']

            my_list = []  # Serve per costruire interpolated_values in maniera più semplice
            interpolated_values = np.array(np.zeros(shape=(32, 128)))

            for i in range(trial_topomaps.shape[0]):
                trial_topomaps_i = trial_topomaps[i]
                coordinates_yellow = np.argwhere(trial_topomaps_i == 0.)
                channelInfoFromInterpolatedMap = retrieveChannelInfoFromInterpolatedMap(trial_topomaps_i,
                                                                                        coordinates_yellow, 40,
                                                                                        montage_ch_location, 32,
                                                                                        channelNames,
                                                                                        onlyValues=True)

                my_list.append(channelInfoFromInterpolatedMap)

            interpolated_values = np.array(my_list).transpose()
            channel_values = np.array(channel_values).transpose()

            # Topomaps modified
            subject_without_extension = subject.rsplit(".", 1)[0]
            trial_with_leading_zero = str(trial).zfill(2)
            topomaps_files_mod = os.listdir(f"topomaps_reduced_{subject_without_extension}_mod")
            for file in topomaps_files_mod:
                my_list2 = []
                trial_topomaps_mod = np.load(f"topomaps_reduced_{subject_without_extension}_mod/{subject_without_extension}_trial{trial_with_leading_zero}.npy")
                for i in range(trial_topomaps_mod.shape[0]):
                    trial_topomaps_i_mod = trial_topomaps_mod[i]
                    coordinates_yellow = np.argwhere(trial_topomaps_i_mod == 0.)
                    channelInfoFromInterpolatedMap = retrieveChannelInfoFromInterpolatedMap(trial_topomaps_i_mod,
                                                                                            coordinates_yellow, 40,
                                                                                            montage_ch_location, 32,
                                                                                            channelNames,
                                                                                            onlyValues=True)
                    my_list2.append(channelInfoFromInterpolatedMap)

                reconstructed_values = np.array(my_list2).transpose()

                for column in range(interpolated_values.shape[1]):
                    if not np.all(my_list[column] == interpolated_values[:, column]):
                        raise Exception("my_list and interpolated_values do not coincide")

                for column in range(reconstructed_values.shape[1]):
                    if not np.all(my_list2[column] == reconstructed_values[:, column]):
                        raise Exception("my_list2 and reconstructed_values do not coincide")

                # Calcolo del tempo totale in base al numero di campioni e la frequenza di campionamento
                tempo_totale = channel_values.shape[1] / sample_rate

                # Creazione dell'asse x in secondi
                asse_x = np.linspace(0, tempo_totale, channel_values.shape[1])

                # SINGOLO CANALE
                # Selezione del canale (esempio: primo canale, indice 0)
                canale_selezionato = 0
                # Tracciamento del segnale nel tempo per il canale selezionato dalle due matrici
                plt.figure(figsize=(10, 6))
                plt.plot(asse_x, channel_values[canale_selezionato], label='Channel')
                plt.plot(asse_x, interpolated_values[canale_selezionato], label='Interpolated')
                plt.plot(asse_x, reconstructed_values[canale_selezionato], label='Reconstructed', linestyle='--')
                plt.xlabel('Tempo (s)')
                plt.ylabel('Intensità del segnale')
                plt.title(f'{file}. Segnale nel tempo per il canale {canale_selezionato}')
                plt.legend()
                plt.grid(True)
                plt.show()

            # exit(0)
