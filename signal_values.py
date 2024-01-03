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


def check_same_values(my_list, values):
    for column in range(values.shape[1]):
        if not np.all(my_list[column] == values[:, column]):
            raise Exception("list and values do not coincide")


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

    print("\n>>> QUESTO SCRIPT MOSTRA E SALVA I DATI DI CHANNEL, INTERPOLATED, MASKED RECONSTRUCTED VALUES"
          " E RECONSTRUCTED VALUES NELLA CARTELLA signal_values/ <<<")

    for subject in tqdm(subjects, desc="Processing subjects", unit="subject"):

        # Creazione della cartella values
        subject_without_extension = subject.rsplit(".", 1)[0]
        signal_values_folder = f"signal_values/{subject_without_extension}"
        os.makedirs(signal_values_folder, exist_ok=True)

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
            # INIZIO CHANNEL VALUES
            channel_values_list = []
            for j in idx_blinks:
                start_index = max(j - int(sec * sample_rate), 0)
                end_index = min(j + int(sec * sample_rate), dataTrial.shape[1])
                for i in range(start_index, end_index):
                    channelValuesForCurrentSample = list(transposedDataset[i, :])
                    interpolatedTopographicMap, CordinateYellowRegion, pos2D = createTopographicMapFromChannelValues(
                        channelValuesForCurrentSample, rawDatasetReReferenced, interpolationMethod="cubic",
                        verbose=False)
                    channel_values_list.append(channelValuesForCurrentSample)
                    trial_topomaps.append(interpolatedTopographicMap)
                    label = labeling(idx_blinks_about, idx_blinks_near)
                    trial_labels.append(label)

            if len(trial_topomaps) == 0:
                continue

            channel_values = np.array(channel_values_list).transpose()
            check_same_values(channel_values_list, channel_values)

            # FINE CHANNEL VALUES

            trial_topomaps = np.array(trial_topomaps)
            trial_labels = np.array(trial_labels)

            rawDatasetForMontageLocation = rawDatasetReReferenced.copy()
            montage_ch_location = rawDatasetForMontageLocation.info['dig']
            channelNames = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7',
                            'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2',
                            'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']

            # INIZIO INTERPOLATED
            interpolated_values_list = []  # Serve per costruire interpolated_values in maniera più semplice
            interpolated_values = None

            for i in range(trial_topomaps.shape[0]):  # Ad esempio 0,...,127 se c'è un solo blink
                trial_topomaps_i = trial_topomaps[i]
                coordinates_yellow = np.argwhere(trial_topomaps_i == 0.)
                channelInfoFromInterpolatedMap = retrieveChannelInfoFromInterpolatedMap(trial_topomaps_i,
                                                                                        coordinates_yellow, 40,
                                                                                        montage_ch_location, 32,
                                                                                        channelNames,
                                                                                        onlyValues=True)

                interpolated_values_list.append(channelInfoFromInterpolatedMap)

            interpolated_values = np.array(interpolated_values_list).transpose()
            check_same_values(interpolated_values_list, interpolated_values)

            # FINE INTERPOLATED

            # Topomaps modified (masked reconstructed)
            # INIZIO MASKED RECONSTRUCTED
            masked_reconstructed_values = None
            trial_with_leading_zero = str(trial).zfill(2)
            folder = f"topomaps_reduced_{subject_without_extension}_mod"
            topomaps_files_mod = os.listdir(folder)
            for file in topomaps_files_mod:

                # Affinché il file corrente sia coerente rispetto al soggetto e al trial corrente del ciclo esterno
                if trial_with_leading_zero not in file:
                    continue

                masked_reconstructed_values_list = []
                trial_topomaps_mod = np.load(f"{folder}/{file}")
                for i in range(trial_topomaps_mod.shape[0]):
                    trial_topomaps_i_mod = trial_topomaps_mod[i]
                    coordinates_yellow = np.argwhere(trial_topomaps_i_mod == 0.)
                    channelInfoFromInterpolatedMap = retrieveChannelInfoFromInterpolatedMap(trial_topomaps_i_mod,
                                                                                            coordinates_yellow, 40,
                                                                                            montage_ch_location, 32,
                                                                                            channelNames,
                                                                                            onlyValues=True)
                    masked_reconstructed_values_list.append(channelInfoFromInterpolatedMap)

                masked_reconstructed_values = np.array(masked_reconstructed_values_list).transpose()
                check_same_values(masked_reconstructed_values_list, masked_reconstructed_values)

                # FINE MASKED RECONSTRUCTED

            # INIZIO RECONSTRUCTED (no mask)
            reconstructed_values = None
            folder = f"topomaps_reduced_{subject_without_extension}_rec"
            topomaps_files_mod = os.listdir(folder)
            for file in topomaps_files_mod:

                # Affinché il file corrente sia coerente rispetto al soggetto e al trial corrente del ciclo esterno
                if trial_with_leading_zero not in file:
                    continue

                reconstructed_values_list = []
                trial_topomaps_rec = np.load(f"{folder}/{file}")
                for i in range(trial_topomaps_rec.shape[0]):
                    trial_topomaps_i_rec = trial_topomaps_rec[i]
                    coordinates_yellow = np.argwhere(trial_topomaps_i_rec == 0.)
                    channelInfoFromInterpolatedMap = retrieveChannelInfoFromInterpolatedMap(trial_topomaps_i_rec,
                                                                                            coordinates_yellow, 40,
                                                                                            montage_ch_location, 32,
                                                                                            channelNames,
                                                                                            onlyValues=True)
                    reconstructed_values_list.append(channelInfoFromInterpolatedMap)

                reconstructed_values = np.array(reconstructed_values_list).transpose()
                check_same_values(reconstructed_values_list, reconstructed_values)

                # FINE RECONSTRUCTED (no mask)

                # TODO:
                #  Anche con VAE
                #  Inviare immagini a sabatina
                #  Provare a lasciare solo reconstructed
                # Selezione del canale (esempio: primo canale, indice 0)
                indice_canale = 0
                canale_selezionato = channelNames[indice_canale]

                # BLINK SEPARATI
                block_size = sample_rate
                num_samples = channel_values.shape[1]
                for i in range(0, channel_values.shape[1], sample_rate):
                    end_index = min(i + block_size, num_samples)
                    plt.figure(figsize=(10, 6))
                    tempo_totale = 1
                    asse_x = np.linspace(0, tempo_totale,  sample_rate)
                    if channel_values[indice_canale, i:end_index].shape[0] != sample_rate:
                        continue
                    plt.plot(asse_x, channel_values[indice_canale, i:end_index], label='Channel')
                    plt.plot(asse_x, interpolated_values[indice_canale, i:end_index], label='Interpolated')
                    # plt.plot(asse_x, masked_reconstructed_values[indice_canale], label='Masked reconstructed',
                    #          linestyle='--')
                    # Reconstructed sono gli output del modello dandogli in input le topomap in sequenza
                    # "modificate"
                    plt.plot(asse_x, reconstructed_values[indice_canale, i:end_index], label='Reconstructed', linestyle='dotted')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Intensity (V)')
                    plt.title(f'{file}. Channel {canale_selezionato}')
                    plt.legend()
                    plt.grid(True)
                    plt.axvline(x=0.5, color='red', linestyle='--')
                    # plt.show()

                    # Salvo le immagini dei tre segnali
                    images_folder = os.path.join(signal_values_folder, f"signal_images_{subject_without_extension}")
                    os.makedirs(images_folder, exist_ok=True)
                    image_file_name = f"{file}_plot.png"
                    image_path = os.path.join(images_folder, image_file_name)
                    counter = 1
                    while os.path.exists(image_path):
                        # Se il file esiste già, incrementa il contatore e genera un nuovo nome file
                        image_file_name = f"{file}_plot_{counter}.png"
                        image_path = os.path.join(images_folder, image_file_name)
                        counter += 1
                    plt.savefig(image_path)
                    plt.close()

                # TUTTI I BLINK INSIEME
                # Calcolo del tempo totale in base al numero di campioni e la frequenza di campionamento
                # Ad esempio se c'è solo un blink, ovvero channel_values.shape[1] = 128 allora
                # Tempo totale sarà 128/128 = 1s
                """tempo_totale = channel_values.shape[1] / sample_rate
                asse_x = np.linspace(0, tempo_totale, channel_values.shape[1])
                plt.figure(figsize=(10, 6))
                plt.plot(asse_x, channel_values[indice_canale], label='Channel')
                plt.plot(asse_x, interpolated_values[indice_canale], label='Interpolated')
                # plt.plot(asse_x, masked_reconstructed_values[indice_canale], label='Masked reconstructed',
                #          linestyle='--')
                # Reconstructed sono gli output del modello dandogli in input le topomap in sequenza
                # "modificate"
                plt.plot(asse_x, reconstructed_values[indice_canale], label='Reconstructed',
                         linestyle='dotted')
                plt.xlabel('Time (s)')
                plt.ylabel('Intensity (V)')
                plt.title(f'{file}. Channel {canale_selezionato}')
                plt.legend()
                plt.grid(True)
                # Linee verticali sui blink
                num_positions = int(tempo_totale)
                x_positions = [i + 0.5 for i in range(num_positions)]
                # Correzione manuale
                if file == "s01_trial35.npy":
                    x_positions = [0.25, 1.25, 2.25, 3.25]
                for x in x_positions:
                    plt.axvline(x=x, color='red', linestyle='--')  # Linee verticali per ciascuna posizione x
                # plt.show()

                # Salvo le immagini dei tre segnali
                images_folder = os.path.join(signal_values_folder, f"signal_images_{subject_without_extension}")
                os.makedirs(images_folder, exist_ok=True)
                image_file_name = f"{file}_plot.png"
                image_path = os.path.join(images_folder, image_file_name)
                plt.savefig(image_path)
                plt.close()"""

                # Salvo channel_values
                file_name = f"{subject_without_extension}_trial{trial_with_leading_zero}.npy"

                channel_values_folder = f"channel_values_{subject_without_extension}"
                os.makedirs(os.path.join(signal_values_folder, channel_values_folder), exist_ok=True)
                np.save(os.path.join(signal_values_folder, channel_values_folder, file_name), channel_values)

                # Salvo interpolated_values
                interpolated_values_folder = f"interpolated_values_{subject_without_extension}"
                os.makedirs(os.path.join(signal_values_folder, interpolated_values_folder), exist_ok=True)
                np.save(os.path.join(signal_values_folder, interpolated_values_folder, file_name), interpolated_values)

                # Salvo masked_reconstructed_values
                masked_reconstructed_folder = f"masked_reconstructed_values_{subject_without_extension}"
                os.makedirs(os.path.join(signal_values_folder, masked_reconstructed_folder), exist_ok=True)
                np.save(os.path.join(signal_values_folder, masked_reconstructed_folder, file_name),
                        masked_reconstructed_values)

                # Salvo reconstructed_values (senza mascheramento)
                reconstructed_folder = f"reconstructed_values_{subject_without_extension}"
                os.makedirs(os.path.join(signal_values_folder, reconstructed_folder), exist_ok=True)
                np.save(os.path.join(signal_values_folder, reconstructed_folder, file_name),
                        reconstructed_values)

            # exit(0)
