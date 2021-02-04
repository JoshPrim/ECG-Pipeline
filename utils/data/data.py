import logging
from datetime import datetime
import os

import json
import numpy as np
import pandas as pd
from os import path

from sklearn.utils import shuffle, resample
from xml.dom import minidom

#TODO: herausfinden welche methoden wirklich gebraucht werden und welche fpür die Demo nicht nötig sind

# from utils.api.wfdb_api import load_norm_and_mi_ecgs, load_patientrecords_mi_norm
from utils.data.validation import validate_and_clean_float, validate_and_clean_char
from utils.file.file import save_dict_as_json, save_string_to_file, load_string_from_file, load_dict_from_json, \
    pickle_data, unpickle_data, make_dirs_if_not_present
# from utils.api.redcap_api import load_report, load_file
from utils.misc.datastructure import perform_shape_switch


def create_snapshot_from_redcap_api(snapshot_directory='../data/kerckhoff/snapshots'):
    snapshot_id = datetime.now().strftime("%Y-%m-%d")
    snapshot_path = '{}/{}'.format(snapshot_directory, snapshot_id)
    snapshot_path_clinicalparameters = '{}/{}'.format(snapshot_path, 'clinicalparameters')
    snapshot_path_ecg = '{}/{}'.format(snapshot_path, 'ecg')

    if path.exists(snapshot_path):
        raise Exception('A snapshot with ID {} already exists! Please manually clean the directory if you want to renew the snapshot. Path: {}'.format(snapshot_id, snapshot_path))
    else:
        os.makedirs(snapshot_path)
        os.mkdir(snapshot_path_clinicalparameters)
        os.mkdir(snapshot_path_ecg)

    report = load_report(report_id='160')

    for record, i in zip(report, range(len(report))):
        print('Processing record {} of {}'.format(i, len(report)))

        record_id = record['record_id']
        filepath = '{}/{}.json'.format(snapshot_path_clinicalparameters, record_id)
        save_dict_as_json(record, filepath)

        ecg_xml = load_file(record_id=record_id, field='varid_ekg_hl7', event='baseline_arm_1')
        filepath = '{}/{}.xml'.format(snapshot_path_ecg, record_id)
        save_string_to_file(ecg_xml, filepath)

    print('Created snapshot with name "{}" at "{}"'.format(snapshot_id, snapshot_directory))


def parse_ecg_xml(xmlcode, leads_to_use=None):
    xmlparsed = minidom.parseString(xmlcode)
    itemlist = xmlparsed.getElementsByTagName('sequence')

    leads = {}
    uom = ''
    length = 0

    for i in range(0, 12):
        cur_sequence = itemlist[i + 1]
        lead = list(np.fromstring(cur_sequence.getElementsByTagName('digits')[0].childNodes[0].nodeValue,
                                  dtype=int,
                                  sep=' '))
        length = len(lead)
        uom = cur_sequence.getElementsByTagName('scale')[0].getAttribute('unit')
        lead_id = cur_sequence.getElementsByTagName('code')[0].getAttribute('code').replace('MDC_ECG_LEAD_', '')

        if leads_to_use is None:
            leads[lead_id] = lead

        elif lead_id in leads_to_use:
            leads[lead_id] = lead

    # TODO: Add active filters, etc.
    metadata = {'sampling_rate_sec': 500,
                'unitofmeasurement': uom,
                'length_sec': 10,
                'length_timesteps': length}

    return leads, metadata


def load_ecg_xml(path, leads_to_use=None):
    xmlcode = load_string_from_file(path)
    leads, metadata = parse_ecg_xml(xmlcode, leads_to_use)

    return leads, metadata


def load_ecg_csv(path, columnnames):
    ecg = pd.read_csv(path)

    leads = {}

    for columnname in columnnames:
        leads[columnname] = np.asarray(list(ecg[columnname]))

    # TODO: read metadata from separate file or column
    metadata = {'sampling_rate_sec': 500,
                'unitofmeasurement': 'uV',
                'length_sec': 10,
                'length_timesteps': 5000}

    return leads, metadata


def save_ecg_csv(path, ecg):
    logging.debug('Saving {}'.format(path))
    leads = ecg['leads']
    df = pd.DataFrame.from_dict(leads)
    df.to_csv(path)


def load_ecgs_from_redcap_snapshot(leads_to_use, record_ids_excluded,
                                   ecg_path='../../data/kerckhoff/xml_data/ecg/'):
    ecgfiles = os.listdir(ecg_path)
    ecgs = {}

    for filename in ecgfiles:
        exclude = False
        record_id = filename.replace('.xml', '')

        if record_ids_excluded is not None:
            if record_id in record_ids_excluded:
                exclude = True
                logging.info('Excluded record "{}" from dataloading (ECG)'.format(record_id))

        if exclude is False:
            leads, metadata = load_ecg_xml(ecg_path + filename, leads_to_use)
            ecgs[record_id] = {'leads': leads, 'metadata': metadata}

    return ecgs


def load_ecgs_from_ptbxl(snapshot, sampling_rate=500, leads_to_use=None, snapshot_directory='../../data/ptbxl/snapshots', record_ids_excluded=None):
    path = snapshot_directory + '/{}/'.format(snapshot)
    ecgs = load_norm_and_mi_ecgs(path, sampling_rate, leads_to_use, record_ids_excluded)

    return ecgs


def load_clinical_parameters_json(path, params_input):
    """
        Converts an ecg to a format, containing absolute voltage numbers
    :param path: the Path of the file to load
    :param params_input: list of clinical parameters to load
    :return: clinical parameters forom a single file
    """
    allparams = load_dict_from_json(path)

    inputs = {}
    outputs = {}

    if params_input is not None:
        for param in params_input:
            try:
                inputs[param] = allparams[param]
            except KeyError:
                raise Exception('Unknown clinical input parameter "{}". Aborting.'.format(param))

        assert (len(inputs)) > 0

    return inputs


def extract_clinical_parameters_from_df(df, params_input, params_output):
    inputs = {}
    outputs = {}

    for p in params_input:
        inputs[p] = df[p]

    for p in params_output:
        outputs[p] = df[p]

    return inputs, outputs


def load_metadata(metadata_id, metadata_directory='./../../data/metadata/'):
    """
        Loads a metadata file
    :param metadata_id: ID of the metadata file(stored in config)
    :param metadata_directory: Directory where the metadata file is stored
    :return: a json metadata file
    """
    path = metadata_directory + metadata_id + '.json'

    try:
        with open(path, 'r') as f:
            metadata = json.load(f)
        return metadata

    except FileNotFoundError:
        raise Exception('Metadata file at "{}" does not exist. Aborting.'.format(path))

    except json.decoder.JSONDecodeError as e:
        raise Exception('Metadata file at "{}" contains errors. JSON could not be parsed. Aborting. Error message: {}'.format(path, str(e)))


def one_hot_encode_clinical_parameters(clinical_parameters, metadata):
    """
        Uses one hot encoding to encode the parameters according to the supplied metadata
    :param clinical_parameters: Clinical parameters to be encoded
    :param metadata: metadata that describes the encoding rules
    :return: encoded parameters
    """
    encoded = {}

    for param in clinical_parameters:
        value = clinical_parameters[param]
        try:
            encoded[param] = np.array(metadata[param]['values_one_hot'][value])
        except KeyError:
            raise Exception(
                'One hot encoding failed because of missing rule for clinical parameter "{}" and value "{}". Check value or implement rule!'.format(param, value))

    return encoded


def scale_ecg(ecg, factor):
    """
            Scales an ECG by a scaling factor and adjusts the unit of measurement accordingly
        :param ecg: ECG containing multiple leads
        :param factor : a scaling value
        :return: an ECG scaled by the factor
        """
    for lead_id in ecg['leads']:
        lead = np.array(ecg['leads'][lead_id])
        ecg['leads'][lead_id] = lead * factor

    if factor == 1 / 1000 and ecg['metadata']['unitofmeasurement'] == 'uV':
        ecg['metadata']['unitofmeasurement'] = 'mV'
    else:
        ecg['metadata']['unitofmeasurement'] = ecg['metadata']['unitofmeasurement'] + '*' + str(factor)

    return ecg


def scale_ecgs(ecgs, factor):
    """
            Scales a list of ECGs by a scaling factor
        :param ecgs: list of ECGs
        :param factor : a scaling value
        :return: a list of ECGs scaled by the factor
        """
    scaled_ecgs = {}

    for record_id in ecgs:
        scaled_ecgs[record_id] = scale_ecg(ecgs[record_id], factor)

    return scaled_ecgs


def derive_ecg_variants_multi(ecgs, variants):
    """
        Converts a list of ECGs to the same format, only containing absolute voltage numbers
    :param ecgs: List of ECGs
    :param variants: possible formats
    :return: a list of ECGs with absolute voltage values
    """
    derived_ecgs = {}

    for record_id in ecgs:
        derived_ecgs[record_id] = derive_ecg_variants(ecgs[record_id], variants)

    return derived_ecgs


def calculate_delta_for_lead(lead):
    """
        Converts a lead that is recorded as delta values into a lead with absolute values
    :param lead: a lead with delta voltage values
    :return: a lead with absolute voltage values
    """
    delta_list = []

    for index in range(0, len(lead) - 1):
        delta_list.append(lead[index + 1] - lead[index])

    delta_list = np.round(np.array(delta_list), 6)

    return delta_list


def calculate_delta_for_leads(leads):
    """
        Converts a leads recorded as delta values into a leads with absolute values
    :param leads: leads with delta voltage values
    :return: leads with absolute voltage values
    """
    delta_leads = {}

    for lead_id in leads:
        delta_leads[lead_id] = calculate_delta_for_lead(leads[lead_id])

    return delta_leads


def derive_ecg_from_delta_for_lead(delta_lead):
    value_list = []
    v = delta_lead[0]

    for index in range(0, len(delta_lead)):
        v += delta_lead[index]
        value_list.append(v)

    value_list = np.round(np.array(value_list), 6)

    return value_list


def derive_ecg_from_delta_for_leads(delta_leads):
    leads = {}

    for lead_id in delta_leads:
        leads[lead_id] = derive_ecg_from_delta_for_lead(delta_leads[lead_id])

    return leads


def derive_ecg_variants(ecg, variants):
    """
        Converts an ecg to a format, containing absolute voltage numbers
    :param ecg: an ECG
    :param variants: possible formats
    :return: an ECG with absolute voltage values
    """
    derived_ecg = {}
    for variant in variants:
        if variant == 'ecg_raw':
            derived_ecg[variant] = ecg['leads']
        elif variant == 'ecg_delta':
            derived_ecg[variant] = calculate_delta_for_leads(ecg['leads'])

    derived_ecg['metadata'] = ecg['metadata']

    return derived_ecg


def crop_ecg(ecg, start, end):
    for lead_id in ecg['leads']:
        lead = np.array(ecg['leads'][lead_id])
        ecg['leads'][lead_id] = lead[start:end]

    ecg['metadata']['crop_start'] = start
    ecg['metadata']['crop_end'] = end

    update_metadata_length(ecg, start, end)

    return ecg


def load_crops(crop_id, crop_path='../../data/crops/'):
    return load_dict_from_json(crop_path + crop_id + '.json')


def crop_ecgs(ecgs, crop_id):
    if crop_id is not None:
        crops = load_crops(crop_id)
        cropped_ecgs = {}

        for record_id in ecgs:
            try:
                start = crops[record_id]['start']
                end = crops[record_id]['end']
                cropped_ecgs[record_id] = crop_ecg(ecgs[record_id], start, end)
            except KeyError:
                logging.warning(
                    'No crop markers found for record {}. Please check if this was intended. Proceeding anyway.'.format(
                        record_id))
                cropped_ecgs[record_id] = ecgs[record_id]

        return cropped_ecgs

    else:
        return ecgs


def extract_subsample_from_leads_dict_based(leads, start, end):
    leads_subsampled = {}

    for lead_id in leads:
        leads_subsampled[lead_id] = leads[lead_id][start:end]

    return leads_subsampled


def update_metadata_length(ecg, start, end):
    secs_old = ecg['metadata']['length_sec']
    timesteps_old = ecg['metadata']['length_timesteps']

    timesteps_new = end - start
    secs_new = round(timesteps_new * secs_old / timesteps_old, 1)

    ecg['metadata']['length_sec'] = secs_new
    ecg['metadata']['length_timesteps'] = timesteps_new


def update_length_in_metadata(metadata, start, end):
    secs_old = metadata['length_sec']
    timesteps_old = metadata['length_timesteps']

    timesteps_new = end - start
    secs_new = round(timesteps_new * secs_old / timesteps_old, 1)

    metadata['length_sec'] = secs_new
    metadata['length_timesteps'] = timesteps_new


def extract_subsample_from_ecg_dict_based(ecg, start, end):
    subsample_ecg = {}

    for elem in ecg:
        if str(elem).startswith('ecg_'):
            subsample_ecg[elem] = extract_subsample_from_leads_dict_based(ecg[elem], start, end)
        else:
            subsample_ecg[elem] = dict(ecg[elem])

    subsample_ecg['metadata']['subsample_start'] = start
    subsample_ecg['metadata']['subsample_end'] = end

    update_metadata_length(subsample_ecg, start, end)

    return subsample_ecg


def extract_subsample_from_ecg_matrix_based(ecg, start, end):
    return ecg[start:end]


def subsample_ecgs(ecgs, subsampling_factor, window_size, ecg_variant='ecg_raw'):
    collected_subsamples = [] #TODO vielleicht dict für untersch. ekg varianten
    collected_clinical_parameters = []
    collected_metadata = []
    collected_record_ids = []

    for record_id in ecgs:
        start = 0
        record = ecgs[record_id]
        metadata = record['metadata']
        length = metadata['length_timesteps']
        ecg = convert_lead_dict_to_matrix(record[ecg_variant])
        clinical_parameters = concatenate_one_hot_encoded_parameters(record['clinical_parameters_inputs'])

        if not length > window_size:
            raise Exception('Record "{}" is shorter ({}) than the configured subsampling window size of {} timesteps. Aborting.'.format(record_id, length, window_size))

        stride = int((length - window_size) / subsampling_factor)

        for i in range(subsampling_factor):
            end = start + window_size

            if end > length:
                break

            subsample = extract_subsample_from_ecg_matrix_based(ecg, start, end)

            record_id_new = '{}_{}'.format(record_id, i)
            metadata_new = dict(metadata)
            update_length_in_metadata(metadata, start, end)
            metadata_new['subsample_start'] = start
            metadata_new['subsample_end'] = end
            metadata_new['original_record_id'] = record_id
            metadata_new['record_id'] = record_id_new

            collected_subsamples.append(subsample)
            collected_clinical_parameters.append(clinical_parameters)
            collected_metadata.append(metadata_new)
            collected_record_ids.append(record_id_new)

            start = start + stride

    return collected_record_ids, collected_metadata, collected_clinical_parameters, collected_subsamples


def load_clinical_parameters_from_redcap_snapshot(clinical_parameters_inputs,
                                                  record_ids_excluded,
                                                  clinical_parameters_directory):
    """
        Fetches the clinical parameters corresponding to the ECGs
    :param clinical_parameters_inputs: list of parameters to load from the files
    :param record_ids_excluded: List of records to be ignored
    :param clinical_parameters_directory: the folder path where clinical parameter files are stored
    :return: loaded clinical parameters
    """
    parameterfiles = os.listdir(clinical_parameters_directory)

    clinicalparameters = {}

    for filename in parameterfiles:
        exclude = False
        record_id = filename.replace('.json', '')

        if record_ids_excluded is not None:
            if record_id in record_ids_excluded:
                exclude = True
                logging.info('Excluded record "{}" from dataloading (clinical parameters)'.format(record_id))

        if exclude is False:
            inputs = load_clinical_parameters_json(clinical_parameters_directory + filename, clinical_parameters_inputs)
            clinicalparameters[record_id] = {'clinical_parameters_inputs': inputs}

    return clinicalparameters

# TODO: No usages in project -> Delete ?
def load_clinical_parameters_from_ptbxl_snapshot(snapshot, clinical_parameters_inputs, clinical_parameters_outputs,
                                                 snapshot_directory='../../data/ptbxl/snapshots',
                                                 record_ids_excluded=None):
    path = snapshot_directory + '/{}/'.format(snapshot)
    records = load_patientrecords_mi_norm(path, record_ids_excluded)
    clinicalparameters = {}

    for index, row in records.iterrows():
        inputs, outputs = extract_clinical_parameters_from_df(row, clinical_parameters_inputs,
                                                              clinical_parameters_outputs)
        clinicalparameters[row.record_id] = {'clinical_parameters_inputs': inputs,
                                             'clinical_parameters_outputs': outputs}

    return clinicalparameters


def validate_and_clean_clinical_parameters_for_records(records, metadata):
    """
        Validates that the clinical parameters are within their accepted value ranges, cleans differing parameter values
    :param records: records containing clinical parameters to be validated
    :param metadata:  metadata corresponding to the files
    :return: validated and cleaned records
    """
    validated_and_cleaned = {}

    for recid in records:
        try:
            inputs = validate_and_clean_clinical_parameters(records[recid]['clinical_parameters_inputs'], metadata)
        except Exception as e:  # In case of other exceptions, raise new exception with record-id information added
            raise Exception('Record-ID {}: {}'.format(recid, e))

        validated_and_cleaned[recid] = {'clinical_parameters_inputs': inputs}

    return validated_and_cleaned


def validate_and_clean_clinical_parameters(clinical_parameters, metadata):
    """
        Validates that the clinical parameters are within their accepted value ranges, cleans differing parameter values
    :param clinical_parameters: Clinical parameters to be validated
    :param metadata: metadata corresponding to the files
    :return: validated and cleaned clinical parameters
    """
    validated_and_cleaned = {}

    for param in clinical_parameters:
        value = clinical_parameters[param]

        if metadata[param]['type'] == 'char':
            value_vc = validate_and_clean_char(param, str(value),
                                               metadata[param]['values_allowed'],
                                               metadata[param]['values_replace'])
        elif metadata[param]['type'] == 'float':
            value_vc = validate_and_clean_float(param, value,
                                                metadata[param]['valmin'],
                                                metadata[param]['valmax'])
        else:
            raise Exception('Unkown parameter: "{}". Please implement validation and cleansing rule!'.format(param))

        validated_and_cleaned[param] = value_vc
    return validated_and_cleaned


def categorize_clinical_parameters(clinical_parameters, metadata):
    """
        Categorizes real valued data into value bands
    :param clinical_parameters: Clinical parameters to be categorised
    :param metadata: metadata corresponding to the files
    :return: categorized parameters
    """
    for param in clinical_parameters:
        if "categorization_rules" in metadata[param]:
            categorylist = metadata[param]['categorization_rules']

            for category in categorylist:
                if category['end'] in ['Infinity', 'INF', 'NaN']:
                    clinical_parameters[param] = category['name']
                    break
                elif category['start'] <= clinical_parameters[param] < category['end']:
                    clinical_parameters[param] = category['name']
                    break

    return clinical_parameters


def categorize_clinical_parameters_for_records(records, metadata):
    """
        Categorizes real valued data within the records into value bands
    :param records: Records containing clinical parameters
    :param metadata: metadata corresponding to the files
    :return: Records containing categorized clinical data
    """
    categorized = {}

    for recid in records:
        inputs = categorize_clinical_parameters(records[recid]['clinical_parameters_inputs'], metadata)
        categorized[recid] = {'clinical_parameters_inputs': inputs}

    return categorized


def one_hot_encode_clinical_parameters_for_records(records, metadata):
    """
            Uses one hot encoding to encode the parameters within the records according to the supplied metadata
        :param records: records, containing clinical parameters to be encoded
        :param metadata: metadata that describes the encoding rules
        :return: encoded records
        """
    onehot_encoded = {}

    for recid in records:
        inputs = one_hot_encode_clinical_parameters(records[recid]['clinical_parameters_inputs'], metadata)
        onehot_encoded[recid] = {'clinical_parameters_inputs': inputs}

    return onehot_encoded


def combine_ecgs_and_clinical_parameters(ecgs, clinical_parameters):
    """
        Combines ECGs and their corresponding clinical parameters
    :param ecgs: List of ECGs
    :param clinical_parameters: Corresponding clinical parameters
    :return: Medical data for each patient including ECGs and the patients clinical parameters
    """
    combined = {}

    for record_id in ecgs:
        ecg = ecgs[record_id]

        try:
            cp = clinical_parameters[record_id]
        except KeyError:
            logging.warning(
                'No clinical parameters available in datapipeline for record "{}". Skipping record.'.format(record_id))
            continue

        combined[record_id] = dict(ecg)
        combined[record_id].update(cp)

    return combined


def save_dataset(records, dataset_id, dataset_directory='../../data/datasets/'):
    make_dirs_if_not_present(dataset_directory)
    pickle_data(records, dataset_directory + dataset_id + '.pickled')


def load_dataset(dataset_id, dataset_directory='../../data/datasets/'):
    return unpickle_data(dataset_directory + dataset_id + '.pickled')


def load_split(split_id, split_directory='../../data/splits/'):
    return load_dict_from_json(split_directory + split_id + '.json')


def load_and_split_dataset(dataset_id, split_id, dataset_directory='../../data/datasets/',
                           split_directory='../../data/splits/'):
    records_split = {}

    records = load_dataset(dataset_id, dataset_directory=dataset_directory)

    split = load_split(split_id, split_directory=split_directory)

    for group in split:
        records_split[group] = {}
        record_counter = {recid: 0 for recid in split[group]}

        for record_id in split[group]:
            if record_counter[record_id] > 0:
                records_split[group]['{}_{}'.format(record_id, record_counter[record_id])] = records[record_id]
            else:
                try:
                    records_split[group][record_id] = records[record_id]
                except KeyError:
                    raise Exception(
                        'Record "{}" contained in split group "{}" of split "{}" not available in dataset "{}". Aborting.'.format(record_id, group, split_id, dataset_id))

            record_counter[record_id] += 1

    return records_split, records


def extract_subdict_from_dict(dct, subdict):
    collected = []

    for lv1 in dct:
        collected.append(dct[lv1][subdict])

    return collected


def concatenate_one_hot_encoded_parameters(dct):
    collected = []

    for p in dct:
        collected += list(dct[p])

    return np.array(collected)


def concatenate_one_hot_encoded_parameters_for_records(records):
    collected = []

    for record in records:
        collected.append(concatenate_one_hot_encoded_parameters(record))

    return collected


def convert_lead_dict_to_matrix(leads, shape_switch=True):
    collected = []

    for lead_id in leads:
        collected.append(leads[lead_id])

    collected = np.asarray(collected)

    if shape_switch:
        collected = perform_shape_switch(collected)

    return collected


def convert_lead_dict_to_matrix_for_records(records):  # TODO: Performance issues -> generate matrix before subsampling
    collected = []
    first = convert_lead_dict_to_matrix(records[0])
    test = np.zeros((len(records), np.shape(first)[0], np.shape(first)[1]))

    i = 0
    for record in records:
        if i%100 == 0:
            print(i, '/', len(records))
        # collected.append(convert_lead_dict_to_matrix(record))
        test[i] = convert_lead_dict_to_matrix(record)
        i = i + 1

    # return collected
    return test


def derive_binary_one_hot_classes_for_list_of_labels(labels):
    collected = []

    for label in labels:
        for b in ['TRUE', 'FALSE']:
            collected.append('{}_{}'.format(label, b))

    return collected


def extract_element_from_dicts(dicts, element):
    collected = []

    for d in dicts:
        collected.append(d[element])

    return collected


def save_split(split_id, records_train, records_val, split_dir='../../data/splits'):
    make_dirs_if_not_present(split_dir)
    split = {'training': records_train, 'validation': records_val}
    path = '{}/{}.json'.format(split_dir, split_id)

    save_dict_as_json(split, path)


def shuffle_based_on_random_seed(records, random_seed):
    r_sorted = sorted(records)
    r_shuffled = shuffle(r_sorted, random_state=random_seed)

    return r_shuffled


def assign_stratified_records_to_k_groups(stratification_groups, k, random_seed):
    grouped_records = {i: [] for i in range(k)}
    g = 0

    # Assign records to k groups, homogeneously based on stratification
    for sg in stratification_groups:

        # Shuffle records before assignment
        records = shuffle_based_on_random_seed(stratification_groups[sg], random_seed)

        # Assign records to groups, run through groups 0 to k-1
        for r in records:
            grouped_records[g].append(r)
            g += 1
            if g == k:
                g = 0

    return grouped_records


def generate_cross_validation_splits(split_id, stratification_groups, variables, random_seed, split_dir='../../data/splits'):
    k = variables['k']

    # Assign records to k groups, homogeneously based on stratification
    grouped_records = assign_stratified_records_to_k_groups(stratification_groups, k, random_seed)

    # k times, split records into 1 validation part and k-1 training parts
    for g in grouped_records:

        # The current g is used for validation
        records_validation = grouped_records[g]
        records_train = []

        # Use all but current group for training
        for x in grouped_records:
            if x != g:
                records_train += grouped_records[x]

        # Save split
        sub_split_id = '{}_k{}'.format(split_id, g)
        save_split(sub_split_id, records_train, records_validation, split_dir=split_dir)


def generate_bootstrapping_splits(split_id, stratification_groups, variables, random_seed, split_dir='../../data/splits'):
    n = variables['n']

    for i in range(n):
        train = []
        val = []

        for sg in stratification_groups:

            # Shuffle records before assignment
            records = shuffle_based_on_random_seed(stratification_groups[sg], random_seed)

            # Randomly draw n=len(records) records witn replacement from all records, use for training
            train_tmp = resample(records, replace=True, n_samples=len(records), random_state=random_seed)

            # Use undrawn records for validation
            val_tmp = list(set(records) - set(train_tmp))

            train += train_tmp
            val += val_tmp

        # Save split
        sub_split_id = '{}_n{}'.format(split_id, i)
        save_split(sub_split_id, train, val, split_dir=split_dir)


def generate_ratio_based_split(split_id, stratification_groups, variables, random_seed, split_dir='../../data/splits'):
    train = []
    val = []
    ratio = variables['ratio']

    # Assign records to k groups, homogeneously based on stratification
    for sg in stratification_groups:
        # Shuffle records before assignment
        records = shuffle_based_on_random_seed(stratification_groups[sg], random_seed)
        split_point = int(len(records) * ratio)
        train_tmp = records[:split_point]
        val_tmp = records[split_point:]
        train += train_tmp
        val += val_tmp

    # Save split
    save_split(split_id, train, val, split_dir=split_dir)


def generate_splits_for_stratification_groups_and_validation_type(split_id, stratification_groups, validation_type, variables, random_seed, split_dir='../../data/splits'):
    if validation_type == 'cross_validation':
        generate_cross_validation_splits(split_id, stratification_groups, variables, random_seed, split_dir=split_dir)
    elif validation_type == 'bootstrapping':
        generate_bootstrapping_splits(split_id, stratification_groups, variables, random_seed, split_dir=split_dir)
    elif validation_type == 'single' or validation_type is None:
        generate_ratio_based_split(split_id, stratification_groups, variables, random_seed, split_dir=split_dir)


def generate_splits_for_dataset_and_validation_type(split_id, dataset_id, validation_type, stratification_variable, random_seed, variables, dataset_directory='../../data/datasets/', split_directory='../../data/splits'):
    # Load records from dataset
    records = load_dataset(dataset_id, dataset_directory)
    record_ids = [r for r in records]

    # Extract label values for stratification
    values = [np.argmax(records[r]['clinical_parameters_outputs'][stratification_variable]) for r in records]

    # Create stratification groups
    stratification_groups = {v: [] for v in values}

    # Assign records to stratification groups based on strat. variable
    for r, v in zip(record_ids, values):
        stratification_groups[v].append(r)

    # Generate splits based on stratification groups and validation type
    generate_splits_for_stratification_groups_and_validation_type(split_id, stratification_groups, validation_type, variables, random_seed, split_dir=split_directory)

