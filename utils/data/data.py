import json
import logging
import os

import numpy as np
# Known Bug in PyCharm -> Minidom will not import correctly
# Pipeline should work as this is only part of the XML extraction
# noinspection PyUnresolvedReferences
from xml.dom import minidom

from utils.data.validation import validate_and_clean_float, validate_and_clean_char
from utils.file.file import load_string_from_file, load_dict_from_json
from utils.misc.datastructure import perform_shape_switch


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
        raise Exception(
            'Metadata file at "{}" contains errors. JSON could not be parsed. Aborting. Error message: {}'.format(path,
                                                                                                                  str(
                                                                                                                      e)))


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
                'One hot encoding failed because of missing rule for clinical parameter "{}" and value "{}". Check value or implement rule!'.format(
                    param, value))

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


def update_length_in_metadata(metadata, start, end):
    secs_old = metadata['length_sec']
    timesteps_old = metadata['length_timesteps']

    timesteps_new = end - start
    secs_new = round(timesteps_new * secs_old / timesteps_old, 1)

    metadata['length_sec'] = secs_new
    metadata['length_timesteps'] = timesteps_new


def extract_subsample_from_ecg_matrix_based(ecg, start, end):
    return ecg[start:end]


def subsample_ecgs(ecgs, subsampling_factor, window_size, ecg_variant='ecg_raw'):
    collected_subsamples = []
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
            raise Exception(
                'Record "{}" is shorter ({}) than the configured subsampling window size of {} timesteps. Aborting.'.format(
                    record_id, length, window_size))

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


def concatenate_one_hot_encoded_parameters(dct):
    collected = []

    for p in dct:
        collected += list(dct[p])

    return np.array(collected)


def convert_lead_dict_to_matrix(leads, shape_switch=True):
    collected = []

    for lead_id in leads:
        collected.append(leads[lead_id])

    collected = np.asarray(collected)

    if shape_switch:
        collected = perform_shape_switch(collected)

    return collected
