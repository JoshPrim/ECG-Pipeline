import os
import sys
import logging
import datetime
import time
import pandas as pd
import configparser
from tensorflow.keras.models import model_from_json
from utils.data.data import load_ecgs_from_redcap_snapshot, scale_ecgs, derive_ecg_variants_multi, \
    load_clinical_parameters_from_redcap_snapshot, validate_and_clean_clinical_parameters_for_records, \
    categorize_clinical_parameters_for_records, \
    one_hot_encode_clinical_parameters_for_records, \
    combine_ecgs_and_clinical_parameters, load_metadata, subsample_ecgs
import numpy as np
from extractors.extractor_schiller import SchillerExtractor
from extractors.extractor_cardiosoft import CardiosoftExtractor
from utils.data.visualisation import visualiseMulti
from utils.file.file import checkpathsandmake

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ExecutionRunner:
    def __init__(self):

        config = configparser.ConfigParser()
        read_ok = config.read('../../config.ini')

        if len(read_ok) == 0:
            raise Exception('Could not read config file.')

        self.initialize_logger(loglevel='WARNING')

        self.IS_PDf = config['pdf'].getboolean('is_pdf')
        self.new_extraction = config['pdf'].getboolean('override')

        self.vis_while_extraction = config['pdf'].getboolean('vis_while_extraction')
        self.vis_after_extraction = config['pdf'].getboolean('vis_after_extraction')
        self.vis_scale = config['pdf'].getfloat('vis_scale')
        self.combined_model = config['pdf'].getboolean('combined_model')

        self.manufacturer = config['pdf'].get('manufacturer')

        self.leads_to_use = config['pdf'].get('leads_to_use')
        self.record_ids_excluded = ''

        self.clinical_parameters_inputs = ['varid_549', 'varid_1891', 'varid_2265',
                                           'varid_2414', 'varid_2359', 'varid_558',
                                           'varid_559', 'varid_560', 'varid_557',
                                           'varid_561']
        self.metadata_id = config['general'].get('metadata_id')

        self.seconds = int(config['pdf'].get('seconds'))
        self.hz = 500

        self.subsampling_factor = config['general'].getint('subsampling_factor')
        self.subsampling_window_size = config['general'].getint('subsampling_window_size')
        self.model_supplied = config['general'].getboolean('model_supplied')

    def initialize_logger(self, loglevel='INFO'):

        consolehandler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter('%(asctime)-15s %(levelname)s %(message)s')
        consolehandler.setFormatter(formatter)

        log = logging.getLogger()

        for hdlr in log.handlers[:]:
            log.removeHandler(hdlr)
        log.addHandler(consolehandler)
        log.setLevel(loglevel)

    def run(self):

        # Preprocessing
        records, index_of_positiv = self.pre_processing()

        # If the user suppies their own model, a prediction of the extracted ecgs can be made
        if self.model_supplied:
            # Loading models
            model_list = self.load_models()

            # init result df
            result_df = pd.DataFrame(columns=['record_id', 'positive_value'])

            for record in records:
                tmp_record = {record: records[record]}

                # Subsampling
                _, _, clinical_parameters, ecg_raw = subsample_ecgs(tmp_record, self.subsampling_factor,
                                                                    self.subsampling_window_size)

                # Create ensemble and predict
                if self.combined_model:
                    net_input = [np.asarray(ecg_raw), np.asarray(clinical_parameters)]

                else:
                    net_input = [np.asarray(ecg_raw)]

                predictions_dict = self.predict(model_list, net_input)

                # Averaging the result --> for 100*6
                predictions_avg_list = []

                for i in predictions_dict:
                    tmp = predictions_dict[i].mean(axis=0)
                    predictions_avg_list.append([tmp[0], tmp[1]])

                predictions_avg = np.array(predictions_avg_list).mean(axis=0)

                positive = ("%.5f" % round((predictions_avg[index_of_positiv] * 100), 5))

                print('The positive-Value for ', record, ' is:  ', positive, '%')
                result_df = result_df.append({'record_id': record, 'positive_value': positive}, ignore_index=True)

            date_n_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
            result_df.to_csv((date_n_time + '_result.csv'), index=False)

    def predict(self, model_list, subsample_list):
        results = {}
        for model in model_list:
            tmp = model.predict_on_batch(subsample_list)
            results[model] = tmp

        return results

    def load_csv(self, path_csv):
        f = []
        for (dirpath, dirnames, filenames) in os.walk(path_csv):
            f.extend(filenames)
            break

        ecg_dict = {}

        for file_name in f:
            ecg_df = pd.read_csv(path_csv + file_name)
            ecg_df = ecg_df.astype('int32')

            file_name = file_name.replace(".csv", "")
            tmp_dict = {}
            tmp_dict2 = {}

            for column in ecg_df:
                ecg_list = ecg_df[column].tolist()

                tmp_dict[column] = ecg_list

            tmp_dict2['leads'] = tmp_dict
            tmp_dict2['metadata'] = {'sampling_rate_sec': 500, 'unitofmeasurement': 'uV', 'length_sec': 10,
                                     'length_timesteps': 5000}

            ecg_dict[file_name] = tmp_dict2

        return ecg_dict

    def pre_processing(self):

        # 1. Load ECGs
        logging.info('Loaded ECGs from snaphot')

        if self.IS_PDf:

            if self.manufacturer == 'Schiller':
                path_source = '../../data/pdf_data/pdf_schiller/original_ecgs/'
                path_sink = '../../data/pdf_data/pdf_schiller/extracted_ecgs/'
                clinical_parameters_directory = '../../data/pdf_data/pdf_schiller/clinicalparameters/'

                checkpathsandmake(path_sink)
                checkpathsandmake(path_source)
                checkpathsandmake(clinical_parameters_directory)


                params = {
                    'ecg_path_sink': path_sink,
                    'ecg_path_source': path_source,
                    'number_of_points': self.seconds * self.hz,
                    'show_visualisation': self.vis_while_extraction,
                    'vis_scale': self.vis_scale,
                }

                # New extraction
                if self.new_extraction:
                    schillerExtractor = SchillerExtractor(params)
                    schillerExtractor.extract()
                    logging.info('Schiller PDF extraction successful')
                else:
                    logging.warning('Please note that no new extraction is performed.')

                original_ecgs = self.load_csv(path_csv=path_sink)

            if self.manufacturer == 'Cardiosoft':
                path_source = './../../data/pdf_data/pdf_cardiosoft/original_ecgs/'
                path_sink = './../../data/pdf_data/pdf_cardiosoft/extracted_ecgs/'
                clinical_parameters_directory = '../../data/pdf_data/pdf_cardiosoft/clinicalparameters/'

                checkpathsandmake(path_sink)
                checkpathsandmake(path_source)
                checkpathsandmake(clinical_parameters_directory)

                params = {
                    'ecg_path_source': path_source,
                    'ecg_path_sink': path_sink,
                    'number_of_points': self.seconds * self.hz,
                    'show_visualisation': self.vis_while_extraction,
                    'vis_scale': self.vis_scale,
                }

                # New extraction
                if self.new_extraction:
                    cardiosoftExtractor = CardiosoftExtractor(params)
                    cardiosoftExtractor.extract()
                    logging.info('CardioSoft PDF extraction successful')
                else:
                    logging.warning('Please note that no new extraction is performed.')

                original_ecgs = self.load_csv(path_csv=path_sink)
        else:
            original_ecgs = load_ecgs_from_redcap_snapshot(self.leads_to_use, self.record_ids_excluded)
            clinical_parameters_directory = '../../data/xml_data/clinicalparameters/'
            checkpathsandmake(clinical_parameters_directory)
        # Visualise Extracted ECGs
        if self.vis_after_extraction:
            visualiseMulti(original_ecgs, self.vis_scale)

        # 2. Scale ECGs
        logging.info('Scaled ECGs')
        scaleded_ecgs = scale_ecgs(original_ecgs, 1 / 1000)

        # 3. Further ECG derivation
        logging.info('Derived further ECG variants')
        derived_ecgs = derive_ecg_variants_multi(scaleded_ecgs, ['ecg_raw'])

        # 4. Load clinical parameters
        logging.info('Load clinical parameters from snapshot')

        clinical_parameters = load_clinical_parameters_from_redcap_snapshot(self.clinical_parameters_inputs,
                                                                            self.record_ids_excluded,
                                                                            clinical_parameters_directory)

        # 5. Load Metadata
        logging.info('Load Metadata')
        metadata = load_metadata(self.metadata_id)

        index_of_positiv = metadata["varid_1657"]['values_one_hot']['True'].index(1)

        # 6. Validity check / replace values (blanks, 99, 88, etc.)
        logging.info('Validated and cleaned clinical parameters')
        valid_clinical_parameters = validate_and_clean_clinical_parameters_for_records(clinical_parameters, metadata)

        # 7. Categorization
        logging.info('Categorized clinical parameters')
        categorized_clinical_parameters = categorize_clinical_parameters_for_records(valid_clinical_parameters,
                                                                                     metadata)

        # 8. One-hot-encoding
        logging.info('One-hot encoded clinical parameters')
        one_hot_encoded_clinical_parameters = one_hot_encode_clinical_parameters_for_records(
            categorized_clinical_parameters, metadata)

        # 9. Combination with ECGs
        logging.info('Combined ECGs and clinical parameters')
        combined_records = combine_ecgs_and_clinical_parameters(derived_ecgs, one_hot_encoded_clinical_parameters)

        return combined_records, index_of_positiv

    def load_models(self):

        if self.combined_model:
            path = './../../models/combined_model/'
        else:
            path = './../../models/ecgmodel/'

        f = []
        for (dirpath, dirnames, filenames) in os.walk(path):
            f.extend(filenames)
            break

        model = path + 'model.json'
        models = []

        for name in f:
            if (name.startswith('weights')) and (name.endswith('.h5')):
                weight_file_path = path + name
                models.append(self.load_model(weight_file_path, model))

        return models

    def load_model(self, h5, json_file):
        # load json and create model
        json_file = open(json_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(h5)
        logging.info("Loaded model from disk")

        return loaded_model

    @staticmethod
    def bootstrap():
        exr = ExecutionRunner()
        try:
            exr.run()
        except Exception as e:
            logging.error(str(e))
            raise Exception(e.args)


if __name__ == '__main__':
    exr = ExecutionRunner()
    try:
        exr.run()
    except Exception as e:
        logging.error(str(e))
        raise Exception(e.args)
