"""
 Authors: Nils Gumpfer, Joshua Prim
 Version: 0.1

 Extractor for Cardiosoft ECGs

 Copyright 2020 The Authors. All Rights Reserved.
"""
import pandas as pd
import PyPDF2
from PyPDF2 import filters
import numpy as np
import math
import os

from extractors.abstract_extractor import AbstractExractor
from utils.extract_utils.extract_utils import rotate_origin_only, move_along_the_axis, scale_values_based_on_eich_peak, \
    create_measurement_points, adjust_leads_baseline, plot_leads, preprocess_page_content, extract_graphics_string
from utils.misc.datastructure import perform_shape_switch
from utils.data.visualisation import visualiseIndividualfromDF
from tqdm import tqdm
import logging


class CardiosoftExtractor(AbstractExractor):

    def __init__(self, params):
        super().__init__(params)

        if 'ecg_path_source' not in params:
            raise ValueError('ecg_path_source is not set in params')
        else:
            self.path_source = params['ecg_path_source']

        if 'ecg_path_sink' not in params:
            raise ValueError('ecg_path_sink is not set in params')
        else:
            self.path_sink = params['ecg_path_sink']
        # reference value for the calibration jag
        self.eich_ref = 1000
        # extracted height for the calibration jag in PDF
        self.eichzacke = 236.99999999999997

        if 'number_of_points' not in params:
            raise ValueError('number_of_points is not set in params')
        else:
            # number of measuring points in XML
            self.number_of_points = params['number_of_points']

        if 'show_visualisation' in params:
            self.show_visualisation = params['show_visualisation']
        else:
            self.show_visualisation = False

        if 'vis_scale' in params:
            self.vis_scale = params['vis_scale']
        else:
            self.vis_scale = 1

        self.gamma = self.eich_ref / self.eichzacke

        if 'version' not in params:
            self.version = '6.5'
        else:
            self.version = params['version']

    def extract(self):
        for file_name in tqdm(os.listdir(self.path_source)):
            logging.info('Converting "{}"'.format(file_name))
            try:
                # Extract leads from PDF
                lead_list, lead_ids, record_id = self.extract_leads_from_pdf(file_name)

                if lead_list is not None:
                    new_lead_list = []

                    for lead in lead_list:
                        tmp_lead = []

                        # Preprocess extracted vectors
                        for t in lead:
                            x, y = rotate_origin_only(float(t[0]), float(t[1]), math.radians(90))
                            tmp_lead.append([x, y])

                        new_lead = move_along_the_axis(tmp_lead)

                        # Scale values based on eich peak
                        new_lead = scale_values_based_on_eich_peak(new_lead, self.gamma)

                        # Create (e.g. 5000) measurement points based on the unevenly distributed points
                        measurement_points = create_measurement_points(new_lead, self.number_of_points)

                        # Collect converted leads
                        new_lead_list.append(measurement_points)

                    # Convert lead list to dataframe
                    df_leads = pd.DataFrame(perform_shape_switch(new_lead_list), columns=lead_ids)

                    # Adjust baseline position of each lead
                    df_leads = adjust_leads_baseline(df_leads)

                    # Plot leads of ECG if config is set to do so
                    if self.show_visualisation:
                        visualiseIndividualfromDF(df_leads,self.vis_scale)

                    df_leads.to_csv(('{}{}.csv'.format(self.path_sink, file_name.replace(".pdf", ""))),
                                    index=False)
                else:
                    logging.error('Lead list is none')
            except Exception as e:
                logging.warning(('Failed to extract ' + str(file_name)))

    def extract_leads_from_pdf(self, filename):
        reader = PyPDF2.PdfFileReader(open(self.path_source + filename, 'rb'))

        try:
            leads = []
            lead_ids = []
            record_id = None

            for p in range(reader.getNumPages()):
                if len(leads) == 12:
                    break

                page = reader.getPage(p)
                text = page.extractText()

                is_cover_page = text.startswith('Page') or text.startswith('Seite')

                if not is_cover_page:

                    self.get_version(text)

                    page_content_raw = reader.getPage(p).getContents()._data

                    page_content = preprocess_page_content(page_content_raw)
                    graphics_string = extract_graphics_string(page_content)

                    leads += self.extract_leads_from_page_content(graphics_string)
                    lead_ids += self.extract_lead_ids(text)
                    record_id = self.extract_record_id(text)


                else:
                    logging.info('Skipping cover page (page {})'.format(p))

            if len(leads) != 12:
                raise Exception('Invalid ECG with {} leads'.format(len(leads)))

        except Exception as e:
            logging.error('Could not convert "{}": '.format(filename, e))
            leads = None
            lead_ids = None
            record_id = None

        return leads, lead_ids, record_id

    def extract_lead_ids(self, pagetext):
        lines = pagetext.split('\n')

        lead_ids = lines[-7: -1]

        if lead_ids[1] == 'III':
            lead_ids[0] = 'I'
            lead_ids[1] = 'II'

        return lead_ids

    def get_version(self, pagetext):
        lines = pagetext.split('\n')

        version = []

        for element in lines:
            if 'GE CardioSoft' in element:
                version.append(element)
            elif 'GE CASE' in element:
                version.append(element)

        if 'V6.0' in version[0]:
            self.version = '6.0'
        else:
            self.version = '6.5'

    def extract_record_id(self, pagetext):
        lines = pagetext.split('\n')
        record_id = None

        for i in range(len(lines)):
            line = lines[i]

            if line.startswith('Patient'):
                parts = line.split(':')
                number = parts[1].replace(' ', '')
                date = lines[i + 2].replace('.', '-')
                time = lines[i + 4].replace(':', '-')

                record_id = '{}_{}_{}'.format(number, date, time)

                break

        return record_id

    def extract_leads_from_page_content(self, graphics_string):
        leads = []

        if float(self.version) < 6.5:
            cutting_range = [7, 13]
        else:
            cutting_range = [8, 14]

        for i in range(cutting_range[0], cutting_range[1]):
            points = graphics_string[i].split('S')[0].split('\n')
            lead = []

            for p in points:
                coordinates = p.split(' ')
                if len(coordinates) == 2:
                    lead.append(coordinates)
            lead = np.array(lead)
            leads.append(lead)

        return leads


if __name__ == '__main__':
    path_source = '../data/pdf_data/pdf_cardiosoft/original_ecgs/'
    path_sink = '../data/pdf_data/pdf_cardiosoft//extracted_ecgs/'

    params = {
        'ecg_path_source': path_source,
        'ecg_path_sink': path_sink,
        'number_of_points': 5000,
        'show_visualisation': True,
    }

    tmp = CardiosoftExtractor(params)
    tmp.extract()

