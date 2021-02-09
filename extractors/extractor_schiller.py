"""
 Authors: Nils Gumpfer, Joshua Prim
 Version: 0.1

 Extractor for Schiller ECGs

 Copyright 2020 The Authors. All Rights Reserved.
"""
import pandas as pd
import PyPDF2
from PyPDF2 import filters
import numpy as np
import math
from os import walk
from extractors.abstract_extractor import AbstractExractor
from utils.data.visualisation import visualiseIndividualfromDF
from utils.extract_utils.extract_utils import rotate_origin_only, move_along_the_axis, scale_values_based_on_eich_peak, \
    create_measurement_points, adjust_leads_baseline, plot_leads, preprocess_page_content, extract_graphics_string
from utils.misc.datastructure import perform_shape_switch
import logging
from tqdm import tqdm
import os


class SchillerExtractor(AbstractExractor):

    def __init__(self, params):
        super().__init__(params)

        if 'ecg_path_source' not in self.params:
            raise ValueError('ecg_path_source is not set in params')
        else:
            self.path_source = params['ecg_path_source']

        if 'ecg_path_sink' not in self.params:
            raise ValueError('ecg_path_sink is not set in params')
        else:
            self.path_sink = params['ecg_path_sink']

        # reference value for the calibration jag
        self.eich_ref = 1000

        # extracted height for the calibration jag in PDF
        self.eichzacke = 28.34800000000001

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

        # factor for scaling
        self.gamma = self.eich_ref / self.eichzacke

        # name of the leads
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    def extract(self):
        for file_name in tqdm(os.listdir(self.path_source)):
            logging.info('Converting "{}"'.format(file_name))
            try:
                lead_list = self.extract_leads_from_pdf(file_name)

                if lead_list is not None:
                    new_lead_list = []

                    for lead in lead_list:
                        tmp_lead = []

                        # Preprocess extracted vectors
                        for t in lead:
                            x, y = rotate_origin_only(float(t[0]), float(t[1]), math.radians(0))
                            tmp_lead.append([x, y])

                        new_lead = move_along_the_axis(tmp_lead)

                        # Scale values based on eich peak
                        new_lead = scale_values_based_on_eich_peak(new_lead, self.gamma)

                        # Create (e.g. 5000) measurement points based on the unevenly distributed points
                        measurement_points = create_measurement_points(new_lead, self.number_of_points)

                        # Collect converted leads
                        new_lead_list.append(measurement_points)

                    # Convert lead list to dataframe
                    df_leads = pd.DataFrame(perform_shape_switch(new_lead_list), columns=self.lead_names)

                    # Adjust baseline position of each lead
                    df_leads = adjust_leads_baseline(df_leads)

                    # Plot leads of ECG if config is set to do so
                    if self.show_visualisation:
                        visualiseIndividualfromDF(df_leads,self.vis_scale)

                    df_leads.to_csv(('{}{}.csv'.format(self.path_sink, file_name.replace(".pdf", ""))),
                                    index=False)
                else:
                    logging.error('Lead list is none')
            except Exception:
                logging.warning(('Failed to extract ' + str(file_name)))

        return True

    def extract_leads_from_pdf(self, filename):
        reader = PyPDF2.PdfFileReader(open(self.path_source + filename, 'rb'))

        num_pages = reader.getNumPages()
        if num_pages == 3:
            pg1 = reader.getPage(1).getContents()._data
            pg2 = reader.getPage(2).getContents()._data
        else:
            pg1 = reader.getPage(0).getContents()._data
            pg2 = reader.getPage(1).getContents()._data
        pg1 = preprocess_page_content(pg1)
        pg1 = extract_graphics_string(pg1)
        pg2 = preprocess_page_content(pg2)
        pg2 = extract_graphics_string(pg2)

        leads1 = self.collectLeads(pg1, 7, 18)
        leads2 = self.collectLeads(pg2, 7, 18)
        leads = leads1 + leads2

        correct_extracted = [False if (len(x) < 700 or len(x) > 800) else True for x in leads]
        if False in correct_extracted:
            leads1 = self.collectLeads(pg1, 8, 19)
            leads2 = self.collectLeads(pg2, 8, 19)
            leads = leads1 + leads2

        correct_extracted = [False if (len(x) < 700 or len(x) > 800) else True for x in leads]
        if False in correct_extracted:
            leads1 = self.collectLeads(pg1, 9, 20)
            leads2 = self.collectLeads(pg2, 9, 20)
            leads = leads1 + leads2

        correct_extracted = [False if (len(x) < 700 or len(x) > 800) else True for x in leads]
        if False in correct_extracted:
            raise Exception('Special case: External limits for the extraction may not be correct!')

        return leads

    def collectLeads(self, graphicsstring, lower=7, upper=18):
        leads = []
        leads_raw = graphicsstring[1].split('C')

        for z in leads_raw[lower:upper: 2]:
            tmp = str(z).split('\n')

            lead = []
            for p in tmp:
                coordinates = p.split(' ')
                if len(coordinates) == 2:
                    lead.append(coordinates)

            lead = np.array(lead)
            leads.append(lead)

        return leads


if __name__ == '__main__':
    path_source = '../data/pdf_data/pdf_schiller/original_ecgs/'
    path_sink = '../data/pdf_data/pdf_schiller/extracted_ecgs/'

    params = {
        'ecg_path_sink': path_sink,
        'ecg_path_source': path_source,
        'number_of_points': 5000,
        'show_visualisation': True,
    }

    tmp = SchillerExtractor(params)
    tmp.extract()
