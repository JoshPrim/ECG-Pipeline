"""
 Authors: Nils Gumpfer, Joshua Prim
 Version: 0.1

 Abstract class for extracting the lead from pdfs

 Copyright 2020 The Authors. All Rights Reserved.
"""

from abc import ABC, ABCMeta, abstractmethod


class AbstractExractor(ABC):
    __metaclass__ = ABCMeta

    def __init__(self, params):
        super().__init__()
        self.params = params

    @abstractmethod
    def extract(self):
        """
            function to extract the ECG leads from the PDF
        """
        pass

