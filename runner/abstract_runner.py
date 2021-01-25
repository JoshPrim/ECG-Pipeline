"""
 Authors: Nils Gumpfer, Joshua Prim
 Version: 0.1

 Abstract class for a runner

 Copyright 2020 The Authors. All Rights Reserved.
"""

import argparse
from abc import ABC, ABCMeta, abstractmethod


class AbstractRunner(ABC):
    __metaclass__ = ABCMeta

    def __init__(self, exp_id=None, default_yes=False):
        super().__init__()

        parser = argparse.ArgumentParser()
        parser.add_argument('-e', '--exp', type=str, help='Experiment identifier')
        parser.add_argument('--yes', action='store_true', help='Skip questions, always answer YES')
        args = parser.parse_args()

        if exp_id is None:
            self.experiment_id = args.exp
        else:
            self.experiment_id = exp_id

        if args.yes is not None:
            self.default_yes = args.yes
        elif default_yes is not None:
            self.default_yes = default_yes
        else:
            self.default_yes = False

        if self.experiment_id in [None, '', ' ', '  ', '.', '..', '/', '-', '_']:
            raise Exception('You have to provide an experiment ID (parameter -e). Use --help for further details.')

    @abstractmethod
    def run(self):
        pass





