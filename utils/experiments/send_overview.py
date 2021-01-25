import argparse
import os

from utils.experiments.evaluation import generate_overview_excel_sheet_for_experiments
from utils.misc.mail import send_file_via_email

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiments', nargs='+', help='Experiments', required=True)
parser.add_argument('-m', '--metrics', nargs='+', help='Metrics', required=True)
parser.add_argument('-cm', '--calculationmethod', type=str, help='Calculation method', required=True)
parser.add_argument('-cls', '--classname', type=str, help='Class name', required=True)
parser.add_argument('-r', '--recipient', type=str, help='Email recipient', required=True)
args = parser.parse_args()

generate_overview_excel_sheet_for_experiments(args.experiments,
                                              args.metrics,
                                              args.calculationmethod,
                                              args.classname,
                                              'overview.xlsx')

send_file_via_email('overview.xlsx', args.recipient)
os.remove('overview.xlsx')
