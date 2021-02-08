import json
import os

from PyPDF2 import PdfFileWriter, PdfFileReader


def save_dict_as_json(dct, path):
    with open(path, 'w') as fp:
        json.dump(dct, fp)
        fp.close()


def load_dict_from_json(path):
    string = load_string_from_file(path)
    dct = json.loads(string)

    return dct


def save_string_to_file(string, path):
    with open(path, 'w') as fp:
        fp.write(string)
        fp.close()


def load_string_from_file(path):
    with open(path, 'r') as fp:
        string = fp.read()
        fp.close()

    return string


def combine_pdfs(paths, targetpath, cleanup=False):
    pdf_writer = PdfFileWriter()

    for path in paths:
        pdf_reader = PdfFileReader(path)
        for page in range(pdf_reader.getNumPages()):
            pdf_writer.addPage(pdf_reader.getPage(page))

    with open(targetpath, 'wb') as fh:
        pdf_writer.write(fh)

    if cleanup:
        for path in paths:
            os.remove(path)
