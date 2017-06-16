import os
import shutil
import openpyxl as xl
import csv
import pickle
import logging
import uuid

logger = logging.getLogger('maps.helper.misc')


def make_or_clear_directory(path):
    if os.path.isdir(path):
        logger.debug('Clearing directory - %s' % path)
        shutil.rmtree(path)
    logger.debug('Creating directory - %s' % path)
    os.makedirs(path)


def pickle_object(data, file_name=None, dumptype='pkl'):
    '''
    Dump data to csv file. data must be a list of tuples. Each item of list (tuple) is 1 row of data. The elements within the row are the cells in the row.
    INPUTS:
        data(LIST): List of tuples of row data
    '''
    data_dump = 'D:\\Scripts\\MaPS\\Data sets\\Raw data\\'  # setting#

    if file_name is None:
        # TODO: Generate unique file name
        return
        # file_name =

    final_path = os.path.join(data_dump, file_name)

    if dumptype == 'xlsx':
        wb = xl.Workbook()
        ws = wb.active
        for row in data:
            ws.append(row)

        wb.save(final_path)

    elif dumptype == 'pkl':
        with open(final_path, 'wb') as pkl_file:
            pickle.dump(data, pkl_file)
    elif dumptype == 'csv':
        with open(final_path, 'wb') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(data)

    print 'Data saved to %s' % final_path

if __name__ == '__main__':
    make_or_clear_directory('./test')
