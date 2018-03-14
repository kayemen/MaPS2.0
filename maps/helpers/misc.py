from maps.settings import setting, make_or_clear_directory

import os
import shutil
import openpyxl as xl
import csv
import pickle
import logging
import uuid
import threading
import matplotlib.pyplot as plt
logger = logging.getLogger('maps.helper.misc')


def unpickle_object(file_name, dumptype='pkl'):
    '''
    Load data from file.
    Loads as pkl format by default

    For csv and xlsx:
    data must be a list of tuples. Each item of list (tuple) is 1 row of data.
    The elements within the row are the cells in the row.
    INPUTS:
        data(LIST): List of tuples of row data
    '''
    workspace = setting['workspace']

    final_path = os.path.join(workspace, file_name)

    print 'Loading "%s"' % final_path

    if not final_path.endswith(dumptype):
        final_path = '%s.%s' % (final_path, dumptype)

    data = None

    if dumptype == 'pkl':
        with open(final_path, 'r') as pkfp:
            data = pickle.load(pkfp)

    elif dumptype == 'csv':
        data = []
        with open(final_path, 'rb') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                data.append(row)

    return data


def pickle_object(data, file_name, dumptype='pkl'):
    '''
    Dump data to file.
    Dumps as pkl format by default

    For csv and xlsx:
    data must be a list of tuples. Each item of list (tuple) is 1 row of data.
    The elements within the row are the cells in the row.
    INPUTS:
        data(LIST): List of tuples of row data
    '''
    workspace = setting['workspace']

    final_path = os.path.join(workspace, file_name)

    if not final_path.endswith(dumptype):
        final_path = '%s.%s' % (final_path, dumptype)

    if dumptype == 'xlsx':
        wb = xl.Workbook()
        ws = wb.active
        for row in data:
            ws.append(row)

        wb.save(final_path)

    elif dumptype == 'pkl':
        with open(final_path, 'w') as pkl_file:
            pickle.dump(data, pkl_file)

    elif dumptype == 'csv':
        with open(final_path, 'wb') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(data)

    print 'Data saved to %s' % final_path


MAX_THREAD_COUNT = 20

threadLimiter = threading.BoundedSemaphore(MAX_THREAD_COUNT)


def write_error_plot(fig, file_name):
    error_data_dir = os.path.join(setting['workspace'], 'error_data')
    make_or_clear_directory(error_data_dir, clear=False)
    fig.savefig(os.path.join(error_data_dir, file_name))


def write_log_data(log_data, label=''):
    print 'writing %s to %s' % (label, os.path.join(setting['workspace'], 'run_data.txt'))
    with open(os.path.join(setting['workspace'], 'run_data.txt'), 'a') as fp:
        data = label + ':' + str(log_data) if label != '' else str(log_data)
        fp.write(data)
        fp.write('\n')


class LimitedThread(threading.Thread):
    current_threads = []
    current_thread_count = 0

    def run(self):
        threadLimiter.acquire()
        try:
            super(LimitedThread, self).run()
        finally:
            threadLimiter.release()


def zook_approval_function(zooks, z_stamps=None):
    print 'Process bad zooks:'
    bz = zooks.get_bad_zooks()
    print '%d bad zooks found.\n%d zooks have bad zstage.\n%d zooks have bad convexity in correlation.\n%d zooks have bad shift' % (len(bz), len(filter(lambda x: x.is_bad_zstage, bz)), len(filter(lambda x: x.is_bad_convexity, bz)), len(filter(lambda x: x.is_bad_shift, bz)))

    print 'Press "s" to skip all remaining checks and drop all. Press "p" to approve all remaining bad zooks'

    choice = ''
    for zook in bz:
        if choice != 'p':
            err_msg = ''
            err_msg += '|Bad zstage' if zook.is_bad_zstage else ''
            err_msg += '|Bad convexity' if zook.is_bad_convexity else ''
            err_msg += '|Bad correlation shift' if zook.is_bad_shift else ''
            err_msg += '|Unknown error' if zook.generic_error else ''
            print 'Zook#%d%s' % (zook.id, err_msg)
            if z_stamps is not None:
                choice = raw_input(
                    'Do you want to review frame plot?[y/n]').lower()
                if choice == 'y':
                    # Plot frame
                    pass
            choice = raw_input('Drop zook? [y/n/s/p]').lower()
            if choice == 'n':
                zook.override = True
            if choice == 's':
                break
            if choice == 'p':
                zook.override = True
        else:
            zook.override = True
    print 'Final stats'
    print '%d zooks dropped. %d zooks usable' % (len(zooks.get_bad_zooks()), len(zooks))


if __name__ == '__main__':
    pass
    # make_or_clear_directory('./test')
