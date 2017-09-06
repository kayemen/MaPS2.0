# File containing default values for use in MaPS
import os
import json
import csv
import time
# import get_inputs

BASE_DIR = os.path.dirname(__file__)
JOBS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'jobs')
THREADED = False
MULTIPROCESSING = True
NUM_PROCESSES = None
NUM_CHUNKS = 5
TIMEOUT = 20

setting = {}
descriptions = {}
helptexts = {}
hidden_settings = {}

step_count = 0


def make_or_clear_directory(path, clear=False):
    if clear and os.path.isdir(path):
        print 'Clearing directory - %s' % path
        shutil.rmtree(path)
    if not os.path.isdir(path):
        print 'Creating directory - %s' % path
        os.makedirs(path)


def read_setting_from_json(job_name, setting_json_path=os.path.join(BASE_DIR, 'current_inputs.json')):
    global setting, descriptions, helptexts

    setting_json = json.load(open(setting_json_path))

    varnames = [obj['varname'] for obj in setting_json]
    get_varobj = lambda varname: setting_json[varnames.index(varname)]

    for varobj in varnames:
        setting_obj = get_varobj(varobj)

        if setting_obj['value'] != '':
            if setting_obj['type'] == 'int':
                setting[varobj] = int(setting_obj['value'])
            elif setting_obj['type'] == 'float':
                setting[varobj] = float(setting_obj['value'])
            elif setting_obj['type'] == 'bool':
                setting[varobj] = (setting_obj['value'].lower() == 'true')
            # TODO: Make path relative
            elif setting_obj['type'] == 'path':
                setting[varobj] = os.path.join(
                    JOBS_DIR, job_name, setting_obj['value'])
            else:
                setting[varobj] = setting_obj['value']
        else:
            setting[varobj] = None

        descriptions[varobj] = setting_obj['description']
        helptexts[varobj] = setting_obj['helptext']
        hidden_settings[varobj] = setting_obj['hidden']

    local_setting_path = os.path.join(
        JOBS_DIR, job_name, 'local_settings.json')
    if os.path.exists(local_setting_path):
        local_setting_json = json.load(open(local_setting_path))
    else:
        local_setting_json = []

    varnames = [obj['varname'] for obj in local_setting_json]
    get_varobj = lambda varname: local_setting_json[varnames.index(varname)]

    for varobj in varnames:
        setting_obj = get_varobj(varobj)

        if setting_obj['value'] != '':
            if setting_obj['type'] == 'int':
                setting[varobj] = int(setting_obj['value'])
            elif setting_obj['type'] == 'float':
                setting[varobj] = float(setting_obj['value'])
            elif setting_obj['type'] == 'bool':
                setting[varobj] = (setting_obj['value'].lower() == 'true')
            elif setting_obj['type'] == 'path':
                setting[varobj] = os.path.join(
                    JOBS_DIR, job_name, setting_obj['value'])
                make_or_clear_directory(setting[varobj])
            else:
                setting[varobj] = setting_obj['value']
        else:
            setting[varobj] = None

    clear_folder_list = (
        'cropped_bf_images',
        'bf_images_dwt',
        'bf_images_dwt_upsampled',
        'stat_images_cropped',
        'stat_images_dwt_upsampled',
        'canon_frames',
        'final_images',
        'workspace'
    )

    if not setting['usePickled']:
        for folder in clear_folder_list:
            make_or_clear_directory(setting[folder])


def write_settings_to_json(setting_objs, setting_json_path=os.path.join(BASE_DIR, 'current_inputs.json')):
    setting_obj_keys = (
        'description',
        'varname',
        'value',
        'helptext',
        'hidden',
        'type'
    )

    final_setting_objs = []

    if descriptions is None:
        descriptions = {}

    for varname in setting_objs.keys():
        setting_obj = {key: '' for key in setting_obj_keys}
        setting_obj['varname'] = varname
        setting_obj['type'] = type(setting_objs[varname]).__name__
        setting_obj['value'] = str(setting_objs[varname])
        setting_obj['hidden'] = hidden_settings.get(varname, False)
        setting_obj['description'] = descriptions.get(varname, '')
        setting_obj['helptext'] = helptexts.get(varname, '')

        final_setting_objs.append(setting_obj)

    with open(setting_json_path, 'wb') as setting_file:
        json.dump(final_setting_objs, setting_file)


def get_setting_list():
    return setting.keys()


def read_setting_from_csv(setting_csv_path=os.path.join(BASE_DIR, 'current_inputs.csv')):
    with open(setting_csv_path) as csv_file:
        setting_csv = csv.DictReader(csv_file)


def write_setting_to_csv(setting_obj, csv_file_name=os.path.join(BASE_DIR, 'current_inputs.csv')):
    pass


def display_current_settings():
    print ' - *' * 19, '-'
    print 'Running job with following settings at ', time.asctime()
    print 'Multiprocessing:', MULTIPROCESSING
    print 'Moving Frame count:', setting['frame_count']
    print 'Moving Frame start index:', setting['index_start_at']
    print 'Stationary Frame count:', setting['stat_frame_count']
    print 'Stationary Frame start index:', setting['stat_index_start_at']
    print 'Z wiggle size:', setting['slide_limit']
    print 'Z wiggle size resized:', setting['slide_limit'] * setting['resampling_factor']
    print 'Y wiggle size:', setting['y_slide_limit']
    print 'Y wiggle size resized:', setting['y_slide_limit'] * setting['resampling_factor']


def display_step(step_name, additional_info=None):
    global step_count

    step_count += 1
    print ' - *' * 19, '-'
    print 'Step %d: %s\n%s\n%s' % (
        step_count,
        step_name,
        time.asctime(),
        additional_info if additional_info is not None else ''
    )
    '- * ' * 19, '-'
