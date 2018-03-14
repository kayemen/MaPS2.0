# File containing default values for use in MaPS
import os
import json
import csv
import time
import re
import shutil
import traceback

BASE_DIR = os.path.dirname(__file__)

try:
    fp = open(os.path.join(os.path.dirname(BASE_DIR), 'job_dir.txt'), 'r')

    job_dir_choices = fp.readlines()

    job_choice = 0
    if len(job_dir_choices) == 1:
        job_choice = 1
    else:
        print '\n'.join(
            map(
                lambda x, y: '%d-%s' % (x, y),
                range(1, len(job_dir_choices) + 1),
                job_dir_choices
            )
        )

        while job_choice not in range(1, len(job_dir_choices) + 1):
            job_choice = int(raw_input('Select job:'))

    JOBS_DIR = job_dir_choices[job_choice - 1]
except:
    traceback.print_exc()
    JOBS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'jobs')

print 'Searching for jobs in:', JOBS_DIR, '...'
# JOBS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'jobs')
THREADED = False
MULTIPROCESSING = True
NUM_PROCESSES = None
NUM_CHUNKS = 5
TIMEOUT = 20

setting = {}
descriptions = {}
helptexts = {}
hidden_settings = {}
steps_performed = []
step_count = 0


def make_or_clear_directory(path, clear=False):
    print 'Checking directory - %s' % path
    if clear and os.path.isdir(path):
        print 'Clearing directory - %s' % path
        shutil.rmtree(path)
    if not os.path.isdir(path):
        print 'Creating directory - %s' % path
        os.makedirs(path)


def read_setting_from_json(job_name, setting_json_path=os.path.join(BASE_DIR, 'current_inputs.json'), save_bkups=False):
    global setting, descriptions, helptexts

    if not os.path.exists(JOBS_DIR):
        raise Exception('Invalid job directory-%s' % JOBS_DIR)

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
                make_or_clear_directory(setting[varobj])
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
        json.dump(setting_json, open(local_setting_path, 'w'))

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

    # clear_folder_list = (
    #     'cropped_bf_images',
    #     'bf_images_dwt',
    #     'bf_images_dwt_upsampled',
    #     'stat_images_cropped',
    #     'stat_images_dwt_upsampled',
    #     'canon_frames',
    #     'final_images',
    #     'workspace'
    # )

    # if not setting['usePickled']:
    #     for folder in clear_folder_list:
    #         make_or_clear_directory(setting[folder])

    if save_bkups:
        bkup_folder_list = (
            'workspace',
            'canon_frames',
            'cropped_ref_windows',
            'canon_frames_valid',
            'canon_dwt_masked',
            'final_images'
        )

        try:
            with open(os.path.join(JOBS_DIR, job_name, 'job_metadata.txt'), 'r') as metafile:
                txtdmp = metafile.read()
                last_run = txtdmp.split(
                    '--------------------------------------------------------------------------------')[-1]

            last_run = re.search(
                r'Run name:[\s]*(?P<run_name>.*)\n', last_run
            ).groupdict()['run_name']

        except:
            last_run = None
            print 'Unable to bkup workspace'
            return

        print 'Saving old workspace\n', last_run
        for folder in bkup_folder_list:
            bkup_name = setting[folder] + '_%s' % last_run

            while os.path.exists(bkup_name):
                choice = raw_input(
                    'Backup folder with previous run name already exists. Do you wanna overwrite? (Y/N): ')
                if choice.lower() == 'y':
                    make_or_clear_directory(bkup_name)
                else:
                    last_run = raw_input(
                        'Please provide alternative name for previous run backup: ')
                    bkup_name = setting[folder] + '_%s' % last_run

            try:
                print setting[folder], '->', bkup_name
                os.rename(
                    setting[folder],
                    bkup_name
                )
                make_or_clear_directory(setting[folder], clear=True)
            except:
                import traceback
                traceback.print_exc()
                print 'Could not save old workspace. Press ENTER to continue'
                raw_input()


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
    global step_count, steps_performed

    step_count += 1
    steps_performed.append(step_name)
    print ' - *' * 19, '-'
    print 'Step %d: %s\n%s\n%s' % (
        step_count,
        step_name,
        time.asctime(),
        additional_info if additional_info is not None else ''
    )
    # print '- * ' * 19, '-'
