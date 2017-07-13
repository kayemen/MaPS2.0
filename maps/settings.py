# File containing default values for use in MaPS
import os
import json
import csv
# import get_inputs

BASE_DIR = os.path.dirname(__file__)

setting = {}


def read_setting_from_json(setting_json_path=os.path.join(BASE_DIR, 'current_inputs.json')):
    global setting

    setting_json = json.load(open(setting_json_path))
    varnames = [obj['varname'] for obj in setting_json]
    get_varobj = lambda varname: setting_json[varnames.index(varname)]

    for varobj in varnames:
        setting_obj = get_varobj(varobj)

        # exec('global %s' % varobj)
        if setting_obj['value'] != '':
            #     exec('%s = %s' % (varobj, setting_obj['value']), globals())
            if setting_obj['type'] == 'int':
                setting[varobj] = int(setting_obj['value'])
            elif setting_obj['type'] == 'float':
                setting[varobj] = float(setting_obj['value'])
            elif setting_obj['type'] == 'bool':
                setting[varobj] = setting_obj['value'].lower() == 'true'
            # elif setting_obj['type'] == 'path':
            #     setting[varobj] = is_path(setting_obj['value'])
            else:
                setting[varobj] = setting_obj['value']
        else:
            #     exec('%s = None' % (varobj, ), globals())
            setting[varobj] = None


def write_settings_to_json(setting_objs, setting_json_path=os.path.join(BASE_DIR, 'current_inputs.json')):
    pass


def get_setting_list():
    return setting.keys()


def read_setting_from_csv(csv_name):
    with open(csv_name) as csv_file:
        setting_csv = csv.DictReader(csv_file)


def write_setting_to_csv(setting_obj, csv_file_name=os.path.join(BASE_DIR, 'current_inputs.csv')):
    pass
