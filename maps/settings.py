# File containing default values for use in MaPS
import json
import os
# import get_inputs

BASE_DIR = os.path.dirname(__file__)

setting = {}

def reload_current_settings(setting_json_path=os.path.join(BASE_DIR, 'current_inputs.json')):
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
            # elif setting_obj['type'] == 'path':
            #     setting[varobj] = is_path(setting_obj['value'])
            else:
                setting[varobj] = setting_obj['value']
        else:
        #     exec('%s = None' % (varobj, ), globals())
            setting[varobj] = None
