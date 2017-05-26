# File containing default values for use in MaPS
import json
import get_inputs

DEFAULT_PARAMETERS_JSON = json.load(open('default_inputs.json'))

FORCE_DEFAULT = False

LOAD_ALTERNATE_INPUTS = False

INPUTS_LIST = [obj['varname'] for obj in DEFAULT_PARAMETERS_JSON]

if LOAD_ALTERNATE_INPUTS:
    try:
        DEFAULT_INPUTS.update(json.load(open('./local_default_inputs.json')))
    except:
        import traceback
        traceback.print_exc()

get_varobj = lambda varname: DEFAULT_PARAMETERS_JSON[INPUTS_LIST.index(varname)]

if FORCE_DEFAULT:
    for inputvar in INPUTS_LIST:
        obj = get_varobj(inputvar)
        exec('%s = get_inputs.get_setting_from_object(obj["type"], obj["default_val"])' % (inputvar, ))
else:
    for inputvar in INPUTS_LIST:
        obj = get_varobj(inputvar)
        exec('%s = get_inputs.get_setting_from_user(obj["description"], obj["type"], obj["default_val"])' % (inputvar, ))
