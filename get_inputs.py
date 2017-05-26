# Module to take inputs from user, or take default values/values loaded from file

class UserInputException(Exception):
    pass


def get_input(data_type):
    if data_type == 'path':
        import Tkinter as tk
        import tkFileDialog as filedialog

        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askdirectory()
        # file_path = filedialog.askopenfilename()

        return file_path
    else:
        data = raw_input('>')
        return data


def parse_input(data, data_type):
    try:
        if data_type == 'path':
            return data
        else:
            return eval('%s(data)' % (data_type, ))
    except:
        raise UserInputException('Failed to parse data')


def validate_input(data, data_type):
    if data_type == 'path':
        return True
    else:
        return eval('isinstance(data, %s)' % (data_type, ))


def get_setting_from_user(varname, data_type, default=None, helptext=''):
    """
    INPUTS:
        varname(STRING): Name of variable expecting data for
        data_type(STRING): Format of data to properly parse and return inputs
        default(VAR): Default value to use if user enters invalid data
        helptext(STRING): Help text to display to user while entering data
    OUTPUTS:
        data(VAR): Data entered by user or default value being returned from user
    """

    try:
        print('{var}{ht}-'.format(var=varname, ht='(%s)' % helptext if helptext != '' else ''))
        data = parse_input(get_input(data_type), data_type)
        if not validate_input(data, data_type):
            if default is None:
                raise UserInputException('Data type mismatch for ""%s". No default value provided' % (varname, ))
            if not validate_input(default, data_type):
                raise UserInputException('Default data type mismatch for ""%s"' % (varname, ))
            data = default
        return data
    except UserInputException, e:
        print(str(e))
    except:
        import traceback
        import sys
        traceback.print_exc()
        sys.exit(1)


def get_setting_from_object(data_type, value):
    try:
        data = parse_input(str(value), data_type)
        return data
    except:
        raise UserInputException('Invalid data entered')
