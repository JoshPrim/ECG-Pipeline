def validate_and_clean_float(param, value, valmin=None, valmax=None):
    """
            Validates that a clinical parameter are within an allowed value range
        :param param: Clinical parameter
        :param value: value of the parameter
        :param valmin: lowest allowed value of this parameter
        :param valmax: highest allowed value of this parameter
        :return: validated parameter
        """
    try:
        value_vc = float(value)

        if valmax is not None:
            if value_vc > valmax:
                raise Exception(
                    'Value {} of clinical parameter "{}" exceeds allowed bounds of [{}:{}]'.format(value_vc, param,
                                                                                                   valmin, valmax))

        if valmin is not None:
            if value_vc < valmin:
                raise Exception(
                    'Value {} of clinical parameter "{}" exceeds allowed bounds of [{}:{}]'.format(value_vc, param,
                                                                                                   valmin, valmax))

    except ValueError:
        raise Exception(
            'Value {} of clinical parameter "{}" could not be parsed to float. Aborting.'.format(value, param))

    return value_vc


def validate_and_clean_int(param, value, valmin=None, valmax=None):
    try:
        value_vc = int(value)

        if valmax is not None:
            if value_vc > valmax:
                raise Exception(
                    'Value {} of clinical parameter "{}" exceeds allowed bounds of [{}:{}]'.format(value_vc, param,
                                                                                                   valmin, valmax))

        if valmin is not None:
            if value_vc < valmin:
                raise Exception(
                    'Value {} of clinical parameter "{}" exceeds allowed bounds of [{}:{}]'.format(value_vc, param,
                                                                                                   valmin, valmax))

    except ValueError:
        raise Exception(
            'Value {} of clinical parameter "{}" could not be parsed to int. Aborting.'.format(value, param))

    return value_vc


def validate_and_clean_char(param, value, allowed, replace=None):
    """
        decodes encoded values in the clinical parameters if it is an allowed value
    :param param: Clinical parameter
    :param value: value of the parameter
    :param allowed: list of allowed values in this field
    :param replace: list on how to replace the allowed values
    :return: decoded parameter
    """

    if value not in allowed:
        raise ValueError('Unexpected value "{}" for clinical parameter "{}". Aborting.'.format(value, param))

    if replace is not None:
        try:
            return replace[value]
        except KeyError:
            return value
