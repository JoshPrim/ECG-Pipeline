def validate_and_clean_float(param, value, valmin=None, valmax=None):
    try:
        value_vc = float(value)

        if valmax is not None:
            if value_vc > valmax:
                raise Exception('Value {} of clinical parameter "{}" exceeds allowed bounds of [{}:{}]'.format(value_vc, param, valmin, valmax))

        if valmin is not None:
            if value_vc < valmin:
                raise Exception('Value {} of clinical parameter "{}" exceeds allowed bounds of [{}:{}]'.format(value_vc, param, valmin, valmax))

    except ValueError:
        raise Exception('Value {} of clinical parameter "{}" could not be parsed to float. Aborting.'.format(value, param))

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
        raise Exception('Value {} of clinical parameter "{}" could not be parsed to int. Aborting.'.format(value, param))

    return value_vc


def validate_and_clean_char(param, value, allowed, replace=None):
    if value not in allowed:
        raise ValueError('Unexpected value "{}" for clinical parameter "{}". Aborting.'.format(value, param))

    if replace is not None:
        try:
            return replace[value]
        except KeyError:
            return value
