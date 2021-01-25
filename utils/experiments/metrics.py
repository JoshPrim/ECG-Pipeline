import logging
import math

from scipy.stats import linregress
from sklearn import metrics
import numpy as np


def roc(y_true, y_pred, y_record_ids, c=0, calculation_method='subsample_level'):
    thresholds = [i / 100 for i in range(100)]
    rocvalues = {'TPR': [0.0], 'FPR': [0.0]}

    for threshold in thresholds:
        tp, fp, tn, fn, _ = confusionmatrix(y_true=y_true, y_pred=y_pred, y_record_ids=y_record_ids, c=c,
                                            threshold=threshold, calculation_method=calculation_method)

        tpr = truepositiverate(tp, fn)
        fpr = falsepositiverate(tn, fp)

        rocvalues['TPR'].append(tpr)
        rocvalues['FPR'].append(fpr)

    rocvalues['TPR'].append(1.0)
    rocvalues['FPR'].append(1.0)

    rocvalues['TPR'] = sorted(rocvalues['TPR'])
    rocvalues['FPR'] = sorted(rocvalues['FPR'])

    return rocvalues


def sroc(TP, FP, TN, FN):
    """ Based on Littenberg et al. (1993) """

    # Test data based on Table A from Littenberg et al. (1993)
    # TP = [80, 35, 50, 64, 17, 151, 28, 44, 190]
    # FP = [3, 3, 3, 4, 1, 2, 9, 1, 10]
    # TN = [26, 10, 7, 68, 1, 16, 16, 5, 24]
    # FN = [26, 2, 2, 15, 4, 23, 2, 5, 39]

    sroc = {'FPR': [0.0], 'TPR': [0.0]}

    S = []
    D = []

    for tp, fp, tn, fn in zip(TP, FP, TN, FN):
        q = (tp + 0.5) / (tp + fn + 1)
        v = np.log(q / (1 - q))
        p = (fp + 0.5) / (fp + tn + 1)
        u = np.log(p / (1 - p))
        s = v + u
        d = v - u
        S.append(s)
        D.append(d)

    i, b = equally_weighted_least_squares(S, D)

    for tp, fp, tn, fn in zip(TP, FP, TN, FN):
        fpr = (fp + 0.5) / (fp + tn + 1)
        tpr = 1 / (1 + (1 / ((math.e ** (i / (1 - b))) * (fpr / (1 - fpr)) ** ((1 + b) / (1 - b)))))
        sroc['FPR'].append(fpr)
        sroc['TPR'].append(tpr)

    sroc['FPR'].append(1.0)
    sroc['TPR'].append(1.0)

    sroc['FPR'] = sorted(sroc['FPR'])
    sroc['TPR'] = sorted(sroc['TPR'])

    return sroc


def auc(rocvalues):
    tpr_values = rocvalues['TPR']
    fpr_values = rocvalues['FPR']

    return metrics.auc(fpr_values, tpr_values)


def confusionmatrix(y_true, y_pred, y_record_ids, c=0, threshold=0.5, calculation_method='subsample_level'):
    """ Supported calculation_methods: sample_level, subsample_level, """
    assert calculation_method in ['sample_level', 'subsample_level']

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    cls = {'TP': [], 'TN': [], 'FP': [], 'FN': []}

    # If calculation on sample level is required, all predictions are aggregated in advance
    if calculation_method == 'sample_level':
        sample_ids = [rec_id.rsplit('_', maxsplit=1)[0] for rec_id in y_record_ids]
        y_pred_s = {s_id: [] for s_id in sample_ids}
        y_true_s = {s_id: y_t for s_id, y_t in zip(sample_ids, y_true)}
        y_true_s_final = [y_true_s[s_id] for s_id in y_true_s]
        sample_ids_final = list(y_pred_s.keys())

        for y_p, s_id in zip(y_pred, sample_ids):
            y_pred_s[s_id].append(y_p)

        y_width = len(y_true[0])
        y_pred_s_mean = {}

        for s_id in y_pred_s:
            for i in range(y_width):
                y_pred_s_mean[s_id] = list(np.mean(y_pred_s[s_id], axis=0)) ### Mean alle pred

        y_pred_s_final = [y_pred_s_mean[s_id] for s_id in y_pred_s_mean]

        y_true = y_true_s_final
        y_pred = y_pred_s_final
        y_record_ids = sample_ids_final

    # Calculation of TP, FP, TN, FN
    for y_t, y_p, y_r in zip(y_true, y_pred, y_record_ids):
        # Positive
        if y_t[c] == 1.0:

            # Prediction positive
            if y_p[c] > threshold:
                tp += 1
                cls['TP'].append(y_r)

            # Prediction negative
            else:
                fn += 1
                cls['FN'].append(y_r)

        # Negative
        elif y_t[c] == 0.0:

            # Prediction positive
            if y_p[c] > threshold:
                fp += 1
                cls['FP'].append(y_r)

            # Prediction negative
            else:
                tn += 1
                cls['TN'].append(y_r)

        else:
            raise ValueError('True label for class {} has to be 0.0 or 1.0, was {}'.format(c, y_t[c]))

    return tp, fp, tn, fn, cls


def sensitivity(tp, fn):
    return tp / (tp + fn)  # divisor can only be zero when no positives are in the set


def specificity(fp, tn):
    return tn / (tn + fp)  # divisor can only be zero when no negatives are in the set


def truepositiverate(tp, fn):
    return sensitivity(tp, fn)


def falsepositiverate(tn, fp):
    return fp / (fp + tn)  # divisor can only be zero when no negatives are in the set


def accuracy(tp, fp, tn, fn):
    return (tp + tn) / (tp + tn + fp + fn)  # divisor can never be zero


def f1score(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn)  # divisor can only be zero only positives are in the set


def youdensjstatistic(tp, fp, tn, fn):
    return sensitivity(tp, fn) + specificity(fp, tn) - 1


def diagnosticoddsratio(tp, fp, tn, fn):
    """ DOR paper: Glas et al. 2003, DOI: 10.1016/S0895-4356(03)00177-X """

    sens = sensitivity(tp, fn)
    spec = specificity(fp, tn)

    # cases needed to prevent zero division
    if spec == 0.0:
        spec = 0.0001

    if spec == 1.0:
        spec = 0.9999

    if sens == 1.0:
        sens = 0.9999

    return (sens / (1 - sens)) / ((1 - spec) / spec)


def positivepredictivevalue(tp, fp):
    return tp / (tp + fp)  # divisor can only be zero when no positives are in the set


def negativepredictivevalue(tn, fn):
    return tn / (tn + fn)  # divisor can only be zero when no negatives are in the set


def add_metric_to_dictionary(metric_name, dictionary, tp, tn, fp, fn):
    if metric_name == 'sensitivity':
        dictionary[metric_name] = sensitivity(tp, fn)
    elif metric_name == 'specificity':
        dictionary[metric_name] = specificity(fp, tn)
    elif metric_name == 'truepositiverate':
        dictionary[metric_name] = truepositiverate(tp, fn)
    elif metric_name == 'falsepositiverate':
        dictionary[metric_name] = falsepositiverate(tn, fp)
    elif metric_name == 'accuracy':
        dictionary[metric_name] = accuracy(tp, fp, tn, fn)
    elif metric_name == 'f1score':
        dictionary[metric_name] = f1score(tp, fp, fn)
    elif metric_name == 'youdensjstatistic':
        dictionary[metric_name] = youdensjstatistic(tp, fp, tn, fn)
    elif metric_name == 'DOR':
        dictionary[metric_name] = diagnosticoddsratio(tp, fp, tn, fn)
    elif metric_name == 'AUC':
        pass  # AUC included by default
    elif metric_name == 'PPV':
        dictionary[metric_name] = positivepredictivevalue(tp, fp)
    elif metric_name == 'NPV':
        dictionary[metric_name] = negativepredictivevalue(tp, fp)
    else:
        logging.warning('Unknown metric "{}". Skipping this one. Please check spelling.'.format(metric_name))


def calculate_metrics_for_predictions(y_true, y_pred, y_classes, y_record_ids, metrics_to_calculate, calculation_methods, class_names_to_log):
    result = {}

    for c in range(len(y_classes)):
        y_class = y_classes[c]

        if y_class in class_names_to_log:
            result[y_class] = {}

            for calculation_method in calculation_methods:
                tp, fp, tn, fn, cls = confusionmatrix(y_true, y_pred, y_record_ids, c=c, calculation_method=calculation_method)
                rocvalues = roc(y_true, y_pred, y_record_ids, c=c, calculation_method=calculation_method)
                area_under_roc = auc(rocvalues)

                met = {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'AUC': area_under_roc, 'ROC': rocvalues}

                for m in metrics_to_calculate:
                    add_metric_to_dictionary(metric_name=m, dictionary=met, tp=tp, tn=tn, fp=fp, fn=fn)

                result[y_class][calculation_method] = {'metrics': met, 'classification': cls}

    return result


def equally_weighted_least_squares(S, D):
    S = np.array(S)
    D = np.array(D)

    b, i, _, _, _ = linregress(S, D)

    return i, b