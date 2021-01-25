"""
 Authors: Nils Gumpfer, Joshua Prim
 Version: 0.1

 Utils for extraction

 Copyright 2020 The Authors. All Rights Reserved.
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def rotate_origin_only(x, y, radians):
    """
        rotates one point around the origin
    :param x: point X-axis
    :param y: point Y-axis
    :param radians:
    :return: rotated point
    """
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy


def move_along_the_axis(lead_list, index=0):
    """
        move along the axis
    :param lead_list:
    :param index: point for orientation of the origin
    :return: new lead list
    """
    tmp = 0
    for (x, y), i in zip(lead_list, range(len(lead_list))):
        if x < index:
            tmp = i

    x0, y0 = lead_list[tmp]
    tmp = [(x, y - y0) for x, y in lead_list]

    delta = index - tmp[0][0]

    new_lead_list = []
    for i in tmp:
        new_lead_list.append((i[0] + delta, i[1]))

    return new_lead_list


def get_y_value(x, list_x, list_y):
    """
        returns the Y value of a transferred X value based on the transferred list of values.
    :param x: x Value
    :param list_x: list of X-values
    :param list_y: list of Y-values
    :return: y value
    """
    x_value, index = find_value1_value2(list_x, x)
    y_value = [list_y[index - 1], list_y[index]]

    m = (y_value[0] - y_value[1]) / (x_value[0] - x_value[1])
    b = (x_value[0] * y_value[1] - x_value[1] * y_value[0]) / (x_value[0] - x_value[1])
    y = m * x + b
    return y


def find_value1_value2(liste, value):
    """
        finds the next smaller and larger value in a list for a passed value.
    :param liste: list to be searched
    :param value: value
    :return: lower value, upper value and index
    """
    tmp_array = np.array(liste)
    index = np.where(tmp_array > value)[0][0]

    value1 = 0 if index == 0 else liste[index - 1]
    value2 = liste[index]

    return [value1, value2], index


def scale_values_based_on_eich_peak(lead_list, gamma=0.5):
    """
        scale values on the Y-axis
        :param lead_list: list of the value
        :param gamma: scaling factor
        :return: rescaled list
    """
    new_lead_list = []
    for xy_pair in lead_list:
        new_y_value = xy_pair[1] * gamma
        new_lead_list.append([xy_pair[0], new_y_value])
    return new_lead_list


def plot_leads(lead, plot_path=None, plot_name='plot'):
    """
        visualizes the lead in a plot
    :param lead: ecg lead for visualization
    :param plot_path: path where the plot should be saved if set
    :param plot_name: name of the plot to be saved
    """
    df = pd.DataFrame(lead, columns=['Y', 'extracted time series'])
    df['extracted time series'] = pd.to_numeric(df['extracted time series'])
    df['Y'] = pd.to_numeric(df['Y'])

    df.plot(kind='line', x='Y', y=['extracted time series'], figsize=(20, 10), legend=False)
    # df.plot(kind='line', x='Y', y=['extracted time series'], figsize=(28, 2), legend=False)

    if plot_path is not None:
        plt.savefig(plot_path+str(plot_name)+'.png')

    plt.show()


def create_measurement_points(lead_list, number_of_points):
    """
        creates measuring points at equidistant intervals from each other
    :param lead_list: list with lead
    :param number_of_points: number of measuring points to be created
    :return: list with measuring points
    """
    measurement_points = []
    max_element = lead_list[-1][0]
    distance = max_element / number_of_points

    x_values = [x[0] for x in lead_list]
    y_values = [y[1] for y in lead_list]

    for i in range(0, number_of_points):
        measurement_points.append(get_y_value(i * distance, x_values, y_values))

    measurement_points = [int(y) for y in measurement_points]
    return measurement_points


def calc_stddev(df, window_size=124):
    """
        calculates the average using the standard deviation
        Note: the procedure is only executed on the first lead
    :param df: DataFrame which is scanned
    :param window_size: size of the sliding window
    :return: average
    """
    min_dev_sum = np.Inf
    avg = []
    for i in range(0, len(df)-window_size):
        df_tmp = df.loc[i:i+window_size]

        if sum(df_tmp.std()) < min_dev_sum:
            min_dev_sum = sum(df_tmp.std())
            avg = df_tmp.mean()
    return avg


def adjust_leads_baseline(df_leads):
    stddev_tmp = calc_stddev(df_leads)

    for column in df_leads.columns:
        df_leads[column] = df_leads[column] - stddev_tmp[column]

    return df_leads
