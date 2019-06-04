import statistics as stats
import numpy as np

def get_temperature(temperature):
    if temperature == '':
        return -1000 # code erreur
    else:
        return float(temperature.replace(',','.'))

def get_drew_point(drew_point):
    if drew_point == '':
        return -1000 # code erreur
    else:
        return float(drew_point.replace(',','.'))

def get_relative_humidity(humidity):
    if humidity == '':
        return -1000 # code erreur
    else:
        return float(humidity.replace(',','.'))

def get_wind_direction(wind_direction):
    if wind_direction == '':
        return -1000
    else:
        return float(wind_direction.replace(',','.'))

def get_wind_speed(wind_speed):
    if wind_speed == '':
        return -1000
    else:
        return float(wind_speed.replace(',','.'))


def get_visibility(visibility):
    if visibility == '':
        return -1000
    else:
        return float(visibility.replace(',','.'))


def get_pressure(pressure):
    if pressure == '':
        return -1000
    else:
        return float(pressure.replace(',','.'))

def get_public_holiday(public_holiday):
    if public_holiday == '':
        return -1000
    else:
        return int(public_holiday.replace(',','.'))


# Tableau de station de code
def get_station_code(station_code):
    dic_station_code = dict.fromkeys(station_code.tolist(), 0)
    i = 1
    for key, value in dic_station_code.items():
        dic_station_code[key] = i
        i += 1
    station_code_one_hot = [[0] * i] * len(station_code) 
    list_one_hot = []
    for k in range (len(station_code)):
        y = dic_station_code[station_code[k]]
        one_hot = [0] * i
        one_hot[y-1] = 1
        # station_code_one_hot[k][y-1] = 1 
        list_one_hot.append(one_hot)
    return np.array(list_one_hot)

def update_median(array):
    liste = []
    for val in array:
        if val != -1000:
            liste.append(val)
    mediane = stats.median(liste)
    for i in range(len(array)):
        if array[i] == -1000:
            array[i] = mediane
    return array