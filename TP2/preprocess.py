import statistics as stats

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


def get_station_code(station_code):
    if station_code == '':
        return -1000
    else:
        return int(station_code.replace(',','.'))

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