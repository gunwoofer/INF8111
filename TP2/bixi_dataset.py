import os
import pandas as pd
import numpy as np
import csv
import preprocess as pp
import datetime

TRAIN_FILE_NAME = "data/training.csv"
TEST_FILE_NAME = "data/test.csv"




def getData(file, type_data='train'):
    print('obtenir les données de : ' + file)
    with open(file, 'r') as csvFile:
        reader = csv.reader(csvFile)
        features = []
        targets = []
        ids = []
        for i, row in enumerate(reader):
            if i != 0 : # On veut pas la ligne avec les titre de colonne
                line = []
                ids.append(row[0].replace(' ', '_') + "_" + row[13])
                date = row[0]
                day = datetime.datetime.strptime(row[0], "%Y-%m-%d  %H:%M").weekday()
                temperature = pp.get_temperature(row[1])
                month = int(date.split()[0].split('-')[1])
                hour = int(date.split()[1].split(':')[0])
                drew_point = pp.get_drew_point(row[2])
                relative_humidity = pp.get_relative_humidity(row[3])
                wind_direction = pp.get_wind_direction(row[4])
                wind_speed = pp.get_wind_speed(row[5])
                visibility = pp.get_visibility(row[6])
                visibility_indicator = row[7]
                pressure = pp.get_pressure(row[8])
                hmdx = row[9]
                wind_chill = row[10]
                weather = pp.get_meteo2(row[11])
                public_holiday = pp.get_public_holiday(row[12])
                station_code = pp.get_station_code(row[13])
                if type_data == 'train':
                    volume = row[15]
                #line.append(date)
                line.append(temperature)
                # line.append(relative_humidity)
                # line.append(wind_direction)
                # line.append(wind_speed)
                # line.append(visibility)
                # line.append(visibility_indicator)
                # line.append(pressure)
                # line.append(hmdx)
                # line.append(wind_chill)
                # line.append(weather)
                line.append(hour)
                line.append(month)
                line.append(drew_point)
                line.append(day)
                # line.append(station_code)

                line.append(public_holiday)
                for meteo in weather:
                    line.append(meteo)

                # for station in station_code:
                #     line.append(station)

                # if month == 9 or month == 10 or month == 11:
                features.append(line)
                if type_data == 'train':
                    targets.append(volume)
    print('nb feature : ' + str(len(features)))
    return (np.array(features).astype('float'), np.array(targets).astype('uint8'), ids)
    


def writeCsv(test_date, predict_volume):
    ids = pd.DataFrame(test_date).iloc[:,0]
    result = pd.DataFrame(predict_volume).iloc[:,0]
    ids.drop_duplicates()
    finalSubmit = pd.DataFrame(dict(id = ids, volume = result))
    finalSubmit.to_csv('submission/submission-val.csv', index=False)



