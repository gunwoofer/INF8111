import os
import pandas as pd
import numpy as np
import csv

TRAIN_FILE_NAME = "data/training.csv"
TEST_FILE_NAME = "data/test.csv"




def getData(file, type_data='train'):
    with open(file, 'r') as csvFile:
        reader = csv.reader(csvFile)
        features = []
        targets = []
        for i, row in enumerate(reader):
            if i != 0 : # On veut pas la ligne avec les titre de colonne
                line = []
                date = row[0]
                temperature = row[1]
                drew_point = row[2]
                relative_humidity = row[3]
                wind_direction = row[4]
                wind_speed = row[5]
                visibility = row[6]
                visibility_indicator = row[7]
                pressure = row[8]
                hmdx = row[9]
                wind_chill = row[10]
                weather = row[11]
                public_holiday = row[12]
                station_code = row[13]
                if type_data == 'train':
                    withdrawals = row[14]
                    volume = row[15]
                line.append(date)
                line.append(temperature)
                line.append(drew_point)
                line.append(relative_humidity)
                line.append(wind_direction)
                line.append(wind_speed)
                line.append(visibility)
                line.append(visibility_indicator)
                line.append(pressure)
                line.append(hmdx)
                line.append(wind_chill)
                line.append(weather)
                line.append(public_holiday)
                line.append(station_code)
                if type_data == 'train':
                    targets.append(volume)
                features.append(np.array(line))
    return np.array(features), np.array(targets)


def writeCsv(test_date, predict_volume):
    ids = pd.DataFrame(test_date).iloc[:,0]
    result = pd.DataFrame(predict_volume).iloc[:,0]
    finalSubmit = pd.DataFrame(dict(IPERE = ids, D_Mode = result))
    finalSubmit.to_csv('submission/submission.csv', index=False)
res = getData(TRAIN_FILE_NAME)
test = 5