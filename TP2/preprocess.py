import statistics as stats
import numpy as np


import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score


# import nltk
# import math
# from nltk.stem.snowball import SnowballStemmer

# nltk.download("punkt")
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')

# def bigram(tokens):
#     """
#     tokens: a list of strings
#     """
#     # Write your code here
#     # This function returns the list of bigrams
    
#     bigram_list = []
#     for i in range(len(tokens)-1):
#         bigram_list.append(tokens[i] + " " + tokens[i+1])
#     return bigram_list
#     #return list(nltk.bigrams(tokens))
    
# def trigram(tokens):
#     """
#     tokens: a list of strings
#     """
#     # Write your code here
#     # This function returns the list of trigrams
    
#     trigram_list = []
#     for i in range(len(tokens)-2):
#         trigram_list.append(tokens[i] + " " + tokens[i+1] + " " + tokens[i+2])
#     return trigram_list

# class SpaceTokenizer(object):
#     """
#     It tokenizes the tokens that are separated by whitespace (space, tab, newline). 
#     We consider that any tokenization was applied in the text when we use this tokenizer.
    
#     For example: "hello\tworld of\nNLP" is split in ['hello', 'world', 'of', 'NLP']
#     """
    
#     def tokenize(self, text):
#         # Write your code here
#         text.lower()
#         tokens = text.split()
        
#         # Have to return a list of tokens
#         return tokens
        
# class NLTKTokenizer(object):
#     """
#     This tokenizer uses the default function of nltk package (https://www.nltk.org/api/nltk.html) to tokenize the text.
#     """
    
#     def tokenize(self, text):
#         # Write your code here
#         text.lower()
#         tokens = nltk.word_tokenize(text)
        
#         # Have to return a list of tokens
#         return tokens

# class Stemmer():
    
#     def __init__(self):
#         self.stemmer = SnowballStemmer("english", ignore_stopwords=True)
    
#     def stem(self, tokens):
#         """
#         tokens: a list of strings
#         """
        
#         # Write your code here
#         tokens = [self.stemmer.stem(word) for word in tokens]
        
#         # Have to return a list of stems
#         return tokens

# class PreprocessingPipeline:
    
#     def __init__(self, tokenization, stemming):
#         """
#         tokenization: enable or disable tokenization.
#         twitterPreprocessing: enable or disable twitter preprocessing.
#         stemming: enable or disable stemming.
#         """

#         self.tokenizer= NLTKTokenizer() if tokenization else SpaceTokenizer()
#         self.stemmer = Stemmer() if stemming else None
    
#     def preprocess(self, message):
#         """
#         Transform the raw data

#         tokenization: boolean value.
#         twitterPreprocessing: boolean value. Apply the
#         stemming: boolean value.
#         """
#         message = self.tokenizer.tokenize(message)
        
#         if self.stemmer:
#             message = self.stemmer.stem(message)
        
#         return message

# class TFIDFBoW(object):
    
#     def __init__(self, pipeline, bigram=False, trigram=False):
#         """
#         pipelineObj: instance of PreprocesingPipeline
#         bigram: enable or disable bigram
#         trigram: enable or disable trigram
#         """
#         self.pipeline = pipeline
#         self.bigram = bigram
#         self.trigram = trigram
#         self.bow = []

        
#     def fit_transform(self, X):
#         """
#         This method preprocesses the data using the pipeline object, calculates the IDF and TF and 
#         transforms the text in vectors. Vectors are weighted using TF-IDF method.
        
#         X: a list that contains tweet contents
        
#         :return: a list that contains the list of integers
#         """

#         self.bow = []
#         for x in X:
#             datas = self.pipeline.preprocess(x)
#             if self.trigram:
#                 datas += trigram(datas)
#             elif self.bigram:
#                 datas += bigram(datas)
#             for data in datas:
#                 if data not in self.bow:
#                     self.bow.append(data)
#         return self.transform(X)
        
        
#     def transform(self, X):
#         """
#         This method preprocesses the data using the pipeline object and  
#             transforms the text in a list of integer.
        
#         X: a list of tweet
        
#         :return: a list of vectors
#         """        
        
#         # transform the dataset to bag-of-words

#         tweet_datas = []
#         for x in X:
#             datas = self.pipeline.preprocess(x)
#             tweet_datas.append(datas)
        
#         vector = []
#         for tweet in tweet_datas:
#             tweet_vec = []
#             for word in self.bow:
#                 tweet_vec.append(tweet.count(word))
#             vector.append(tweet_vec)
#         clipedBow = np.clip(np.array(vector), a_min = 0, a_max = 1) 
#         dfiVector = np.sum(clipedBow,axis=0)
#         idfiVector = np.zeros_like(dfiVector).astype('float')
#         for i in range(0,dfiVector.shape[0]):
#             idfiVector[i] = math.log(len(X)/ (1+ dfiVector[i]),2)
#         vector = np.array(vector,dtype='f')
#         for i in range(0,vector.shape[0]):
#             vector[i] = vector[i] * idfiVector
#         return vector



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
    # dic_station_code = dict.fromkeys(station_code.tolist(), 0)
    # i = 1
    # for key, value in dic_station_code.items():
    #     dic_station_code[key] = i
    #     i += 1
    # station_code_one_hot = [[0] * i] * len(station_code) 
    # list_one_hot = []
    # for k in range (len(station_code)):
    #     y = dic_station_code[station_code[k]]
    #     one_hot = [0] * i
    #     one_hot[y-1] = 1
    #     # station_code_one_hot[k][y-1] = 1 
    #     list_one_hot.append(one_hot)
    # return np.array(list_one_hot)
    y = [0] * 10
    if station_code == 6184:
        y[0] = 1
    if station_code == 6100:
        y[1] = 1
    if station_code == 6214:
        y[2] = 1
    if station_code == 6078:
        y[3] = 1
    if station_code == 6221:
        y[4] = 1
    if station_code == 6070:
        y[5] = 1
    if station_code == 6026:
        y[6] = 1
    if station_code == 6015:
        y[7] = 1
    if station_code == 6136:
        y[8] = 1
    if station_code == 6012:
        y[9] = 1
    # if station_code == 6036:
    #     y[10] = 1
    # if station_code == 6216:
    #     y[11] = 1
    # if station_code == 6034:
    #     y[12] = 1
    # if station_code == 6206:
    #     y[13] = 1
    # if station_code == 6009:
    #     y[14] = 1
    # if station_code == 6173:
    #     y[15] = 1
    # if station_code == 6211:
    #     y[16] = 1
    # if station_code == 6050:
    #     y[17] = 1
    # if station_code == 6067:
    #     y[18] = 1
    # if station_code == 6064:
    #     y[19] = 1
    # if station_code == 6052:
    #     y[20] = 1
    # if station_code == 6227:
    #     y[21] == 1
    # if station_code == 6248:
    #     y[22] = 1
    # if station_code == 6154:
    #     y[23] = 1
    # if station_code == 6748:
    #     y[24] = 1
    # if station_code == 6190:
    #     y[25] = 1
    # if station_code == 6155:
    #     y[26] = 1
    # if station_code == 6501:
    #     y[27] = 1
    # if station_code == 6073:
    #     y[28] = 1
    # if station_code == 6143:
    #     y[29] = 1
    # if station_code == 6250:
    #     y[30] = 1
    # if station_code == 6411:
    #     y[31] = 1
    # if station_code == 6114:
    #     y[32] = 1
    # if station_code == 6223:
    #     y[34] = 1
    # if station_code == 6729:
    #     y[35] = 1
    # if station_code == 6193:
    #     y[36] = 1
    # if station_code == 6165:
    #     y[37] = 1
    # if station_code == 6906:
    #     y[38] = 1
    # if station_code == 6046:
    #     y[39] = 1
    # if station_code == 6083:
    #     y[40] = 1
    # if station_code == 6199:
    #     y[41] = 1
    # if station_code == 6148:
    #     y[42] = 1
    # if station_code == 6023:
    #     y[43] = 1
    # if station_code == 6213:
    #     y[44] = 1
    # if station_code == 6209:
    #     y[45] = 1
    # if station_code == 6086:
    #     y[46] = 1
    # if station_code == 6063:
    #     y[47] = 1
    # if station_code == 6418:
    #     y[48] = 1
    # if station_code == 6194:
    #     y[49] = 1

    return y


def categorizeTemperatures(temperatures):
    hotvec = []
    for temp in temperatures:
        y = [0] * 14
        if temp < 19 :
            y[0] = 1
        elif temp >= 19 and temp < 20:
            y[1] = 1
        elif temp >= 20 and temp < 21:
            y[2] = 1
        elif temp >= 21 and temp < 22:
            y[3] = 1
        elif temp >= 22 and temp < 23:
            y[4] = 1
        elif temp >= 23 and temp < 24:
            y[5] = 1
        elif temp >= 24 and temp < 25:
            y[6] = 1
        elif temp >= 25 and temp < 26:
            y[7] = 1
        elif temp >= 26 and temp < 27:
            y[8] = 1
        elif temp >= 27 and temp < 28:
            y[9] = 1
        elif temp >= 28 and temp < 29:
            y[10] = 1
        elif temp >= 29 and temp < 30:
            y[11] = 1
        elif temp >= 30 and temp < 31:
            y[12] = 1
        else :
            y[13] = 1
        hotvec.append(y)

    return np.array(hotvec)



def makeHotVector(vectors):
    b = np.zeros((vectors.size, int(vectors.max()+1)))
    b[np.arange(vectors.size),vectors] = 1
    return b

def findFreqStation(stations, labels):
    pass


    
# def update_median(array):
#     col_median = np.nanmean(array, axis=0)
#     inds = np.where(np.isnan(array))
#     array[inds] = np.take(col_median, inds[1])
#     return array

# def standardize(x): 
#     mean_px = X_train.mean()
#     std_px = X_train.std()
#     return (x-mean_px)/std_px

# def reformat_data(x):
#     for i in range (x.shape[0]):
#         for j in range (x.shape[1]):
#             if x[i][j] == '':
#                 x[i][j] = np.nan
#             else:
#                 x[i][j] = float(str(x[i][j]).replace(',', '.'))
#     return x

def get_meteo2(meteo):
    y = [0] * 8
    if "pluie" in meteo:
        y[0] = 1
    if "neige" in meteo:
        y[1] = 1
    if "Brouillard" in meteo or "brouillard" in meteo:
        y[2] = 1
    if "Bruine" in meteo or "bruine" in meteo:
        y[3] = 1
    if "Nuageux" in meteo or "nuageux" in meteo:
        y[4] = 1
    if "Orages" in meteo or "orages" in meteo:
        y[5] = 1
    if "gag" in meteo:
        y[6] = 1
    if "ND" in meteo:
        y[7] = 1
    return y


def categorizeDrewPoint(drewPoints):
    hotvec = []
    for temp in drewPoints:
        y = [0] * 29 # de -5 à 23
        if temp < -5 :
            y[0] = 1
        elif temp >= -5 and temp < -4:
            y[1] = 1
        elif temp >= -4 and temp < -3:
            y[2] = 1
        elif temp >= -3 and temp < -2:
            y[3] = 1
        elif temp >= -2 and temp < -1:
            y[4] = 1
        elif temp >= -1 and temp < 0:
            y[5] = 1
        elif temp >= 0 and temp < 1:
            y[6] = 1
        elif temp >= 1 and temp < 2:
            y[7] = 1
        elif temp >= 2 and temp < 3:
            y[8] = 1
        elif temp >= 3 and temp < 4:
            y[9] = 1
        elif temp >= 4 and temp < 5:
            y[10] = 1
        elif temp >= 5 and temp < 6:
            y[11] = 1
        elif temp >= 6 and temp < 7:
            y[12] = 1
        elif temp >= 7 and temp < 8:
            y[13] = 1
        elif temp >= 8 and temp < 9:
            y[14] = 1
        elif temp >= 9 and temp < 10:
            y[15] = 1
        elif temp >= 10 and temp < 11:
            y[16] = 1
        elif temp >= 11 and temp < 12:
            y[17] = 1
        elif temp >= 12 and temp < 13:
            y[18] = 1
        elif temp >= 13 and temp < 14:
            y[19] = 1
        elif temp >= 14 and temp < 15:
            y[20] = 1
        elif temp >= 15 and temp < 16:
            y[21] = 1
        elif temp >= 16 and temp < 17:
            y[22] = 1
        elif temp >= 17 and temp < 18:
            y[23] = 1
        elif temp >= 18 and temp < 19:
            y[24] = 1
        elif temp >= 19 and temp < 20:
            y[25] = 1
        elif temp >= 20 and temp < 21:
            y[26] = 1
        elif temp >= 21 and temp < 22:
            y[27] = 1
        elif temp >= 22 and temp < 23:
            y[28] = 1
        else :
            y[0] = 1
        hotvec.append(y)

    return np.array(hotvec)

def get_month(months):
    result = []
    for month in months:
        y = [0] * 3
        if month == 9:
            y[0] = 1
        elif month == 10:
            y[1] = 1
        elif month == 11:
            y[2] = 2
        result.append(y)
    return result