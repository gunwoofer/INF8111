import statistics as stats
import numpy as np

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
    y = [0] * 20
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
    if station_code == 6036:
        y[10] = 1
    if station_code == 6216:
        y[11] = 1
    if station_code == 6034:
        y[12] = 1
    if station_code == 6206:
        y[13] = 1
    if station_code == 6009:
        y[14] = 1
    if station_code == 6173:
        y[15] = 1
    if station_code == 6211:
        y[16] = 1
    if station_code == 6050:
        y[17] = 1
    if station_code == 6067:
        y[18] = 1
    if station_code == 6064:
        y[19] = 1
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


    
def update_median(array):
    col_median = np.nanmean(array, axis=0)
    inds = np.where(np.isnan(array))
    array[inds] = np.take(col_median, inds[1])
    return array

def standardize(x): 
    mean_px = X_train.mean()
    std_px = X_train.std()
    return (x-mean_px)/std_px

def reformat_data(x):
    for i in range (x.shape[0]):
        for j in range (x.shape[1]):
            if x[i][j] == '':
                x[i][j] = np.nan
            else:
                x[i][j] = float(str(x[i][j]).replace(',', '.'))
    return x

def get_meteo2(meteo):
    y = [0] * 7
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
    return y