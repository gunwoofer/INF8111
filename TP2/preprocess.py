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


def get_meteo(meteos):
    pipeline = PreprocessingPipeline(True, True)
    bow = TFIDFBoW(pipeline, True, True)
    result = bow.fit_transform(meteos)
    return result

def categorizeTemperatures(temperatures):
    hotvec = []
    for temp in temperatures:
        if temp < 20 :
            hotvec.append([1,0,0])
        elif temp > 20 and temp < 29.9:
            hotvec.append([0,1,0])
        else:
            hotvec.append([0,0,1])
    return np.array(hotvec)



def makeHotVector(vectors):
    b = np.zeros((vectors.size, int(vectors.max()+1)))
    b[np.arange(vectors.size),vectors] = 1
    return b

