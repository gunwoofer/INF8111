#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'INF8111\TP1'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # 1 - Overview
# 
# Twitter is a mix of social network and microblogging. In this network, people post information and communicate among themselves through messages, called tweets, that can contain up to 280 characters. In this assignment, *we will implement a prototype that can detect if an airline company is positively or negatively mentioned in a tweet*. 
# 
# 
#%% [markdown]
# # 2 - Sentiment Analysis Model (13 points)
# 
# In the literature, the task of extracting the sentiment of a text is called *sentiment analysis*. We will implement a bag-of-words (BoW) model for this task.
# 
# ## 2.1 -  Setup
# 
# Please run the code below to install the packages needed for this assignment.

#%%
# If you want, you can use anaconda and install after nltk library
# pip3 install --user numpy
# pip3 install --user sklearn
# pip3 install --user scipy
# pip3 install --user nltk


#python
import nltk
from math import log
nltk.download("punkt")
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('wordnet')

#%% [markdown]
# ## 2.2 - Dataset
# 
# Please download the zip file in the following url: https://drive.google.com/file/d/1iGmESwPXpO3sIZFGOCrysxJ27AHdly-Y/view?usp=sharing
# 
# In this zip file, there are 2 files:
# 1. train.tsv: training dataset
# 2. dev.tsv: validation dataset
# 
# Each line of the files has the following information about a tweet: *tweet id*, *user id*, *label* and *message text*.
# 
# There are three labels in the dataset: *negative*, *neutral* and *positive*. We represent each one of these labels as 0, 1 and 2 respectively.
# 
# In the code above read the training and validation datasets.

#%%
import codecs
import re

def load_dataset(path):
    dtFile = codecs.open(path, 'r')
    
    x = []
    y = []
    
    for l in dtFile:
        sid, uid, label,text = re.split(r"\s+", l, maxsplit=3)
        
        text = text.strip()
        
        # Remove not available
        if text == "Not Available":
            continue
        
        x.append(text)
        
        if label == "negative": 
            y.append(0)
        elif label == "neutral": 
            y.append(1)
        elif label == "positive": 
            y.append(2)
        
    assert len(x) == len(y)
            
    return x,y
            

# Path of training dataset
trainingPath="sentiment_analysis/train_data.tsv"

# Path of validation dataset
validationPath="sentiment_analysis/dev_data.tsv"

training_X, training_Y = load_dataset(trainingPath)
validation_X, validation_Y = load_dataset(validationPath)

#%% [markdown]
# ## 2.3 - Preprocessing
# 
# Preprocessing is a crucial task in data mining. This task clean and transform the raw data in a format that can better suit data analysis and machine learning techniques. In natural language processing (NLP), *tokenization* and *stemming* are two well known preprocessing steps. Besides these two steps, we will implement an additional step that is designed exclusively for the twitter domain.
# 
# ### 2.3.1 - Tokenization
# 
# In this preprocessing step, a *tokenizer* is responsible for breaking a text in a sequence of tokens (words, symbols, and punctuations). For instance, the sentence *"It's the student's notebook."* can be split into the following list of tokens: ['It', "'s", 'the', 'student', "'s", 'notebook', '.'].
# 
# 
# #### 2.3.1.1 - Question 1 (0.5 point) 
# 
# Implement the SpaceTokenizer and NLTKTokenizer tokenizers: 
# - **SpaceTokenizer** tokenizes the tokens that are separated by whitespace (space, tab, newline). This is a naive tokenizer.
# - **NLTKTokenizer** uses the default method of the nltk package (https://www.nltk.org/api/nltk.html) to tokenize the text.
# 
# **All tokenizers have to lowercase the tokens.**

#%%


class SpaceTokenizer(object):
    """
    It tokenizes the tokens that are separated by whitespace (space, tab, newline). 
    We consider that any tokenization was applied in the text when we use this tokenizer.
    
    For example: "hello\tworld of\nNLP" is split in ['hello', 'world', 'of', 'NLP']
    """
    
    def tokenize(self, text):
        # Write your code here
        
        tokens = text.split()
        
        # Have to return a list of tokens
        return tokens
        
class NLTKTokenizer(object):
    """
    This tokenizer uses the default function of nltk package (https://www.nltk.org/api/nltk.html) to tokenize the text.
    """
    
    def tokenize(self, text):
        # Write your code here
        
        tokens = nltk.tokenize.word_tokenize(text)
        
        # Have to return a list of tokens
        return tokens

        

#%% [markdown]
# ### 2.3.2 - Stemming
# 
# In the tweets *"I should have bought a new shoes today"* and *"I spent too much money buying games"*, the words *"buy"* and *"bought"* represent basically the same concept. Considering both words as different can unnecessarily increase the dimensionality of the problem and can negatively impact the performance of simple models. Therefore, a unique form (e.g., the root buy) can represent both words. The process to convert words with the same stem (word reduction that keeps word prefixes) to a standard form is called *stemming*.
# 
# #### 2.3.2.1 - Question 2 (0.5 point) 
# 
# Retrieve the stems of the tokens using the attribute *stemmer* from the class *Stemmer*.

#%%
from nltk.stem.snowball import SnowballStemmer

class Stemmer(object):
    
    def __init__(self):
        self.stemmer = SnowballStemmer("english", ignore_stopwords=True)
    
    def stem(self, tokens):
        """
        tokens: a list of strings
        """
        # Write your code here
        tokens_stem = [self.stemmer.stem(token) for token in tokens]
        
        # Have to return a list of stems
        return tokens_stem
        

#%% [markdown]
# ### 2.3.3 - Twitter preprocessing
# 
# Sometimes only applying the default NLP preprocessing steps is not enough. Data for certain domains can have peculiar characteristics which requires specific preprocessing steps to remove the noise and create a more suitable format for the models. 
# 
# In NLP, methods store a set of words, called dictionary, and all the words out of the dictionary are considered as unknown. In this assignment, the feature space dimensionality of a model is directly related to the number of words in the dictionary. Since high-dimensional spaces can suffer from the curse of dimensionality, our goal is to create preprocessing steps that decrease vocabulary size.  
# 
# #### 2.3.3.1 - Question 3 (2.0 points)
# 
# Briefly explain and implement at least two preprocessing steps that reduce the dictionary size (number of unique words). These preprocessing steps must be related to the specific characteristic of the Twitter data. Therefore, for instance, the stop words removal will not be accepted as a preprocessing step.

#%%
class TwitterPreprocessing2(object):
    
    def preprocess(self, tweet):
        """
        tweet: original tweet
        """
        # Write your preprocessing steps here.
        tweet = self.preprocess1(self.preprocess2(tweet))
    
        # return the preprocessed twitter
        return tweet
        
    #the chunking process to classify word in pre defined category, as verbs, nouns, adjective, topic...
    def preprocess1(self, tweet):
        result = nltk.pos_tag(nltk.word_tokenize(tweet))
        reg_exp = "NP: {<DT>?<JJ>*<NN>}"
        rp = nltk.RegexpParser(reg_exp)
        result = rp.parse(result)
        return result.pprint()
    
    #Lemmatization process get the correct base form of words
    def preprocess2(self, tweet):
        lemmatizer = nltk.WordNetLemmatizer()
        input_str=nltk.word_tokenize(tweet)
        output_str =""
        for word in input_str:
            output_str += lemmatizer.lemmatize(word) + " "
        return output_str
    
class TwitterPreprocessing(object):
    
    def __init__(self):
        self.stemmer = Stemmer()
        self.tokenizer = SpaceTokenizer()
        self.emoji = [";)", ":)", ":(", ";(", ":3", "xd", ":D", ":p"]
        
        
    def preprocess(self, tweet):
        """
        tweet: original tweet
        """
        
        new_tweet = ""
        words = tweet.split()
        for word in words:
            # Remove smileys
            if (word not in self.emoji 
                and not word.startswith("@") 
                and not word.startswith("http") 
                and nltk.pos_tag([word])[0][1] != 'DT'):
                new_tweet += word + " "
                
        # return the preprocessed twitter
        return new_tweet

#%% [markdown]
# ### 2.3.3  Pipeline
# 
# The pipeline is sequence of preprocessing steps that transform the raw data to a format that is suitable for your problem. We implement the class *PreprocessingPipeline* that apply the tokenizer, twitter preprocessing and stemer to the text.
# 
# **Feel free to change the preprocessing order.**

#%%
class PreprocessingPipeline:
    
    def __init__(self, tokenization, twitterPreprocessing, stemming):
        """
        tokenization: enable or disable tokenization.
        twitterPreprocessing: enable or disable twitter preprocessing.
        stemming: enable or disable stemming.
        """

        self.tokenizer= NLTKTokenizer() if tokenization else SpaceTokenizer()
        self.twitterPreprocesser = TwitterPreprocessing() if twitterPreprocessing else None
        self.stemmer = Stemmer() if stemming else None
    
    def preprocess(self, tweet):
        """
        Transform the raw data

        tokenization: boolean value.
        twitterPreprocessing: boolean value. Apply the
        stemming: boolean value.
        """
        if self.twitterPreprocesser:
            tweet = self.twitterPreprocesser.preprocess(tweet)
        
        tokens = self.tokenizer.tokenize(tweet)

        if self.stemmer:
            tokens = self.stemmer.stem(tokens)
            
        

        return tokens
    

#%% [markdown]
# ## 2.4 N-grams
# 
# An n-gram is a contiguous sequence of *n* tokens from a text. Thus, for instance,the sequence *"bye as"* and *"walked through"* are example of 2-grams from the sentence *"He said bye as he walked through the door ."*. 1-gram, 2-gram and 3-gram are, respectively, called unigram, bigram and trigram. We list all the possible unigram, bigram and trigram from the *"He said bye as he walked through the door ."*:
# 
# - Unigram: ["He", "said", "bye", "as", "he", "walked", "through", "the", "door", "."]
# - Bigram: ["He said", "said bye", "bye as", "as he", "he walked", "walked through", "through the", "the door", "door ."] 
# - Trigram: ["He said bye", "said bye as", "bye as he", "as he walked", "he walked through", "walked through the", "through the door", "the door ."] 
# 
# 
# ### 2.4.1 - Question 4 (1 point)
# 
# Implement bigram and trigram.
# 
# **For this exercise, you cannot use any external python library (e.g., scikit-learn).**

#%%
def bigram(tokens):
    result = []
    for i in range (len(tokens) - 1):
        result.append(tokens[i] + " "+ tokens[i+1])
    return result
    
def trigram(tokens):
    result = []
    for i in range (len(tokens) - 2):
        result.append(tokens[i] + " " + tokens[i+1] + " " + tokens[i + 2])
    return result
    

#%% [markdown]
# ## 2.5 Bag-of-words
# 
# Logistic regression, SVM and other well-known models only accept inputs that have the same size. However, there are some data types whose sizes are not fixed, for instance, a text can have an unlimited number of words. Imagine that we retrieve two tweets: ”Board games are much better than video games” and ”Pandemic is an awesome game!”. These sentences are respectively named as Sentence 1 and 2. Table below depicts how we could represent both sentences using a fixed representation.
# 
# |            | an | are | ! | pandemic | awesome | better | games | than | video | much | board | is | game |
# |------------|----|-----|---|----------|---------|--------|-------|------|-------|------|-------|----|------|
# | Sentence 1 | 0  | 1   | 0 | 0        | 0       | 1      | 2     | 1    | 1     | 1    | 1     | 0  | 0    |
# | Sentence 2 | 1  | 0   | 0 | 1        | 1       | 0      | 0     | 0    | 0     | 0    | 0     | 1  | 1    |
# 
# Each column of this table 2.1 represents one of 13 vocabulary words, whereas the rows contains the word
# frequencies in each sentence. For instance, the cell in row 1 and column 7 has the value 2
# because the word games occurs twice in Sentence 1. Since the rows have always 13 values, we
# could use those vectors to represent the Sentences 1 and 2. The table above illustrates a technique called bag-of-words. Bag-of-words represents a document as a vector whose dimensions are equal to the number of times that vocabulary words appeared in the document. Thus, each token will be related to a dimension, i.e., an integer.
# 
# ### 2.5.1 - Question 5 (2 points)
# 
# Implement the bag-of-words model that weights the vector with the absolute word frequency.
# 
# **For this exercise, you cannot use any external python library (e.g., scikit-learn). However, if you have a problem with memory size, you can use the class scipy.sparse.csr_matrix (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html)
# **

#%%
class CountBoW(object):
    
    def __init__(self, pipeline, bigram=False, trigram=False):
        """
        pipelineObj: instance of PreprocesingPipeline
        bigram: enable or disable bigram
        trigram: enable or disable trigram
        """
        self.pipeline = pipeline
        self.bigram = bigram
        self.trigram = trigram
        
    def fit_transform(self, X):
        """
        This method preprocesses the data using the pipeline object, relates each unigram, bigram or trigram to a specific integer and  
        transforms the text in a vector. Vectors are weighted using the token frequencies in the sentence.
        
        X: a list that contains tweet contents
        
        :return: a list of vectors
        """   
        vector_text = []
        vectors = []
        for x in X:
            datas = self.pipeline.preprocess(x)
            if self.bigram:
                datas = bigram(datas)
            elif self.trigram:
                datas = trigram(datas)
            for data in datas:
                if data not in vector_text:
                    vector_text.append(data)
        
        for i in range(len(X)):
            datas = self.pipeline.preprocess(X[i])
            vector = []
            if self.bigram:
                datas = bigram(datas)
            elif self.trigram:
                datas = trigram(datas) 
            for text in vector_text:
                vector.append(datas.count(text))
            vectors.append(vector)
        return vectors
        
    def transform(self, X):
        """
        This method preprocesses the data using the pipeline object and  transforms the text in a list of integer.
        Vectors are weighted using the token frequencies in the sentence.
        
        X: a list of vectors
        
        :return: a list of vectors
        """        
        vector_text = []
        vectors = []
        for x in X:
            datas = self.pipeline.preprocess(x)
            for data in datas:
                if data not in vector_text:
                    vector_text.append(data)
        vectors.append(vector_text)
        for i in range(len(X)):
            datas = self.pipeline.preprocess(X[i])
            vector = []
            if self.bigram:
                datas = bigram(datas)
            elif self.trigram:
                datas = trigram(datas) 
            for text in vector_text:
                vector.append(datas.count(text))
            vectors.append(vector)
        return vectors 
        
    

#%% [markdown]
# ### 2.5.2 - TF-IDF
# 
# Using raw frequency in the bag-of-words can be problematic. The word frequency distribution
# is skewed - only a few words have high frequencies in a document. Consequently, the
# weight of these words will be much bigger than the other ones which can give them more
# impact on some tasks, like similarity comparison. Besides that, a set of words (including
# those with high frequency) appears in most of the documents and, therefore, they do not
# help to discriminate documents. For instance, the word *of* appears in a significant
# part of tweets. Thus, having the word *of* does not make
# documents more or less similar. However, the word *terrible* is rarer and documents that
# have this word are more likely to be negative. TF-IDF is a technique that overcomes the word frequency disadvantages.
# 
# TF-IDF weights the vector using inverse document frequency (IDF) and word frequency, called term frequency (TF).
# TF is the local information about how important is a word to a specific document.  IDF measures the discrimination level of the words in a dataset.  Common words in a domain are not helpful to discriminate documents since most of them contain these terms. So, to reduce their relevance in the documents, these words should have low weights in the vectors . 
# The following equation calculates the word IDF:
# \begin{equation}
# 	idf_i = \log\left( \frac{N}{df_i} \right),
# \end{equation}
# where $N$ is the number of documents in the dataset, $df_i$ is the number of documents that contain a word $i$.
# The new weight $w_{ij}$ of a word $i$ in a document $j$ using TF-IDF is computed as:
# \begin{equation}
# 	w_{ij} = tf_{ij} \times idf_i,
# \end{equation}
# where $tf_{ij}$ is the term frequency of word $i$ in the document $j$.
# 
# 
# 
# 
# ### 2.5.2.1 - Question 6 (3 points)
# 
# Implement a bag-of-words model that weights the vector using TF-IDF.
# 
# **For this exercise, you cannot use any external python library (e.g., scikit-learn). However, if you have a problem with memory size, you can use the class scipy.sparse.csr_matrix (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html)**

#%%
class TFIDFBoW(object):
    
    def __init__(self, pipeline, bigram=False, trigram=False):
        """
        pipelineObj: instance of PreprocesingPipeline
        bigram: enable or disable bigram
        trigram: enable or disable trigram
        """
        self.pipeline = pipeline
        self.bigram = bigram
        self.trigram = trigram
        
    def find_n_elem(self, word, tweet_datas):
        return sum(1 for elem in tweet_datas if word in elem)
    
    def find_idf(self, df, N):
        idf = [N] * len(df) 
        idf = [x/y for x, y in zip(idf, df)]
        idf = [log(x) for x in idf]
        return idf
    
    def fit_transform(self, X):
        """
        This method preprocesses the data using the pipeline object, calculates the IDF and TF and 
        transforms the text in vectors. Vectors are weighted using TF-IDF method.
        
        X: a list that contains tweet contents
        
        :return: a list that contains the list of integers
        """

        vector_text = []
        vectors = []
        tweet_datas = []
        for x in X:
            datas = self.pipeline.preprocess(x)
            if self.bigram:
                datas = bigram(datas)
            elif self.trigram:
                datas = trigram(datas)
            tweet_datas.append(datas)
            for data in datas:
                if data not in vector_text:
                    vector_text.append(data)
        vectors.append(vector_text)
        
        N = len(X)
        df = []
        for text in vector_text:
            df.append(self.find_n_elem(text, tweet_datas))
        idf = self.find_idf(df, N)
        countBow = CountBoW(self.pipeline, self.bigram, self.trigram)
        vectors_count =countBow.fit_transform(X)
        vectors_count.pop(0)
        tf = []
        for vector in vectors_count:
            tf.append([x/sum(vector) for x in vector])
            
        #calcul de w
        print(tf)
        print("idf : ")
        print(idf)
        w = []
        for elem in tf:
            w.append( [x * y for x, y in zip(elem, idf)])
        return w
        
    def transform(self, X):
        """
        This method preprocesses the data using the pipeline object and  
            transforms the text in a list of integer.
        
        X: a list of vectors
        
        :return: a list of vectors
        """        
        
        # transform the dataset to bag-of-words
        vector_text = []
        vectors = []
        tweet_datas = []
        for x in X:
            datas = self.pipeline.preprocess(x)
            tweet_datas.append(datas)
            for data in datas:
                if data not in vector_text:
                    vector_text.append(data)
        vectors.append(vector_text)
        
        N = len(X)
        df = []
        for text in vector_text:
            df.append(self.find_n_elem(text, tweet_datas))
        idf = self.find_idf(df, N)
        countBow = CountBoW(self.pipeline, self.bigram, self.trigram)
        vectors_count =countBow.fit_transform(X)
        vectors_count.pop(0)
        tf = []
        for vector in vectors_count:
            tf.append([x/sum(vector) for x in vector])
            
        #calcul de w
        print(tf)
        print("idf : ")
        print(idf)
        w = []
        for elem in tf:
            w.append( [x * y for x, y in zip(elem, idf)])
        return w
    
        return []      
    

#%% [markdown]
# ## 2.6 - Classifier using BoW
# 
# We are going to use logistic regression as a classifier. Read the following page to now more about this classifier: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# 
# 
# The method *train_evaluate* trains and evaluates the logistic regression model.

#%%
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def train_evaluate(training_X, training_Y, validation_X, validation_Y, bowObj):
    """
    training_X: tweets from the training dataset
    training_Y: tweet labels from the training dataset
    validation_X: tweets from the validation dataset
    validation_Y: tweet labels from the validation dataset
    bowObj: Bag-of-word object
    
    :return: the classifier and its accuracy in the training and validation dataset.
    """
    
    classifier = LogisticRegression()
    
    training_rep = bowObj.fit_transform(training_X)
    classifier.fit(training_rep, training_Y)
   
    trainAcc = accuracy_score(training_Y,classifier.predict(training_rep))
    validationAcc = accuracy_score(validation_Y,classifier.predict(bowObj.transform(validation_X)))
    
    return classifier, trainAcc, validationAcc

#%% [markdown]
# 
# ### 2.6.1 - Question 7 (4 points)
# 
# Train and calculate the logistic regression accuracy in the *training and validation dataset* using each one of the following configurations:
#     1. CountBoW + SpaceTokenizer(without tokenizer) + unigram 
#     2. CountBoW + NLTKTokenizer + unigram
#     3. TFIDFBoW + NLTKTokenizer + unigram
#     3. TFIDFBoW + NLTKTokenizer + Stemming + unigram
#     4. TFIDFBoW + NLTKTokenizer + Twitter preprocessing + Stemming  + unigram
#     5. TFIDFBoW + NLTKTokenizer + Twitter preprocessing + Stemming  + unigram + bigram
#     6. TFIDFBoW + NLTKTokenizer + Twitter preprocessing + Stemming  + unigram + bigram + trigram
# Besides the accuracy, you have to report the dictionary size for each one of configurations. Finally, describe the results found by you and answer the following questions:
# - Which preprocessing has helped the model? Why?
# - TF-IDF has achieved a better performance than CountBoW? If yes, why do you think that this has occurred? 
# - Has the bigram and trigram improved the performance? If yes, can you mention the reasons of this improvement?

#%%
model1 = CountBoW(PreprocessingPipeline(False, False, False), False, False)
# model2 = CountBoW(PreprocessingPipeline(True, False, False), False, False)
# model3 = TFIDFBoW(PreprocessingPipeline(True, False, False), False, False)
# model3bis = TFIDFBoW(PreprocessingPipeline(True, False, True), False, False)
# model4 = TFIDFBoW(PreprocessingPipeline(True, True, True), False, False)
# model5 = TFIDFBoW(PreprocessingPipeline(True, True, True), True, False)
# model6 = TFIDFBoW(PreprocessingPipeline(True, True, True), True, True)

result_model1 = train_evaluate(training_X, training_Y, validation_X, validation_Y, model1)
# result_model2 = train_evaluate(training_X, training_Y, validation_X, validation_Y, model2)
# result_model3 = train_evaluate(training_X, training_Y, validation_X, validation_Y, model3)
# result_model3bis = train_evaluate(training_X, training_Y, validation_X, validation_Y, model3bis)
# result_model4 = train_evaluate(training_X, training_Y, validation_X, validation_Y, model4)
# result_model5 = train_evaluate(training_X, training_Y, validation_X, validation_Y, model5)
# result_model6 = train_evaluate(training_X, training_Y, validation_X, validation_Y, model6)

print(result_model1)

