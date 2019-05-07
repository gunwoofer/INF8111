#!/usr/bin/env python
# coding: utf-8

# # 1 - Overview
# 
# Twitter is a mix of social network and microblogging. In this network, people post information and communicate among themselves through messages, called tweets, that can contain up to 280 characters. In this assignment, *we will implement a prototype that can detect if an airline company is positively or negatively mentioned in a tweet*. 
# 
# 

# # 2 - Sentiment Analysis Model (13 points)
# 
# In the literature, the task of extracting the sentiment of a text is called *sentiment analysis*. We will implement a bag-of-words (BoW) model for this task.
# 
# ## 2.1 -  Setup
# 
# Please run the code below to install the packages needed for this assignment.

# In[3]:


# If you want, you can use anaconda and install after nltk library
# pip install --user numpy
# pip install --user sklearn
# pip install --user scipy
# pip install --user nltk


#python
#import nltk
#nltk.download("punkt")
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('universal_tagset')


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

# In[18]:


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

# In[ ]:




class SpaceTokenizer(object):
    """
    It tokenizes the tokens that are separated by whitespace (space, tab, newline). 
    We consider that any tokenization was applied in the text when we use this tokenizer.
    
    For example: "hello\tworld of\nNLP" is split in ['hello', 'world', 'of', 'NLP']
    """
    
    def tokenize(self, text):
        # Write your code here
        raise NotImplementedError("")
        
        # Have to return a list of tokens
        return tokens
        
class NLTKTokenizer(object):
    """
    This tokenizer uses the default function of nltk package (https://www.nltk.org/api/nltk.html) to tokenize the text.
    """
    
    def tokenize(self, text):
        # Write your code here
        
        raise NotImplementedError("")
        
        # Have to return a list of tokens
        return tokens

        


# ### 2.3.2 - Stemming
# 
# In the tweets *"I should have bought a new shoes today"* and *"I spent too much money buying games"*, the words *"buy"* and *"bought"* represent basically the same concept. Considering both words as different can unnecessarily increase the dimensionality of the problem and can negatively impact the performance of simple models. Therefore, a unique form (e.g., the root buy) can represent both words. The process to convert words with the same stem (word reduction that keeps word prefixes) to a standard form is called *stemming*.
# 
# #### 2.3.2.1 - Question 2 (0.5 point) 
# 
# Retrieve the stems of the tokens using the attribute *stemmer* from the class *Stemmer*.

# In[ ]:


from nltk.stem.snowball import SnowballStemmer

class Stemmer(object):
    
    def __init__(self):
        self.stemmer = SnowballStemmer("english", ignore_stopwords=True)
    
    def stem(self, tokens):
        """
        tokens: a list of strings
        """
        # Write your code here
        raise NotImplementedError("")
        
        # Have to return a list of stems
        return tokens
        


# ### 2.3.3 - Twitter preprocessing
# 
# Sometimes only applying the default NLP preprocessing steps is not enough. Data for certain domains can have peculiar characteristics which requires specific preprocessing steps to remove the noise and create a more suitable format for the models. 
# 
# In NLP, methods store a set of words, called dictionary, and all the words out of the dictionary are considered as unknown. In this assignment, the feature space dimensionality of a model is directly related to the number of words in the dictionary. Since high-dimensional spaces can suffer from the curse of dimensionality, our goal is to create preprocessing steps that decrease vocabulary size.  
# 
# #### 2.3.3.1 - Question 3 (2.0 points)
# 
# Briefly explain and implement at least two preprocessing steps that reduce the dictionary size (number of unique words). These preprocessing steps must be related to the specific characteristic of the Twitter data. Therefore, for instance, the stop words removal will not be accepted as a preprocessing step.

# In[1]:


class TwitterPreprocessing(object):
    
    def preprocess(self, tweet):
        """
        tweet: original tweet
        """
        # Write your preprocessing steps here.
        raise NotImplementedError("")
    
        # return the preprocessed twitter
        return tweet
        
        
        


# ### 2.3.3  Pipeline
# 
# The pipeline is sequence of preprocessing steps that transform the raw data to a format that is suitable for your problem. We implement the class *PreprocessingPipeline* that apply the tokenizer, twitter preprocessing and stemer to the text.
# 
# **Feel free to change the preprocessing order.**

# In[3]:


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

# In[ ]:


def bigram(tokens):
     """
    tokens: a list of strings
    """
    # Write your code here
    # This function returns the list of bigrams
    
def trigram(tokens):
     """
    tokens: a list of strings
    """
    # Write your code here
    # This function returns the list of trigrams
    
    


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

# In[ ]:


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
        raise NotImplementedError("")
    
        return []
        
    def transform(self, X):
        """
        This method preprocesses the data using the pipeline object and  transforms the text in a list of integer.
        Vectors are weighted using the token frequencies in the sentence.
        
        X: a list of vectors
        
        :return: a list of vectors
        """        
        raise NotImplementedError("")
        
        return []
      
        
        
        
        
    


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

# In[2]:


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

        
    def fit_transform(self, X):
        """
        This method preprocesses the data using the pipeline object, calculates the IDF and TF and 
        transforms the text in vectors. Vectors are weighted using TF-IDF method.
        
        X: a list that contains tweet contents
        
        :return: a list that contains the list of integers
        """
        raise NotImplementedError("")
    
        return []
        
    def transform(self, X):
        """
        This method preprocesses the data using the pipeline object and  
            transforms the text in a list of integer.
        
        X: a list of vectors
        
        :return: a list of vectors
        """        
        
        # transform the dataset to bag-of-words
        raise NotImplementedError("")
    
        return []


# ## 2.6 - Classifier using BoW
# 
# We are going to use logistic regression as a classifier. Read the following page to now more about this classifier: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# 
# 
# The method *train_evaluate* trains and evaluates the logistic regression model.

# In[1]:


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

# In[2]:






# # 3 Prototype (7 points)
# 
# During the last years, *E Corp* has collected tweets to create a dataset to their sentiment analysis tool. Now, airline companies have contracted *E Corp* to analyze the consumer opinion about them. Your job is to extract information from the tweet database about the following companies: Air France, American, British Airways,  Delta, Southwest, United, Us Airways and Virgin America.
# 
# *For the prototype, you have to use the best model found in the Section 2.*
# 
# ## 3.1 Dataset
# 
# In https://drive.google.com/file/d/1Cuw6Y12Bj91vF_iH49mqPZZfJkY92iBY/view?usp=sharing, you can find the raw tweet retrieved by E corp.  Each tweet is represented as json that the have attributes listed in the page https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object.
# 
# ** You will answer the question of this section using this tweet database (https://drive.google.com/file/d/1Cuw6Y12Bj91vF_iH49mqPZZfJkY92iBY/view?usp=sharing).**
# 
# ## 3.2 Sentiment Analysis
# 
# 
# ### 3.2.1 Question 8 (0.5 point)
# 
# Implement the method *extract_tweet_content* that extracts the content of each tweet in the database.

# In[ ]:


def extract_tweet_content(raw_tweet_file):
    """
    Extract the tweet content for each json object
    
    raw_tweet_file: file path that contains all json objects
    
    :return: a list with the tweet contents
    """
    raise NotImplementedError("")
    
    
    
    


# ### 3.2.1 Question 9 (1 points)
# 
# Implement the method *detect_airline* that detects the airline companies in a tweet. Besides that, explain your approach to detect the companies and its possible drawbacks.
# 
# The detect_airline has to be able to return if none or more than one airline companies are mentioned in a tweet

# In[ ]:



def detect_airline(tweet):
    """
    Detect and return the airline companies mentioned in the tweet
    
    tweet: represents the tweet message. You should define the data type
    
    :return: list of detected airline companies
    """
    raise NotImplementedError("")


# 
# ### 3.2.1 Question 10 (0.5 points)
# 
# Implement the method *extract_sentiment* that receives a tweet and extracts its sentiment.

# In[20]:


def extract_sentiment(classifier, tweet):
    """
    Extract the tweet sentiment
    
    classifier: classifier object
    tweet: represents the tweet message. You should define the data type
    
    :return: list of detected airline companies
    """
    raise NotImplementedError("")


# ### 3.2.1 Question 11 (2 points)
# 
# Using the *extract_tweet_content*, *detect_airline* and *extract_sentiment*, implement a code that generates a bar chart that contains the number of positive, neutral and negatives tweets for each one of the companies. Briefly describe your bar chart (e.g, which was the company with most negative tweets) and how this chart can help airline companies.   

# In[ ]:





# ## 3.3 - Term Analysis
# 
# POS-tagging consists of extracting the part-of-speech (POS) of each token in a sentence. For instance, the table below depicts the part-of-speechs of the sentence *The cat is white!* are.
# 
# 
# 
# |   The   | cat  |  is  | white     |    !       |
# |---------|------|------|-----------|------------|
# | article | noun | verb | adjective | punctation |
# 
# 
# The part-of-speech can be more complex than what we have learned in the school. Linguistics need to have a more detailed information about systax information of the words in a sentence. For our problem, we do not need this level of information and, thus, we will use a less complex set, called universal POS tags. 
# 
# In POS-tagging, each part-of-speech is represented by a tag. You can find the POS tag list used in this assignement at https://universaldependencies.org/u/pos/ .

# In[28]:


# NLTK POS-tagger

import nltk


#before using pos_tag function, you have to tokenize the sentence.
s = ['The', 'cat', 'is',  'white', '!']
nltk.pos_tag(s,tagset='universal')


# ### 3.3.1 Question 12 (2 points)
# 
# **Implement a code** that retrieves the top 10 most frequent terms for each airline company. You will only consider the terms that appear in a positive and negative tweets. Besides that, we consider as term:
# 1. Words that are either an adjective or a noun
# 2. n-grams that are composed by adjectives followed by a noun (e.g., dirty place) or a noun followed by another noun (e.g.,sports club).
# 
# Moreover, **generate a table** with the top 10 most frequent terms and their normalized frequencies(percentage) for each airline company.
# 
# **Do not forget to remove the company names from the chart.**

# In[ ]:





# ### 3.3.2 Question 13 (1 point)
# 
# The table generated in the Question 12 can lead us to any conclusion about each one of the 9 companies? Can we identify specific events that have occured during the data retrieval?

# 

# # 4 - Bonus (2 points)
# 
# Person names, companies names and locations are called named entities. Named-entity recognition (NER) is the task of extracting named entities  classifying them using pre-defined categories. In this bonus section, you will use a Named Entity Recognizer to automatically extract named entities from the tweets. This approach is generic enough to retrieve information about other companies or even product and people names.
# 
# **For the bonus, you are free to use any Named Entity Recognizer that has python wrapper or is implemented in python. Moreover, you have to use the tweet database of the previous section (https://drive.google.com/file/d/1Cuw6Y12Bj91vF_iH49mqPZZfJkY92iBY/view?usp=sharing)**
# 

# ## 4.1 - Bonus 2 (1 point)
# 
# Implement a code that generates the table with the top 10 most mentioned named entities in the database (this table has to contain the frequencies of the name entities). After that, generates a bar chart that despicts the number of positive, negative and neutral tweets for each one of these 10 named entities. Briefly describe the results found in the bar chart.
# 
# *Ignore the named entities related to the following airline companies : Air France, American, British Airways,  Delta, Southwest, United, Us Airways and Virgin America.*

# In[ ]:





# ## 4.2 - Bonus 3 (1 point)
# 
# Generate a similar table produced in the Question 12 for the 10 most mentioned named entities in Bonus 2. Can we draw any conclusion about these named entities?

# In[ ]:




