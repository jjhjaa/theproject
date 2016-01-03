# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 12:33:32 2015

@author: Jaaksi
"""
import csv
import nltk
import evaluator
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem  import PorterStemmer


# DATA IMPORT


pos_tweets, neg_tweets, neutr_tweets = [], [], [] # TRAIN DATA

with open("/Users/Jaaksi/Documents/Github/learnpython/harkkatyo/train_data2.tsv") as trainfile:
    tsvreader = csv.reader(trainfile, delimiter="\t")
    for line in tsvreader:
        if (line[2] == "negative"):
            neg_tweets.append((line[3], line[2]))
        elif (line[2] == "positive"):
            pos_tweets.append((line[3], line[2]))
        else:
            neutr_tweets.append((line[3], line[2]))

# print(pos_tweets[:2])
tweets = []

for (words, sentiment) in pos_tweets + neg_tweets:
    words_filtered = [e.lower() for e in words.split() if len(e) >=3]
    tweets.append((words_filtered, sentiment))

# stop_words = set(stopwords.words("english"))
# ps = PorterStemmer()

# for w in tweets:
#    print(ps.stem(w))
# print tweets[:50]
# filtered_tweets = []
# for w in tweets:
#    if w not in stop_words:
#       filtered_tweets.append(w)
# print(filtered_tweets)

# print(tweets[:2])
# [(['gas', 'house', 'hit', '$3.39!!!!', "i'm", 'going', 'chapel', 'hill', 'sat.'], 'positive')]
# print tweets
# pos_test_tweets, neg_test_tweets, neutr_test_tweets = [], [], [] #DEVEL DATA


# test_tweets = []
# for (words, sentiment) in pos_test_tweets + neg_test_tweets:
 #    words_filtered = [e.lower() for e in words.split() if len(e) >=3]
  #   test_tweets.append((words_filtered, sentiment))
# print test_tweets
# CLASSIFIER


def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    # print(nltk.FreqDist(wordlist).most_common(50))
    return word_features


word_features = get_word_features(get_words_in_tweets(tweets))


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


training_set = nltk.classify.apply_features(extract_features, tweets)

# DEMOCRACY


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


# CLASSIFIERS algo/train_data2/train_data
# # # # #LinearSVC_ LogisticRegression_

# NB_classifier = nltk.NaiveBayesClassifier.train(training_set)
# -
# 0,34

# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
# 0,38
# 0,38


# BNB_classifier = SklearnClassifier(BernoulliNB())
# BNB_classifier.train(training_set)
# 0,26
# 0,26

# LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
# LogisticRegression_classifier.train(training_set)
# 0,42
# 0,46
# C 1000 ~479
# C 1200 479
# C 0.01 26
# C 100 477
# C 100000 482

# SGDClassifier_
#classifier = SklearnClassifier(SGDClassifier())
#classifier.train(training_set)
# -
#  0,47

# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# -
# 0,26


classifier = SklearnClassifier(LinearSVC(C=500, max_iter=1000000))
classifier.train(training_set)
# 0,43
# 0,48
# 0.01 039
#F-score: 0.493201251539 100000000 #LinearSVC_

# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)
# -
# 0,45

# lol vote LinearSVC_ LinearSVC_# NuSVC_# LinearSVC_



#   print(wordlist.most_common(10))
# print(classifier.show_most_informative_features(32))
# print(extract_features())
# tweet = "'Love-cheat' Daniel Radcliffe splits with girlfriend Rosie Coker: London, Oct 19: Daniel Radcliffe has split wit... http://tinyurl.com/8oxx2nsÂ "
# print(classifier.classify(extract_features(tweet.split())))


with open("/Users/Jaaksi/Documents/Github/learnpython/harkkatyo/test_data.tsv", "r") as testfile, open("/Users/Jaaksi/Documents/Github/learnpython/harkkatyo/evalfile.tsv", "w") as evalfile:
    tsvreader = csv.reader(testfile, dialect='excel-tab',delimiter="\t")
    evalwriter = csv.writer(evalfile, dialect='excel-tab', delimiter='\t')
    for line in tsvreader:
        tweet = line[3]
        result = classifier.classify(extract_features(tweet.split()))
        evalwriter.writerow([line[0], line[1], result, line[3]])

evaluator.evaluate("/Users/Jaaksi/Documents/Github/learnpython/harkkatyo/test_data.tsv", "/Users/Jaaksi/Documents/Github/learnpython/harkkatyo/evalfile.tsv")

# print(classifier.show_most_informative_features(15))









