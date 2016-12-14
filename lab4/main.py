from __future__ import division
import nltk, re
import random
import collections
from nltk.metrics import *
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from math import log
import math
import numpy as np
np.set_printoptions(threshold=np.nan)

# Get documents from movie_reviews corpus with pos/neg tags
documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]
random.seed(5000)
random.shuffle(documents)
documents = documents[:50]

# Extract 1000 most frequent words in the whole corpus
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = all_words.keys()[:1000]

# Indicate if document contains the most frequent words in corpus
def document_features(document):
    document_words = document
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d,c) in documents]

def bigram_features(document, bigrams):
    document_words = document
    features = {}
    for bigram in bigrams:
        found = False
        for i in range(1, len(document[0])):
            if ((document[0][i-1], document[0][i]) == bigram):
                found = True
                break

        features['containsBigram(%s)' % ' '.join(bigram)] = found

    return features

def count_features(document, word_features):
    features = {}
    counter = Counter(document[0])
    for word in word_features:
        summ = counter[word]
        features['(%s)<2' % (word)] = (summ<2)
        features['2<=occurances(%s)<=4' % (word)] = (summ>=2 and summ<=4)
        features['occurances(%s)>4' % (word)] = (summ>4)
    return features


# #print featuresets[5]
#
# # Train set contains last 1600 words and test set contains first 400 words.
# train_set = featuresets[:int(round(0.8*len(featuresets)))]
# test_set = featuresets[int(round(0.8*len(featuresets))):]
# classifier = nltk.NaiveBayesClassifier.train(train_set)
#
# # Empty collections
# refsets = collections.defaultdict(set)
# testsets = collections.defaultdict(set)
#
# # Create gold standard and run words through classifier to get observed set
# for i, (feats, label) in enumerate(test_set):
#     refsets[label].add(i)
#     observed = classifier.classify(feats)
#     testsets[observed].add(i)
#
# # Print measurements
# classifier.show_most_informative_features()
# print 'overall precision: ', nltk.classify.accuracy(classifier, test_set)
# print 'pos precision:', precision(refsets['pos'], testsets['pos'])
# print 'pos recall:', recall(refsets['pos'], testsets['pos'])
# print 'pos F-measure:', f_measure(refsets['pos'], testsets['pos'])
# print 'neg precision:', precision(refsets['neg'], testsets['neg'])
# print 'neg recall:', recall(refsets['neg'], testsets['neg'])
# print 'neg F-measure:', f_measure(refsets['neg'], testsets['neg'])


# normalization, stemming, removing stop words, removing punctuation, remove numbers, or whatever you  nd interesting
def cleanUpString(document):
    pattern2 = re.compile(r'([^\s\w]|_)+')
    for i in range(0, len(document[0])):
        document[0][i] = pattern2.sub("", document[0][i])
        document[0][i] = document[0][i].lower()
    filterEmpty = filter(None, document[0])
    return tuple([filterEmpty, document[1]])

def preprocess(tokenList):
    filtered_tokens = [w for w in tokenList[0] if w.lower() not in stopwords.words('english')]


    #porter = nltk.PorterStemmer()
    snowball = nltk.SnowballStemmer('english')

    #print len(filtered_tokens)
    tokens_stem = [snowball.stem(t) for t in filtered_tokens]

    preprocessedList = tokens_stem;
    return tuple([preprocessedList, tokenList[1]])

docCleanUp = []
docProcess = []
for i in range(0, len(documents)):
    docCleanUp.append(cleanUpString(documents[i]))
for i in range(0, len(documents)):
    docProcess.append(preprocess(docCleanUp[i]))


# Stem and stopwords removal for corpus
filtered_tokens = [w for w in word_features if w.lower() not in stopwords.words('english')]
snowball = nltk.SnowballStemmer('english')
word_features_stem = [snowball.stem(t) for t in filtered_tokens]

bigrams = list(nltk.bigrams(word_features_stem))
bigram_featureset = []
for i in range(0, len(documents)):
    featuresets[i][0].update(bigram_features(docProcess[i], bigrams))
    featuresets[i][0].update(count_features(docProcess[i], word_features_stem))

#print featuresets[1]


print word_features_stem

# Indicate if document contains the most frequent words in corpus
# def document_features(document):
#     document_words = document
#     features = {}
#     for word in word_features_stem:
#         features['contains(%s)' % word] = (word in document_words)
#     return features

#featuresets = [(document_features(d), c) for (d,c) in docProcess]

# Train set contains last 1600 words and test set contains first 400 words.
train_set = featuresets[:int(round(0.8*len(featuresets)))]
test_set = featuresets[int(round(0.8*len(featuresets))):]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Empty collections
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

# Create gold standard and run words through classifier to get observed set
for i, (feats, label) in enumerate(test_set):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)

# Print measurements
classifier.show_most_informative_features()
print 'overall precision: ', nltk.classify.accuracy(classifier, test_set)
print 'pos precision:', precision(refsets['pos'], testsets['pos'])
print 'pos recall:', recall(refsets['pos'], testsets['pos'])
print 'pos F-measure:', f_measure(refsets['pos'], testsets['pos'])
print 'neg precision:', precision(refsets['neg'], testsets['neg'])
print 'neg recall:', recall(refsets['neg'], testsets['neg'])
print 'neg F-measure:', f_measure(refsets['neg'], testsets['neg'])


def count_features(document, word_features):
    features = {}
    counter = Counter(document[0])
    for word in word_features:
        summ = counter[word]
        features['(%s)<2' % (word)] = (summ<2)
        features['2<=occurances(%s)<=4' % (word)] = (summ>=2 and summ<=4)
        features['occurances(%s)>4' % (word)] = (summ>4)
    return features

def calcTF(document, word_feat):
    tfDict = {}
    counter = Counter(document[0])

    #  TF = (Number of times term t appears in a document) / (Total number of terms in the document).
    for word in word_feat:
        summ = counter[word]
        print summ
        tfDict[word] = summ / len(document[0])
        #print summ / len(document[0])
    return tfDict;

tfArray = []
for i in range(0,len(docProcess)):
    tfArray.append(calcTF(docProcess[i], word_features_stem))

print len(tfArray)
print len(tfArray[0])

def calcIDF(documents, tfArray, word_features):
    IDF = {}

    # calculate IDF
    # IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
    N = len(documents)

    for word in word_features:
        wordCount = 0
        for i in range(0, len(documents)):
            if tfArray[i][word] > 0:
                wordCount = wordCount + 1
        currentIDF = math.log(N / (1 + wordCount))
        IDF[word] = currentIDF
    return IDF

def tfidf_features(allwords, tf, idf):
    features = {}
    countFirst = 0
    countMiddle = 0
    countLast = 0
    for i in range(0, len(tf)):
        for word in allwords:
            summ = (tf[i][word]) * idf[word]
            #print 'idf ' + str(idf[word])
            #print 'tf ' + str(tf[i][word])
            if (summ < 0.5):
                countFirst = countFirst+1
            elif (summ >= 0.5 and summ <= 1.5):
                countMiddle = countMiddle+1
            else:
                countLast = countLast+1
            features['tf-idf(%s)<0.5' % (word)] = (summ < 0.5)
            features['0.5<=tf-idf(%s)<=1.5' % (word)] = (summ >= 0.5 and summ <= 1.5)
            features['tf-idf(%s)>1.5' % (word)] = (summ > 1.5)
    print countFirst
    print countMiddle
    print countLast
    return features

idfDict = calcIDF(docProcess, tfArray, word_features_stem)
tfidf = tfidf_features(word_features_stem, tfArray, idfDict)
print tfidf


#print len(idfDict)

#def calcTFIDF():
#
 #   svar = tf[idfDict[word]
 #   return tf(word, blob) * idfDict[word]
  #  return features
#
# # Create array containing all documents with its weight vector
# def constructVectors(allwords, tf, idf, weight1, weight2):
#     documents = []
#     documentVec = OrderedDict()
#     for i in range(0, len(tf)):
#         tempVec = []
#         for key, value in allwords.iteritems():
#             documentVec.update({key: 0})
#         for key, value in tf[i].iteritems():
#             #print key
#             if key in idf:
#                 documentVec[key] = (weight1 + weight2 * tf[i][key]) * idf[key]
#         for key, value in documentVec.iteritems():
#             tempVec.append(value)
#         documents.append(np.array(tempVec))
#     return documents
#

# tdfidf_dict = idf(docProcess, word_features_stem)
# print tdfidf_dict
