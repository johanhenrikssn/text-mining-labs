import nltk, re
import random
import collections
from nltk.metrics import *
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords

# Get documents from movie_reviews corpus with pos/neg tags
documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]
random.seed(5000)
random.shuffle(documents)

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

# # normalization, stemming, removing stop words, removing punctuation, remove numbers, or whatever you  nd interesting
# def cleanUpString(document):
#     pattern2 = re.compile(r'([^\s\w]|_)+')
#     for i in range(0, len(document[0])):
#         document[0][i] = pattern2.sub("", document[0][i])
#         document[0][i] = document[0][i].lower()
#     filterEmpty = filter(None, document[0])
#     return tuple([filterEmpty, document[1]])
#
# def preprocess(tokenList):
#     filtered_tokens = [w for w in tokenList[0] if w.lower() not in stopwords.words('english')]
#
#
#     #porter = nltk.PorterStemmer()
#     snowball = nltk.SnowballStemmer('english')
#
#     #print len(filtered_tokens)
#     tokens_stem = [snowball.stem(t) for t in filtered_tokens]
#
#     preprocessedList = tokens_stem;
#     return tuple([preprocessedList, tokenList[1]])
#
# docCleanUp = []
# docProcess = []
# for i in range(0, len(documents)):
#     docCleanUp.append(cleanUpString(documents[i]))
# for i in range(0, len(documents)):
#     docProcess.append(preprocess(docCleanUp[i]))
#
#
# # Stem and stopwords removal for corpus
# filtered_tokens = [w for w in word_features if w.lower() not in stopwords.words('english')]
# snowball = nltk.SnowballStemmer('english')
# word_features_stem = [snowball.stem(t) for t in filtered_tokens]
# print word_features_stem
#
# # Indicate if document contains the most frequent words in corpus
# def document_features(document):
#     document_words = document
#     features = {}
#     for word in word_features_stem:
#         features['contains(%s)' % word] = (word in document_words)
#     return features
#
# featuresets = [(document_features(d), c) for (d,c) in docProcess]
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