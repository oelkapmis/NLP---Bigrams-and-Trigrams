import nltk

from nltk.corpus import *
from nltk.tokenize import *
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.stem import PorterStemmer
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist

corpusdir = 'C:\Python37\corpus'
corpus = PlaintextCorpusReader(corpusdir, 'news.txt')

rawWords = [w.lower() for w in corpus.words()]

stemmer = PorterStemmer()
stemWords = [stemmer.stem(word) for word in rawWords]

finderBi = BigramCollocationFinder.from_words(rawWords)
finderTri = TrigramCollocationFinder.from_words(rawWords)

ignored_words = set(stopwords.words("english"))
filterStops = lambda w: len(w)<3 or w in ignored_words
## Eliminate stop words and words shorter than 3 letters

finderBi.apply_word_filter(filterStops)
finderBi.apply_freq_filter(3)
## Eliminate bigrams whose frequency less than three times

finderTri.apply_word_filter(filterStops)
finderTri.apply_freq_filter(2)
## Eliminate trigrams whose frequency less than twice

print('The text contains', len(corpus.sents()), 'sentences and', len(corpus.words()), 'words')
print('BIGRAMS:')
for x,y in finderBi.ngram_fd.items():
    print(x,y)
    
print('TRIGRAMS:')
for x,y in finderTri.ngram_fd.items():
    print(x,y)
