from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from NLP.NLPFunctions import get_pos, lemmatize_words
from NLP.Tokenization import get_corpus, get_words_from_corpus

file_path = '../Datasets/NLP/stemming.txt'
# values for pos param
# n = noun
# v = verb
# a = adjective
# r = adverb

if __name__ == '__main__':
    corpus = get_corpus(file_path)
    words = get_words_from_corpus(corpus)
    lemmatize_words(words)
