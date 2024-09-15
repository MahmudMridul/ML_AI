from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import TreebankWordTokenizer

from nltk.corpus import wordnet

from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer

from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

def get_corpus(f_path):
    with open(f_path, 'r') as file:
        corps = file.read()
    return corps

def get_documents(cor):
    docs = sent_tokenize(cor)
    return docs

def get_words_from_corpus(corp, tokenizer = 0):
    if tokenizer == 0:
        return word_tokenize(corp)
    elif tokenizer == 1:
        return wordpunct_tokenize(corp)
    elif tokenizer == 2:
        obj = TreebankWordTokenizer()
        return obj.tokenize(corp)

def get_pos(_pos):
    if _pos.startswith('J'):
        return wordnet.ADJ
    elif _pos.startswith('V'):
        return wordnet.VERB
    elif _pos.startswith('N'):
        return wordnet.NOUN
    elif _pos.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def stem_word(word_list, algo = 0):
    stemmer = None

    if algo == 0:
        stemmer = PorterStemmer()
    elif algo == 1:
        stemmer = SnowballStemmer('english')

    for word in word_list:
        print(f'{word}\t{stemmer.stem(word)}')

def lemmatize_words(words):
    pos_tags = pos_tag(words)
    lemm = WordNetLemmatizer()

    for _word, _pos in pos_tags:
        pos = get_pos(_pos)
        word = lemm.lemmatize(_word, pos)
        print(f'Word: {_word}\t\tPOS: {pos}\t\tResult: {word}')
