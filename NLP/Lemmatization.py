from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from NLP.Tokenization import get_corpus, get_words_from_corpus

file_path = '../Datasets/NLP/stemming.txt'

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

# values for pos param
# n = noun
# v = verb
# a = adjective
# r = adverb

if __name__ == '__main__':
    corpus = get_corpus(file_path)
    words = get_words_from_corpus(corpus)
    pos_tags = pos_tag(words)

    lemm = WordNetLemmatizer()
    for _word, _pos in pos_tags:
        pos = get_pos(_pos)
        word = lemm.lemmatize(_word, pos)
        print(f'Word: {_word}\t\tPOS: {pos}\t\tResult: {word}')
