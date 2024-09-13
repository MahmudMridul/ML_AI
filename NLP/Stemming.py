from NLP.Tokenization import get_words_from_corpus
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer

file_path = '../Datasets/NLP/stemming.txt'

def get_corpus():
    with open(file_path, 'r') as file:
        corps = file.read()
    return corps

def stem_word(list, algo = 0):
    stemmer = None

    if algo == 0:
        stemmer = PorterStemmer()
    elif algo == 1:
        stemmer = SnowballStemmer('english')

    for word in list:
        print(f'{word}\t{stemmer.stem(word)}')


if __name__ == '__main__':
    corpus = get_corpus()
    words = get_words_from_corpus(corpus)
    stem_word(words, 1)