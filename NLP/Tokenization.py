from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import TreebankWordTokenizer

file_path = '../Datasets/NLP/chatgpt-generated.txt'

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


if __name__ == '__main__':
    corpus = get_corpus(file_path)
    documents = get_documents(corpus)
    words = get_words_from_corpus(corpus)
    print(words)


