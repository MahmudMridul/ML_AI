from NLP.NLPFunctions import get_corpus, stem_word, get_words_from_corpus

file_path = '../Datasets/NLP/stemming.txt'


if __name__ == '__main__':
    corpus = get_corpus(file_path)
    words = get_words_from_corpus(corpus)
    stem_word(words, 1)