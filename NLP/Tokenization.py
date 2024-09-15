from NLP.NLPFunctions import get_corpus, get_documents, get_words_from_corpus

file_path = '../Datasets/NLP/chatgpt-generated.txt'

if __name__ == '__main__':
    corpus = get_corpus(file_path)
    documents = get_documents(corpus)
    words = get_words_from_corpus(corpus)
    print(words)


