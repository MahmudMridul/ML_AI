from nltk.stem import WordNetLemmatizer

# values for pos param
# n = noun
# v = verb
# a = adjective
# r = adverb

if __name__ == '__main__':
    lemm = WordNetLemmatizer()
    word = lemm.lemmatize('went', pos='v')
    print(word)
