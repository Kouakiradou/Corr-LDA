import gensim
import numpy as np
from scipy.special import digamma, polygamma
import time
from decimal import Decimal

f = open('ap.txt', 'r')
text = f.readlines()
f.close()

D1 = set([x for x in text if '<DOC>\n' not in x])
D2 = set([x for x in text if '</DOC' not in x])
D3 = set([x for x in text if 'TEXT>' not in x])
D = list(set.intersection(D1, D2, D3))


import toolz as tz
import toolz.curried as c
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re
import pandas as pd
import numpy as np


stops = [
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"]


d = tz.pipe(
    D,
    c.map(lambda x: x.strip()),
    c.map(lambda x: x.lower()),
    c.map(lambda x: x.translate(str.maketrans('', '', string.punctuation))),
    c.map(lambda x: re.sub('[0-9]+', '', x)),
    c.map(lambda x: x.split()),
    c.map(lambda x: [word for word in x if word not in stops]),
    list
)

d_sub = d[:500]


tf = {id: tz.frequencies(doc) for id, doc in enumerate(d_sub)}
df = pd.DataFrame(tf).fillna(0)
words = df.index

ds = df.values.T
ds = ds.astype(int)


def DataTrans(x):
    """Turn the data into the desired structure"""

    N_d = np.sum(x)
    V = len(x)

    row = 0

    doc = np.zeros((N_d, V))
    for i in range(V):
        if x[i] == 0:
            pass
        else:
            for j in range(x[i]):
                doc[row, i] = 1
                row += 1

    return doc

# def DataTrans(x):
#     """Turn the data into the desired structure"""
#
#     N_d = np.count_nonzero(x)
#     V = len(x)
#
#     row = 0
#
#     doc = np.zeros((N_d, V))
#     for i in range(V):
#         if x[i] == 0:
#             pass
#         else:
#             doc[row, i] = x[i]
#             row += 1
#
#     return doc

docs = list(map(DataTrans, ds))
print(docs[0])



import heapq


print("transforming data...")
# docs = list(map(DataTrans, ds))
dictionary = gensim.corpora.Dictionary(d_sub)
dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in d_sub]
print(bow_corpus[20])



print("start training")
# a, B = M_step_Realdata(docs=docs, k=8, tol=1e-3, tol_estep=1e-3, max_iter=100, initial_alpha_shape=100,
#                        initial_alpha_scale=0.01)
lda_model =  gensim.models.LdaMulticore(bow_corpus,
                                   num_topics = 8,
                                   id2word = dictionary,
                                   passes = 10,
                                   workers = 2)

for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")