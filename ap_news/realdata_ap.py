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

def E_step_Realdata(alpha, BETA, doc, Phi0, gamma0, max_iter=100, tol=1e-3):
    """
    Latent Dirichlet Allocation: E-step.
    Do to a specific document.
    ------------------------------------
    Input:
    alpha as a k*1 vector;
    BETA as a k*V matrix;
    doc as a Nd*V matrix;
    Phi0 as a Nd*k matrix;
    gamma0 as a k*1 vector;
    tol as a float: tolerance.
    -------------------------------------
    Output:
    optimal Nd*k matrix Phi;
    optimal k*1 vector gamma."""

    # Initialization
    Phi = Phi0
    gamma = gamma0
    phi_delta = 1
    gamma_delta = 1

    Phi = Phi.astype(np.float128)
    Phi0 = Phi0.astype(np.float128)

    # relative tolerance is for each element in the matrix
    tol = tol ** 2
    for iteration in range(max_iter):
        ##update Phi
        gamma = gamma / min(gamma)
        Phi = (doc @ BETA.T) * np.exp(digamma(gamma) - digamma(sum(gamma)))
        Phi = Phi / (Phi.sum(axis=1)[:, None])  # row sum to 1

        ##update gamma
        gamma = alpha + Phi.sum(axis=0)

        ##check the convergence
        phi_delta = np.mean((Phi - Phi0) ** 2)
        gamma_delta = np.mean((gamma - gamma0) ** 2)

        ##refill
        Phi0 = Phi
        gamma0 = gamma
        if ((phi_delta <= tol) and (gamma_delta <= tol)):
            break

    return Phi, gamma


def M_step_Realdata(docs, k, tol=1e-3, tol_estep=1e-3, max_iter=100, initial_alpha_shape=5, initial_alpha_scale=2):
    """
    Latent Dirichlet Allocation: M-step.
    Do to a list of documnents. -- a list of matrix.
    -------------------------------------------------
    Input:
    docs: a list of one-hot-coding matrix ;
    k: a fixed positive integer indicate the number of topics.
    -------------------------------------------------
    Output:
    optimal Nd*k matrix Phi;
    optimal k*1 vector gamma;
    optimal k*V matrix BETA;
    optimal k*1 vector alpha.
    """

    # get basic iteration
    M = len(docs)
    V = docs[1].shape[1]
    N = [doc.shape[0] for doc in docs]

    # initialization
    BETA0 = np.random.dirichlet(np.ones(V), k)
    alpha0 = np.random.gamma(shape=initial_alpha_shape, scale=initial_alpha_scale, size=k)
    PHI = [np.ones((N[d], k)) / k for d in range(M)]
    GAMMA = np.array([alpha0 + N[d] / k for d in range(M)])

    BETA = BETA0
    alpha = alpha0
    alpha_dis = 1
    beta_dis = 1

    # relative tolerance: tolerance for each element
    tol = tol ** 2

    for iteration in range(max_iter):
        print(iteration)
        # update PHI,GAMMA,BETA
        BETA = np.zeros((k, V))
        for d in range(M):  # documents
            PHI[d], GAMMA[d,] = E_step_Realdata(alpha0, BETA0, docs[d], PHI[d], GAMMA[d,], max_iter, tol_estep)
            BETA += PHI[d].T @ docs[d]
        BETA = BETA / (BETA.sum(axis=1)[:, None])  # rowsum=1

        # update alpha

        z = M * polygamma(1, sum(alpha0))
        h = -M * polygamma(1, alpha0)
        g = M * (digamma(sum(alpha0)) - digamma(alpha0)) + (digamma(GAMMA) - digamma(GAMMA.sum(axis=1))[:, None]).sum(
            axis=0)
        c = (sum(g / h)) / (1 / z + sum(1 / h))
        alpha = alpha0 - (g - c) / h

        alpha_dis = np.mean((alpha - alpha0) ** 2)
        beta_dis = np.mean((BETA - BETA0) ** 2)
        alpha0 = alpha
        BETA0 = BETA
        if ((alpha_dis <= tol) and (beta_dis <= tol)):
            break

    return alpha, BETA

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


a, B = M_step_Realdata(docs=docs,k=10,tol=1e-3,tol_estep=1e-3,max_iter=100,initial_alpha_shape=100,initial_alpha_scale=0.01)

import heapq


def find_index(x):
    """find the index of the largest 10 values in a list"""

    x = x.tolist()
    max_values = heapq.nlargest(50, x)
    index = [0] * 50
    for i in range(50):
        index[i] = x.index(max_values[i])

    return index

rep_words_index = list(map(find_index, B))

print(words[rep_words_index[0]])
print(words[rep_words_index[1]])
print(words[rep_words_index[2]])
print(words[rep_words_index[3]])
print(words[rep_words_index[4]])
print(words[rep_words_index[5]])
print(words[rep_words_index[6]])
print(words[rep_words_index[7]])
print(words[rep_words_index[8]])
print(words[rep_words_index[9]])