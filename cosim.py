#!/usr/bin/python
import nltk, string, numpy
# first-time use only
import numpy as np
import glob
import operator
import io
import gensim
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
filenames = glob.glob('/home/sai/Desktop/corp/*.txt')
array=[]
count=0
for fname in filenames:
    
	tfile = io.open(fname, "r",encoding='utf-8',errors='ignore')
	
	line = tfile.read()
	df=line
	array.insert(count,df)
	count=count+1
	stemmer = nltk.stem.porter.PorterStemmer()
def StemTokens(tokens):
	     return [stemmer.stem(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def StemNormalize(text):
	     return StemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
	     return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
	     return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
	
LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
LemVectorizer.fit_transform(array)

print(LemVectorizer.vocabulary_)
tf_matrix = LemVectorizer.transform(array).toarray()
tfidfTran = TfidfTransformer(norm="l2")
tfidfTran.fit(tf_matrix)
print tfidfTran.idf_
tfidf_matrix = tfidfTran.transform(tf_matrix)
print("")
print("*************** printing tfidf_martrix ****************************************")
print("")
print tfidf_matrix.toarray()
cos_similarity_matrix = tfidf_matrix * tfidf_matrix.T
print("")
print("*************** printing  cos_similarity_matrix *******************************")
print("")
print (cos_similarity_matrix).toarray()
print("")

nx_graph = nx.from_scipy_sparse_matrix(cos_similarity_matrix)
scores = nx.pagerank(nx_graph)
ranked = sorted(((scores[i],s) for i,s in enumerate(remove_punct_dict)),reverse=True)
print("")
print("*************** printing  page ranks *******************************")
print("")
print(sorted(ranked))
