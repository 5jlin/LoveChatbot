import jieba
from gensim.summarization import bm25
from gensim import corpora
import collections
from collections import OrderedDict
import gensim
import random

import readfile

question = []
answer = []
topN = 50
stopword = readfile.read('stopword.txt')

words = jieba.cut("123", cut_all=False)
st = ""
for word in words:
        st = st + word + " "
        
for line in open('Question.txt', 'r', encoding='UTF-8'):
    question.append(line)

for line in open('Answer.txt', 'r', encoding='UTF-8'):
    tmp = line.split('****')
    for n in tmp:
        if len(n) < 2 : tmp.remove(n)
    answer.append(tmp)

model = gensim.models.doc2vec.Doc2Vec.load('model_all_2.bin')

while True:

    inputst = input('請輸入問題：')
    st = ""
    words = jieba.cut(inputst, cut_all=False)
    for word in words:
        st = st + word + " "
    sttmp = st
    st = st.split()
    
    new_doc_vec = model.infer_vector(st)
    best = model.docvecs.most_similar([new_doc_vec], topn=topN)
    selectQ = []
    posA = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    for i in range(20):
        if '置底' in question[best[topN-1-i][0]] : continue
        if ('公告' or '水桶') in question[best[19-i][0]][:20] : continue
        content = question[best[topN-1-i][0]]
        #print(question.index(content))
        posA[i] = best[topN-1-i][0]
        tmp = []
        for i in content.strip().split() :
            if i not in stopword :
                tmp.append(i)
        selectQ.append(tmp)

    dictionary = corpora.Dictionary(selectQ)
    bm25Model = bm25.BM25(selectQ)
    average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
    query = []
    for word in st:
        query.append(word)
    scores = bm25Model.get_scores(query,average_idf)

    idx = scores.index(sorted(scores)[-1])

    length = []
    for n in answer[posA[idx]]:
        length.append(len(n))
    print(answer[posA[idx]][length.index(min(length))].replace(' ', ''))
 
