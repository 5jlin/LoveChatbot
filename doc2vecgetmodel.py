# 训练doc2vec模型代码：
from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec
import multiprocessing


from collections import namedtuple

c = 0
doc = []
for line in open('Question.txt', 'r', encoding='UTF-8'):
    c += 1
    doc.append(line)
    if c % 5000 == 0 : print(c)



docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(doc):
    words = text.split()
    tags = [i]
    docs.append(analyzedDocument(words, tags))
    #if i % 10000 == 0 : print(text)

model = doc2vec.Doc2Vec(docs, size = 90, window = 10, min_count = 2, workers= multiprocessing.cpu_count())
model.save("model_all_2_90.bin")

