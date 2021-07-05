import pyspark
import json
import pandas as pd
from gensim.utils import tokenize

sc_conf = pyspark.SparkConf() \
    .setAppName('task1') \
    .setMaster('local[*]') \
    .set('spark.driver.memory', '14g') \
    .set('spark.executor.memory', '7g') \
    .set("spark.driver.maxResultSize", "2g")



sc = pyspark.SparkContext(conf=sc_conf)
print('config set')
rdd = sc.textFile("b_r_filtered.json")
rdd = rdd.map(lambda line: json.loads(line))
all_data = rdd.map(lambda x:x["text"]).collect()
data = all_data[:500000]
print('data loaded')

import gensim.downloader as api
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, stem_text, strip_tags,strip_punctuation
dataset = []


for i in data:
    # print(tokenize(i))
    # print('i',i)

    CUSTOM_FILTERS = [remove_stopwords, stem_text, strip_tags, strip_punctuation]
    # print('preprocessing ->',preprocess_string(i, CUSTOM_FILTERS))
    # print(type(preprocess_string(i, CUSTOM_FILTERS)))
    # print('tokenizing ->', list(tokenize(i)))
    # dataset.append(list(tokenize(i)))
    dataset.append(preprocess_string(i, CUSTOM_FILTERS))

# print(dataset[0])
# print(type(dataset[0]))
dct = Dictionary(dataset)  # fit dictionary
# print(dct.token2id)
corpus = [dct.doc2bow(line) for line in dataset]  # convert corpus to BoW format
print('corpus, ',corpus[1])

model = TfidfModel(corpus)  # fit model
vector = model[corpus[0]]  # apply model to the first corpus document
print(vector)
vector2 = model[corpus[1]]
print(vector2)
print(len(model[corpus]))

# gen = (for i in range(len(model[corpus])))
# model.save("2000tfidf.model")

resultSet =set()
print(len(model[corpus]))
trainVectors = []

for review_id in range(len(model[corpus])):
    recovered_review = model[corpus[review_id]]
    result = sorted(recovered_review, key = lambda x:x[1], reverse = True)[:5]
    for i in result:
        resultSet.add(i[0])

    # print('type',type(recovered_review))
    trainVectors.append(result)
print('trainVectors',trainVectors[:5])

print(len(resultSet))

