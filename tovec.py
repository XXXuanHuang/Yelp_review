import pyspark
import json

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

# if __name__ == '__main__':

sc_conf = pyspark.SparkConf() \
    .setAppName('task1') \
    .setMaster('local[*]') \
    .set('spark.driver.memory', '14g') \
    .set('spark.executor.memory', '7g') \
    .set("spark.driver.maxResultSize", "2g")

sc = pyspark.SparkContext(conf=sc_conf)
b_rdd = sc.textFile("b_r_filtered.json")
b_rdd = b_rdd.map(lambda line: json.loads(line))
all_data = b_rdd.map(lambda x:x["text"]).collect()
print(all_data[:3])
print(len(all_data))
print(b_rdd.count())
#
N = 500000
# N = 275000

data = all_data[:N]
print('tagging')
# tagged_gen = (TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data))
# print(type(tagged_gen))

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]


print('tagging completed')
max_epochs = 20
vec_size = 100
alpha = 0.025
print('building model...')
model = Doc2Vec(workers = 5,
                size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)
model.build_vocab(tagged_data)
print('model completed')


gen = (model.train(tagged_data,total_examples=N,epochs=model.iter) for i in range(max_epochs))
print('gen',type(gen))

state = True
counter = 0

while state:

    print('iteration {0}'.format(counter))
    gen.__next__()
    counter += 1
    if counter == 19:
        state = False


model.save("sReducedN.model")
print("Model Saved")
 