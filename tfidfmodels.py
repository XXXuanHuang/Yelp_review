from gensim.models import TfidfModel


model_name = '2000tfidf.model'
N = 2000


model = TfidfModel.load(model_name)


import pandas as pd
corpus1 = pd.read_json('b_r_filtered.json',lines = True)

print('test_model',model)

corpus = corpus1.iloc[:N]
print(type(corpus))
print(len(corpus))
trainVectors = []
for review_id in range(len(model.docvecs)):
    recovered_review = model.docvecs[review_id]
    trainVectors.append(recovered_review)

print(trainVectors[:5])