from gensim.models.doc2vec import Doc2Vec
max_epochs = 20
vec_size = 100
alpha = 0.025
print('building model...')
model = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)




gen = (model.train(tagged_data,total_examples=N,epochs=model.iter) for i in range(max_epochs))
print('gen',type(gen))

state = True
counter = 0

while state:

    print('iteration {0}'.format(counter))
    gen.__next__()
    counter += 1
    if counter == 49:
        state = False



model.load("standard275000.model")