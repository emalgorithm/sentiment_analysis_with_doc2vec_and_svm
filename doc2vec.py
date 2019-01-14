import multiprocessing
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def get_tagged_documents(reviews):
    documents = []
    i = 0

    for review in reviews:
        documents.append(TaggedDocument(review, [i]))
        i += 1

    return documents


def train_doc2vec(documents, max_epochs=20, vec_size=100, min_count=1, dm=1):
    cores = multiprocessing.cpu_count()

    model = Doc2Vec(vector_size=vec_size,
                    min_count=min_count,
                    dm=dm,
                    workers=cores)

    model.build_vocab(documents)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch + 1))
        model.train(documents,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)

    model.save(
        "doc2vec/d2v_{}vecsize_{}mincount_{}dm_{}epochs.model".format(vec_size, min_count, dm,
                                                                      max_epochs))
    print("Model Saved")