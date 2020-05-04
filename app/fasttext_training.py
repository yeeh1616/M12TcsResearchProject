from pprint import pprint as print
from gensim.models.fasttext import FastText as FT_gensim
from gensim.test.utils import datapath
import tempfile
import os

def fasttext():
    # Set file names for train and test data
    corpus_file = datapath('lee_background.cor')

    model = FT_gensim(size=100)

    # build the vocabulary
    model.build_vocab(corpus_file=corpus_file)

    # train the model
    model.train(
        corpus_file=corpus_file,
        epochs=model.epochs,
        total_examples=model.corpus_count,
        total_words=model.corpus_total_words
    )

    print(model)




    with tempfile.NamedTemporaryFile(prefix='saved_model_gensim-', delete=False) as tmp:
        model.save(tmp.name, separately=[])

    loaded_model = FT_gensim.load(tmp.name)
    print(loaded_model)

    os.unlink(tmp.name)
