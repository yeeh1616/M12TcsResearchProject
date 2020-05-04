import pprint
import json
from datetime import date, datetime
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim import (corpora, models, similarities)
from flask import (
    Blueprint, render_template, request
)
import os
import smart_open

bp = Blueprint('trainning', __name__, template_folder='templates')

text_corpus = [
        "Human machine interface for lab abc computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user perceived response time to error measurement",
        "The generation of random binary unordered trees",
        "The intersection graph of paths in trees",
        "Graph minors IV Widths of trees and well quasi ordering",
        "Graph minors A survey",
    ]

@bp.route('/test')
def test():
    return 'Training test.'


@bp.route('/training')
def training():
    return render_template('training.html')


@bp.route('/tfidf', methods=['GET', 'POST'])
def tfidf():
    '''01. 去除停用词'''
    # Create a set of frequent words
    stoplist = set('for a of the and to in'.split(' '))
    # Lowercase each document, split it by white space and filter out stopwords
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in text_corpus]

    # Count word frequencies
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # Only keep words that appear more than once
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
    pprint.pprint(processed_corpus)

    dictionary = corpora.Dictionary(processed_corpus)

    '''02. 生成向量'''
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]


    '''训练模型'''
    tfidf = models.TfidfModel(bow_corpus)
    tfidf.dictionary = dictionary
    tfidf.bow_corpus = bow_corpus
    # tfidf.save('app/static/temp/tfidf' + datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    tfidf.save('app/static/temp/tfidf')

    return "success!"


@bp.route('/word2vec', methods=['GET', 'POST'])
def word2vec():
    '''01. 去除停用词'''
    # Create a set of frequent words
    stoplist = set('for a of the and to in'.split(' '))
    # Lowercase each document, split it by white space and filter out stopwords
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in text_corpus]

    # Count word frequencies
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # Only keep words that appear more than once
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
    pprint.pprint(processed_corpus)

    dictionary = corpora.Dictionary(processed_corpus)

    '''02. 生成向量'''
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]


    '''训练模型'''
    # tfidf = models.TfidfModel(bow_corpus)
    model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
    model.train([text_corpus], total_examples=1, epochs=1)
    model.dictionary = dictionary
    model.bow_corpus = bow_corpus
    # model.save('app/static/temp/word2vec' + datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    model.save('app/static/temp/word2vec')

    return "success!"


@bp.route('/doc2vec', methods=['GET', 'POST'])
def doc2vec():
    train_corpus = get_train_file()
    '''Training the Model'''
    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('app/static/temp/doc2vec')

    return "success!"


def get_train_file():
    # Set file names for train and test data
    test_data_dir = os.path.join(gensim.__path__[0], 'test', 'test_data')
    lee_train_file = os.path.join(test_data_dir, 'lee_background.cor')
    train_corpus = list(read_corpus(lee_train_file))
    return train_corpus


def get_abstracts():
    result = []
    test_data_dir = os.path.join(gensim.__path__[0], 'test', 'test_data')
    fname = os.path.join(test_data_dir, 'lee_background.cor')
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            result.append(line[:100]+'...')
    return result


def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
            