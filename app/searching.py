import pprint
import json
import gensim
from gensim import (corpora, models, similarities)
from flask import (
    Blueprint, render_template, request
)
from flask_paginate import Pagination, get_page_parameter
from gensim.models import Word2Vec

from app.training import text_corpus, get_abstracts

import gensim.downloader as api
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import Text8Corpus
from gensim.similarities.index import AnnoyIndexer
import os
import smart_open

bp = Blueprint('search', __name__, template_folder='templates')


@bp.route('/test')
def test():
    return 'Search engine started.'


@bp.route('/test2')
def test2():
    return render_template('searching.html')


@bp.route('/')
@bp.route('/searching', methods=['GET', 'POST'])
def searching():
    search = False
    q = request.args.get('q')
    if q:
        search = True

    search_term = request.args.get("search")
    page = request.args.get(get_page_parameter(), type=int, default=1)
    per_page = 1
    offset = (page - 1) * per_page

    # doc_list=tfidf_go(search_term)
    doc_list=doc2vec_go(search_term)

    pagination = Pagination(page=page,
                            total=len(doc_list),
                            search=search,
                            record_name='all_modules_c',
                            per_page=per_page,
                            show_single_page=True,
                            link='<li><a class="pgn__num" href="{0}">{1}</a></li>')

    pagination.current_page_fmt = '<li><span class="pgn__num current">{0}</span></li>'
    pagination.prev_page_fmt = '<li><a class="pgn__prev" href="{0}">{1}</a></li>'
    pagination.next_page_fmt = '<li><a class="pgn__next" href="{0}">{1}</a></li>'
    pagination.gap_marker_fmt = '<li><span class="pgn__num dots">…</span></li>'
    pagination.link = '<li><a class="pgn__num" href="{0}">{1}</a></li>'
    pagination.link_css_fmt = '<div class="{0}{1}"><ul>'
    pagination.prev_disabled_page_fmt = ''
    pagination.next_disabled_page_fmt = ''

    return render_template('searching.html',
                           doc_list=doc_list,
                           pagination=pagination,
                           search=search_term)

    # return render_template('searching.html')


def tfidf_go(key_words):
    if key_words is '':
        return []

    if key_words is None:
        return []

    '''计算相似度'''
    # tfidf = models.TfidfModel.load("app/static/temp/word2vec")
    tfidf = Word2Vec.load("app/static/temp/word2vec")
    dictionary = tfidf.dictionary
    bow_corpus = tfidf.bow_corpus

    query_document = key_words.split()
    query_bow = dictionary.doc2bow(query_document)
    index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)
    sims = index[tfidf[query_bow]]
    print(list(enumerate(sims)))

    result = []
    for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
        print(document_number, score)

        if score > 0:
            doc_obj = Doc_obj(document_number, text_corpus[document_number], round(score, 2), 'https://www.google.nl/')
            result.append(doc_obj)

    return result


def doc2vec_go(key_words):
    if key_words is '':
        return []

    if key_words is None:
        return []

    abstracts = get_abstracts()

    model = gensim.models.Doc2Vec.load("app/static/temp/doc2vec")
    vector = model.infer_vector(list(key_words.split()))
    # print(vector)

    # sims = model.docvecs.most_similar([vector], topn=len(model.docvecs))
    sims = model.docvecs.most_similar([vector], topn=10)

    result = []
    for item in sims:
        document_number = int(item[0])
        score = round(item[1], 2)

        if score > 0:
            doc_obj = Doc_obj(document_number, abstracts[document_number], score, 'https://www.google.nl/')
            result.append(doc_obj)

    return result

@bp.route('/demo02')
def demo02():
    text8_path = api.load('text8', return_path=True)
    text8_path

    # Using params from Word2Vec_FastText_Comparison
    params = {
        'alpha': 0.05,
        'size': 100,
        'window': 5,
        'iter': 5,
        'min_count': 5,
        'sample': 1e-4,
        'sg': 1,
        'hs': 0,
        'negative': 5
    }
    model = Word2Vec(Text8Corpus(text8_path), **params)
    print(model)

    # 100 trees are being used in this example
    annoy_index = AnnoyIndexer(model, 100)
    # Derive the vector for the word "science" in our model
    vector = model.wv["science"]
    # The instance of AnnoyIndexer we just created is passed
    approximate_neighbors = model.wv.most_similar([vector], topn=11, indexer=annoy_index)
    # Neatly print the approximate_neighbors and their corresponding cosine similarity values
    print("Approximate Neighbors")
    for neighbor in approximate_neighbors:
        print(neighbor)

    normal_neighbors = model.wv.most_similar([vector], topn=11)
    print("\nNormal (not Annoy-indexed) Neighbors")
    for neighbor in normal_neighbors:
        print(neighbor)



@bp.route('/demo01')
def demo01():
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
    dic = dictionary.token2id

    new_doc = "Human computer interaction"
    new_vec = dictionary.doc2bow(new_doc.lower().split())

    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]


    '''训练模型'''
    tfidf = models.TfidfModel(bow_corpus)

    # transform the "system minors" string
    words = "system minors".lower().split()
    print(tfidf[dictionary.doc2bow(words)])

    '''计算相似度'''
    index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)
    query_document = 'system engineering'.split()
    query_bow = dictionary.doc2bow(query_document)
    sims = index[tfidf[query_bow]]
    print(list(enumerate(sims)))
    for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
        print(document_number, score)

    return str(new_vec)



class Doc_obj():
    id = -1
    abstract = ''
    score = 0.0
    url = ''

    def __init__(self, id, abstract, score, url):
        self.id = id
        self.abstract = abstract
        self.score = score
        self.url = url
