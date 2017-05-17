# -*- coding: utf-8 -*-

import os
from konlpy.tag import Twitter
import nltk


def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]   # header 제외
    return data


def tokenize(doc, pos_tagger):
    # norm, stem은 optional
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]


def term_exists(doc, selected_words):
    return {'exists({})'.format(word): (word in set(doc)) for word in selected_words}


pos_tagger = Twitter()

ratings_train_cache_file_path = './cache/ratings_train.txt'
ratings_test_cache_file_path = './cache/ratings_test.txt'

if os.path.exists(ratings_train_cache_file_path):
    f = open(ratings_train_cache_file_path, 'r')
    data = f.read()
    f.close()
    train_docs = eval(data)
else:
    train_data = read_data('./data/ratings_train.txt')
    train_docs = [(tokenize(row[1], pos_tagger), row[2]) for row in train_data]
    f = open(ratings_train_cache_file_path, 'w')
    f.write(str(train_docs))
    f.close()

if os.path.exists(ratings_test_cache_file_path):
    f = open(ratings_test_cache_file_path, 'r')
    data = f.read()
    f.close()
    test_docs = eval(data)
else:
    test_data = read_data('./data/ratings_test.txt')
    test_docs = [(tokenize(row[1], pos_tagger), row[2]) for row in test_data]
    f = open(ratings_test_cache_file_path, 'w')
    f.write(str(test_docs))
    f.close()

tokens = [t for d in train_docs for t in d[0]]

text = nltk.Text(tokens, name='NMSC')

# 여기서는 최빈도 단어 2000개를 피쳐로 사용
# WARNING: 쉬운 이해를 위한 코드이며 time/memory efficient 하지 않습니다
selected_words = [f[0] for f in text.vocab().most_common(2000)]

# 시간 단축을 위한 꼼수로 training corpus의 일부만 사용할 수 있음
train_docs = train_docs[:10000]
train_xy = [(term_exists(d, selected_words), c) for d, c in train_docs]
test_xy = [(term_exists(d, selected_words), c) for d, c in test_docs]

# 훈련
classifier = nltk.NaiveBayesClassifier.train(train_xy)

# 테스트
err = 0
print('test on: ', len(test_xy))
for r in test_xy:
    sent = classifier.classify(r[0])
    if sent != r[1]:
        err += 1.
print('error rate: ', err / float(len(test_xy)))
