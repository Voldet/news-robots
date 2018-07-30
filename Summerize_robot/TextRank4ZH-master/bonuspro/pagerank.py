# -*- coding: utf-8 -*

import numpy as np
import networkx as nx
import sys


from json import *


'''*************************更改编码方式*******************************'''
def input(path):
    import codecs
    from textrank4zh import TextRank4Sentence

    text = codecs.open(path, 'r', 'gbk').read()
    # text = codecs.open(path, 'r', 'utf8').read()

    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')

    return tr4s


class Abstract(object):

    def __init__(self):
        self.sentences = None
        self.key_sentence = None
        self.word_list = None
        self.text_rank = None
        self.length = None
        self.text = None
        self.first = None
        self.last = None

    def read_text(self, st, wl):
        self.sentences = st[1:len(st)-2]
        # print(type(st))
        self.first = st[0]
        self.last = st[len(st)-1]
        self.word_list = wl

    @staticmethod
    def similarity(word_list1, word_list2):
        if len(word_list1) == 0 or len(word_list2) == 0:
            return 0
        words = list(set(word_list1+word_list2))
        size = len(words)
        v = [0] * size
        for i in range(size):
            if words[i] in word_list1 and words[i] in word_list2:
                v[i] = 1
        sim = sum(v)
        dom = np.log(len(word_list1)) + np.log((len(word_list2)))
        if sim < 1e-12 or dom < 1e-12:
            return 0
        else:
            return sim / dom

    def weight_matrix(self):
        size = len(self.sentences)
        self.length = size
        source = self.word_list
        weight = np.zeros((size, size))
        for i in range(size):
            for j in range(i, size):
                dis = self.similarity(source[i], source[j])
                weight[i][j] = dis
                weight[j][i] = dis
        return weight

    def find_abstract(self):
        rs = []
        textrank = PageRank()
        graph = self.weight_matrix()
        textrank.weight_to_rank(graph)

        # for row in graph:
        #     print('sel', row)
        # factor = {'alpha': 0.85, }
        # nx_graph = nx.from_numpy_matrix(graph)
        # scores = nx.pagerank(nx_graph, **factor)  # this is a dict
        # sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        # rs = list()
        # for index, score in sorted_scores:
        #     rs.append((index, score, self.sentences[index]))

        res = textrank.page_rank(100)
        for k in range(res.shape[0]):
            rs.append([k, res[k], self.sentences[k]])
        rs.sort(key=lambda rs: rs[1], reverse=True)
        # rs is a list, each element is consist of (index, text rank, sentence).
        self.key_sentence = rs
        self.text_rank = textrank

    def visual(self):
        """
        to make the ranked result be visualized, sort the key_sentence in order 'name weight'
        the return value could be used by some tools (ex. gephi)
        """
        G = nx.Graph()
        G.add_nodes_from(self.sentences)
        graph = self.text_rank.rank_m
        for i in range(graph.shape[0]):
            for j in range(graph.shape[1]):
                G.add_edges_from((i, j, graph[i][j]))
        # print(G.nodes)
        # print(G.edges)

        pass






class PageRank(object):

    def __init__(self, rank_matrix=None):
        self.rank_m = rank_matrix
        self.factor = 0.85

    def page_rank(self, num, factor=None):
        ini = np.array([1] * self.rank_m.shape[0])
        if factor is None:
            factor = self.factor
        i_f = 1 - factor
        err = ini
        for n in range(num):
            temp = i_f * ini + factor * np.dot(self.rank_m, ini)
            err = temp - ini
            ini = temp
            if np.dot(err, err.T) < 1e-6:
                # print('Iteration number: %d' % n)
                return ini
        raise Exception('Does not convergence!', err)

    def weight_to_rank(self, weight_m):
        size = weight_m.shape[0]
        rank_m = np.array(weight_m)
        for i in range(size):
            a = sum(weight_m[:, i])
            if a != 0:
                rank_m[:, i] = weight_m[:, i] / a
        self.rank_m = rank_m

'''*************************更改编码方式*******************************'''

def write(first, text, last):
    name = '概要.txt'

    f = open(name, 'w', buffering=1)
    # f = open(name, 'w', buffering=1,encoding='utf8')

    j = 0
    f.write('下面是一则新闻简讯： \n\n')
    f.write(first+ '。 \n')
    for i in text[:]:
        f.write(i + '。')
    f.write('\n')
    f.write(last+'。 \n')
    f.close()

def main():
    '''news name inport'''

    path = r'C:\Users\machenike\Desktop\news.txt'#  修改标题  #
    article = input(path)
    ab = Abstract()
    ab.read_text(article.sentences, article.words_all_filters)
    ab.find_abstract()
    size = int(ab.length/4)
    import operator
    ab.key_sentence.sort(key=operator.itemgetter(0))
    document = ab.key_sentence
    # print(document)
    _text = []
    for item in document[0:size]:
        _text.append(item[2])
    text = _text[1:]
    write(ab.first, text, ab.last)

if __name__ == '__main__':
    main()
    print('新闻缩写撰写完毕')
