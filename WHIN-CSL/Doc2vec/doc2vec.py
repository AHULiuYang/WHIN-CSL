# coding=utf-8
import gensim.models as g
from common import DatasetPaths, Load
import logging, time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class doc2vec:

    def __init__(self, **kwargs):
        self.dp = DatasetPaths(kwargs["experiment_data"])
        self.l = Load(kwargs["experiment_data"])

        print("start train doc2vec...")
        self.__save_abstract()
        kwargs.pop("experiment_data")
        self.model = g.Doc2Vec(g.doc2vec.TaggedLineDocument(self.dp.ABSTRACT),
                               pretrained_emb=self.dp.WORD2VEC,
                               **kwargs)

        print("save doc2vec model to ", self.dp.DOC2VEC)
        self.model.save(self.dp.DOC2VEC)

    def __save_abstract(self):
        print("save abstract to ", self.dp.ABSTRACT)
        with open(self.dp.ABSTRACT, "w", encoding="utf8") as f:
            for l in self.l.all_abstract:
                f.write(" ".join(l) + "\n")
        time.sleep(3)
