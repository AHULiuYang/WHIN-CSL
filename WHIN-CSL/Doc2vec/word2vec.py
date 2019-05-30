import logging
from common import DatasetPaths,Load
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class word2vec:

    def __init__(self,**kwargs):
        self.dp = DatasetPaths(kwargs["experiment_data"])
        self.l = Load(kwargs["experiment_data"])
        print("start train word2vec")
        kwargs.pop("experiment_data")
        self.model = Word2Vec(self.l.all_abstract,**kwargs)
        print("save word2vec to ",self.dp.WORD2VEC)
        self.model.wv.save_word2vec_format(self.dp.WORD2VEC)