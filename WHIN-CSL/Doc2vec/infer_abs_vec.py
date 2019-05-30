import gensim.models as g
from common import Load,DatasetPaths
import codecs
import json

class Infer_abstract_vec:

    def __init__(self,**kwargs):
        self.dp = DatasetPaths(kwargs["experiment_data"])
        self.l = Load(kwargs["experiment_data"])
        self.paperid_abstract2vec_dict = {}
        self.m = g.Doc2Vec.load(self.dp.DOC2VEC)
        self.__abstract2vec(**kwargs)
        self.__save()

    def __abstract2vec(self, **kwargs):
        print("start infer abstract vector...")
        kwargs.pop("experiment_data")
        i = 0
        for paper_id in range(self.l.papers_all_num):
            abstract = self.l.abstract(paper_id).strip().split()
            self.paperid_abstract2vec_dict[paper_id] = " ".join([str(x) for x in self.m.infer_vector(abstract,
                                                                                                     **kwargs)])
            i += 1
            if i % 1000 == 0:
                print("finish [%d]/[%d]" % (i, self.l.papers_all_num))

    def __save(self):
        print("save abstract vector to ",self.dp.ABSTRACT_VEC)
        with codecs.open(self.dp.ABSTRACT_VEC, "w", "utf8") as f_o:
            json.dump(self.paperid_abstract2vec_dict,f_o,
                      indent=2,
                      ensure_ascii=False)

