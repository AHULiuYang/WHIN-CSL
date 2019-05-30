from common import Parameter
from Doc2vec.word2vec import word2vec
from Doc2vec.doc2vec import doc2vec
from common import DatasetPaths
from Doc2vec.infer_abs_vec import Infer_abstract_vec
from recommendation import Recommend
from build_graph import Build_graph
from node2vec import node2vec
from evaluate import Evaluete

class Main:

    def __init__(self):
        self.experiment_data = "aan"
        self.p = Parameter(self.experiment_data)
        self.r = Recommend(self.experiment_data)

    def w2v(self):
        word2vec(**self.p.w2v_config())

    def d2v(self):
        doc2vec(**self.p.d2v_config())

    def infer_abstract_vec(self):
        Infer_abstract_vec(**self.p.infer_config())

    def calculate_sim_on_abstract(self,model,t,sim_method):
        self.r.recommend(model=model,t=t,sim_method=sim_method)

    def build_graph(self,doc_rec_tok_k,semantic_linking,weighted,authored):
        Build_graph(self.experiment_data).build_graph(doc_rec_tok_k=doc_rec_tok_k,
                                                      semantic_linking=semantic_linking,
                                                      weighted=weighted,
                                                      authored=authored)

    def n2v(self):
        node2vec.start(**self.p.node2vec_config())

    def citation_recommendation(self,model,t,sim_method):
        self.r.recommend(model=model, t=t, sim_method=sim_method)

    def evaluete(self,model):
        Evaluete(experiment_data=self.experiment_data, model=model)





