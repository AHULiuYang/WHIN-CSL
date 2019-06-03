import unittest

from main import Main


class main_test(unittest.TestCase):

    def setUp(self):

        print("setUp")

    def test_w2v(self):

        m = Main()
        m.w2v()

    def test_d2v(self):

        m = Main()
        m.d2v()

    def test_infer_abstract_vec(self):

        m = Main()
        m.infer_abstract_vec()

    def test_calculate_sim_on_abstract(self):

        m = Main()
        m.calculate_sim_on_abstract(model="doc2vec",
                                    t="all",
                                    sim_method="cos")

    def test_build_graph(self):

        m = Main()
        m.build_graph(doc_rec_tok_k=10,
                      semantic_linking=True,
                      weighted=True,
                      authored=True)

    def test_n2v(self):

        m = Main()
        m.n2v()

    def test_citation_recommendation(self):

        m = Main()
        m.citation_recommendation(model="node2vec", t="test", sim_method="cos")

    def test_evaluete(self):

        m = Main()
        m.evaluete(model="node2vec")


if __name__ == '__main__':

    unittest.main()
