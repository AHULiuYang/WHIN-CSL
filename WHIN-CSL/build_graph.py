from common import Load, DatasetPaths


class Build_graph():

    def __init__(self, experiment_data: str):
        self.experiment_data = experiment_data
        self.l = Load(self.experiment_data)
        self.dp = DatasetPaths(self.experiment_data)
        self.author_init_num = 20000

    def build_graph(self, doc_rec_tok_k=10, semantic_linking=True, weighted=True, authored=True):
        graph = self.l.load_recommend_result_top_k("doc2vec", doc_rec_tok_k)

        f_out = open(self.dp.GRAPH, "w")

        for k, v in graph.items():

            if semantic_linking:
                max = 1.0
                for p in v:
                    if max <= 0:
                        max = 0.01
                    if weighted:
                        f_out.write(" ".join(map(str, [k, p])) + " " + str(max) + "\n")
                    else:
                        f_out.write(" ".join(map(str, [k, p])) + "\n")
                    max = max - 1.0 / doc_rec_tok_k
            else:
                paper_venue = self.l.venue(k)
                if weighted:
                    f_out.write(" ".join(map(str, [k, paper_venue])) + " " + str(1.0) + "\n")
                else:
                    f_out.write(" ".join(map(str, [k, paper_venue])) + "\n")

            if self.l.is_train(k):
                for citation in self.l.citations(k):
                    if weighted:
                        f_out.write(" ".join(map(str, [k, citation])) + " " + str(1.0) + "\n")
                    else:
                        f_out.write(" ".join(map(str, [k, citation])) + "\n")

            if authored:
                paper_authors = self.l.authors(k)
                max = 1.0
                for a in paper_authors:
                    a = int(a) + self.author_init_num
                    if max <= 0:
                        max = 0.01
                    if weighted:
                        f_out.write(" ".join(map(str, [k, a])) + " " + str(max) + "\n")
                    else:
                        f_out.write(" ".join(map(str, [k, a])) + "\n")
                    max = max - 1.0 / len(paper_authors)

                author_author = self.__get_a_a(paper_authors)
                for a_a in author_author:
                    if weighted:
                        f_out.write(" ".join(map(str, [a_a[0], a_a[1]])) + " " + str(1.0) + "\n")
                    else:
                        f_out.write(" ".join(map(str, [a_a[0], a_a[1]])) + "\n")

        print("finish build graph on [%s]" % self.experiment_data)

    def __get_a_a(self, v: list) -> list:
        v = v
        a_a = []
        while True:
            if len(v) <= 1:
                return a_a
            p = v.pop()
            for q in v:
                a_a.append([int(p) + self.author_init_num, int(q) + self.author_init_num])
