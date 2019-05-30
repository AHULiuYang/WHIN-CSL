from common import Load, DatasetPaths
import json, tqdm
from util import similarity


class Recommend:

    def __init__(self, experiment_data: str):
        self.sm = similarity()
        self.l = Load(experiment_data)
        self.dp = DatasetPaths(experiment_data)
        self.author_init_num = 20000

    def recommend(self, model: str = "doc2vec", t: str = "all", sim_method: str = "cos"):

        recommend_result = {}
        print("test papers num:", self.l.papers_test_num)
        self.vec = self.l.vec(model=model)
        no_paper_vec = [i for i in range(self.l.papers_all_num) if str(i) not in self.vec.keys()]
        print("no vector papers", no_paper_vec)

        for tp_id in tqdm.tqdm(range(self.l.papers_all_num)):
            tp_id = str(tp_id)

            if self.l.is_train(tp_id) and t == "test":
                continue
            recommend_result[tp_id] = {
                "references": self.l.citations(tp_id),
                "recommend": [],
                "year": self.l.year(tp_id),
                "out_citation_count": len(self.l.citations(tp_id))
            }
            sims = {}
            for cp_id in self.vec.keys():

                if int(cp_id) >= self.author_init_num:
                    continue
                if int(self.l.year(tp_id)) <= int(self.l.year(cp_id)) and model == "node2vec":
                    continue

                sim_pp = 0.
                if sim_method == "cos":
                    sim_pp = self.sm.cos(self.vec[tp_id], self.vec[cp_id])
                elif sim_method == "Manhattan_Distance":
                    sim_pp = self.sm.Manhattan(self.vec[tp_id], self.vec[cp_id])
                elif sim_method == "Chebyshev":
                    sim_pp = self.sm.Chebyshev(self.vec[tp_id], self.vec[cp_id])
                elif sim_method == "MahalanobisDistance":
                    sim_pp = self.sm.Mahalanobis(self.vec[tp_id], self.vec[cp_id])
                elif sim_method == "Pearson":
                    sim_pp = self.sm.Pearson(self.vec[tp_id], self.vec[cp_id])

                if model == "doc2vec":
                    sims[cp_id] = sim_pp
                elif model == "node2vec":
                    # sim_pa = self.__sim_paper_author(tp_id,self.l.authors(cp_id))
                    sims[cp_id] = sim_pp
            recommend_result[tp_id]["recommend"] = self.__get_top_200(sims)
        self.__save(recommend_result, model)

    def __sim_paper_author(self, tp_id: str, authors: list):
        sim_pas = []
        for a in authors:
            a = int(a) + self.author_init_num
            sim_pa = self.sm.cos(self.vec[tp_id], self.vec[a])
            sim_pas.append(sim_pa)
        return max(sim_pas)

    def __save(self, recommend_result: dict, model: str):
        with open(eval("self.dp." + model.upper() + "_RECOMMEND_RESULT"), "w", encoding="utf8") as f:
            json.dump(recommend_result, f, indent=2, ensure_ascii=False)

    def __get_top_200(self, paper_simil_dict: dict) -> list:
        top_all = sorted(paper_simil_dict.items(), key=lambda x: x[1], reverse=True)[:201]
        return [" ".join(map(str, x)) for x in top_all]
