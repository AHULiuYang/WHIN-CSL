import math
from common import DatasetPaths

class Evaluete(DatasetPaths):

    def __init__(self,experiment_data:str,model:str):
        self.experiment_data = experiment_data
        print("calculate recall and ndcg on [%s] by [%s]"%(experiment_data,model))
        super(Evaluete,self).__init__(experiment_data)
        self.ks = ["10", "20", "25", "50", "75", "100", "150", "200"]
        self._load_rec_result(model)
        self.evaluete()

    def _load_rec_result(self,model:str):

        with open(eval("self." + model.upper() + "_RECOMMEND_RESULT"), encoding="utf8") as f:
            import json
            papers = json.loads(f.read())

        for r in self.ks:
            setattr(self,"recall_list_" + r,{})

        def count(num:str) ->dict:

            v = {"right":[],
               "real_recommendation":[int(x.split()[0]) for x in paper["recommend"]][:int(num)],
               "references":paper["references"]}
            for ref in paper["references"]:
                recommends = [int(x.split()[0]) for x in paper["recommend"]]
                if ref in recommends[:int(num)]:
                    v["right"].append(ref)
            return v

        for k in papers.keys():
            paper = papers[k]
            if len(paper["references"]) == 0:
                continue
            if self.experiment_data == "aan":
                if paper["year"] != 2012:
                    continue
            elif self.experiment_data == "dblp":
                if paper["year"] != 2008:
                    continue
            for r in self.ks:
                getattr(self, "recall_list_" + r)[k] = count(r)

    def __get_average(self,list:list)->float:
        sum = 0
        for item in list:
            sum += item
        return sum / len(list)


    def __recall(self):
        for r in self.ks:
            recall = self.__get_average([v["right"].__len__() / v["references"].__len__()
                       for k,v in getattr(self, "recall_list_" + r).items()])
            print("recall" + r,recall)


    def __ndcg(self):

        def get_one_ndcg(right_l, rec_l,rr_l):

            right_l = [int(x) for x in right_l]
            rec_l = [int(x) for x in rec_l]

            if len(rr_l) == 0:
                return 0.
            dcg = 0.
            idcg = 1.
            for i in range(1,len(rr_l)):
                idcg += 1 / math.log(i + 1, 2)
            if rec_l[0] in right_l:
                dcg = 1.
            for i in range(1,len(rec_l)):
                if rec_l[i] in right_l:
                    dcg += 1 / math.log(i + 1, 2)
            return dcg / idcg

        for r in self.ks:
            ndcg = self.__get_average([get_one_ndcg(v["references"],v["real_recommendation"],v["right"])
                                                 for k, v in getattr(self, "recall_list_" + r).items()])
            print("ndcg" + r, ndcg)

    def evaluete(self):

        self.__recall()
        self.__ndcg()

