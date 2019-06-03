import json, multiprocessing
from util import cur_file_dir


class DatasetPaths(object):

    def __init__(self, experiment_data: str):

        self.BASE_DIR = cur_file_dir().split("WHIN-CSL")[0] + "WHIN-CSL/data/"
        self.ALL_DATA = self.BASE_DIR + experiment_data + "/ALL_EXPERIMENT_DATA"
        self.WORD2VEC = self.BASE_DIR + experiment_data + "/WORD2VEC"
        self.DOC2VEC = self.BASE_DIR + experiment_data + "/DOC2VEC"
        self.ABSTRACT = self.BASE_DIR + experiment_data + "/ABSTRACT"
        self.ABSTRACT_VEC = self.BASE_DIR + experiment_data + "/ABSTRACT_VEC"
        self.GRAPH = self.BASE_DIR + experiment_data + "/GRAPH"
        self.NODE2VEC_VEC = self.BASE_DIR + experiment_data + "/NODE2VEC_VEC"
        self.NODE2VEC_RECOMMEND_RESULT = self.BASE_DIR + experiment_data + "/NONE2VEC_RECOMMEND_RESULT"
        self.DOC2VEC_RECOMMEND_RESULT = self.BASE_DIR + experiment_data + "/DOC2VEC_RECOMMEND_RESULT"


class Parameter(DatasetPaths):

    def __init__(self, experiment_data: str):

        self.experiment_data = experiment_data
        super(Parameter, self).__init__(experiment_data)

    def infer_config(self) -> dict:

        return {
            "experiment_data": self.experiment_data,
            "alpha": 0.025,
            "steps": 3
        }

    def w2v_config(self) -> dict:

        return {
            "experiment_data": self.experiment_data,
            "size": 200,
            "window": 10,
            "negative": 5,
            "min_count": 3,
            "sg": 1,
            "hs": 0,
            "iter": 50,
            "workers": multiprocessing.cpu_count()
        }

    def d2v_config(self) -> dict:

        return {
            "experiment_data": self.experiment_data,
            "size": 200,
            "window": 10,
            "min_count": 1,
            "sample": 1e-3,
            "negative": 5,
            "iter": 50,
            "dm": 0,
            "workers": multiprocessing.cpu_count(),
            "hs": 0,
            "dm_concat": 1,
            "dbow_words": 1
        }

    def node2vec_config(self) -> dict:

        return {
            "experiment_data": self.experiment_data,
            "dimensions": 200,
            "walk_length": 40,
            "num_walks": 10,
            "window_size": 20,
            "iter": 5,
            "hs": 1,
            "sg": 1,
            "workers": multiprocessing.cpu_count(),
            "p": 1,
            "q": 1,
            "weighted": True,
            "directed": False,
            "input": self.GRAPH,
            "output": self.NODE2VEC_VEC
        }


class DataCache:

    def __init__(self, experiment_data: str):

        self.experiment_data = experiment_data
        self.__load()

    def __load(self):

        with open(DatasetPaths(self.experiment_data).ALL_DATA, encoding="utf8") as f:
            self.all_data = json.loads(f.read())

    @property
    def papers_all_num(self) -> int:

        return self.all_data["papers_all_num"]

    @property
    def authors_num(self) -> int:

        return self.all_data["authors_num"]

    @property
    def venues_num(self) -> int:

        return self.all_data["venues_num"]

    @property
    def papers_test_num(self) -> int:

        return self.all_data["papers_test_num"]

    def is_train(self, paper_id) -> bool:

        return self._query("paper_is_train", paper_id)

    def citations(self, paper_id) -> list:

        return self._query("paper_citations", paper_id)

    def outcite_num(self, paper_id) -> int:

        return self._query("paper_outcite_num", paper_id)

    def title(self, paper_id) -> str:

        return self._query("paper_title", paper_id)

    def abstract(self, paper_id) -> str:

        return self._query("paper_abstract", paper_id)

    def title_raw(self, paper_id) -> str:

        return self._query("paper_title_raw", paper_id)

    def abstract_raw(self, paper_id) -> str:

        return self._query("paper_abstract_raw", paper_id)

    def venue(self, paper_id) -> int:

        return self._query("paper_venue", paper_id)

    def authors(self, paper_id) -> list:

        return self._query("paper_authors", paper_id)

    def authors_raw(self, paper_id) -> str:

        return self._query("paper_authors_raw", paper_id)

    def index(self, paper_id) -> str:

        return self._query("paper_index", paper_id)

    def year(self, paper_id) -> int:

        return self._query("paper_year", paper_id)

    def _check_paper_id(self, paper_id):

        if isinstance(paper_id, int):
            return str(paper_id)

        return paper_id

    def _query(self, type: str, paper_id):

        return self.all_data[type][self._check_paper_id(paper_id)]


class Vector:

    def __init__(self, experiment_data: str, model: str = "doc2vec"):

        self.experiment_data = experiment_data
        self._load(model)

    def _load(self, model:str):

        if model == "doc2vec":

            vec_p = DatasetPaths(self.experiment_data).ABSTRACT_VEC
            with open(vec_p, encoding="utf8") as f:
                self.paperid_vec = json.loads(f.read())

        elif model == "node2vec":

            self.paperid_vec = {}
            vec_p = DatasetPaths(self.experiment_data).NODE2VEC_VEC

            with open(vec_p, encoding="utf8") as f:
                for line in f:
                    line = line.strip("\n").split()
                    if len(line) == 2:
                        continue
                    else:
                        self.paperid_vec[line[0]] = " ".join(line[1:])


class Recommend_result:

    def __init__(self, experiment_data: str, model: str):

        self.experiment_data = experiment_data
        self.__load(model)

    def __load(self, model):

        with open(getattr(DatasetPaths(self.experiment_data), model.upper() + "_RECOMMEND_RESULT"),
                  encoding="utf8") as f:
            self._recommend_result = json.loads(f.read())


class Load(DataCache,
           Vector,
           Recommend_result):

    def __init__(self, experiment_data: str):

        self.experiment_data = experiment_data
        super(Load, self).__init__(experiment_data)

    @property
    def all_abstract(self) -> list:

        data = [self.abstract(i).split() for i in range(self.papers_all_num)]
        print("load abstract num:", len(data))
        return data

    def vec(self, model: str = "doc2vec") -> dict:

        super(DataCache, self).__init__(self.experiment_data, model=model)
        print("load [%s] dataset [%s] vectors num: [%d]"
              % (self.experiment_data, model, len(self.paperid_vec)))

        return {str(k): [float(ve) for ve in v.split()]
                for k, v in self.paperid_vec.items()}

    def load_recommend_result_top_k(self, model: str, top_k: int) -> dict:

        super(Vector, self).__init__(self.experiment_data, model=model)
        print("load [%s] dataset [%s] recommend result num: [%d]"
              % (self.experiment_data, model, len(self._recommend_result)))

        return {k: [x.split(" ")[0] for x in v["recommend"][1:top_k + 1]]
                for k, v in self._recommend_result.items()}
