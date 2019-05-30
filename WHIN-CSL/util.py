import math,sys,os,nltk,json,time,re
import numpy as np

def chunks_by_average(tasks:list,m:int):
    n = int(math.ceil(len(tasks) / float(m)))
    return [tasks[i:i+n] for i in range(0,len(tasks),n)]

def cur_file_dir():
    path = sys.path[0]
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        return os.path.dirname(path)

def clear_text(line,clear_stop = True):
    ret_list = False
    if isinstance(line,list):
        line = " ".join(line)
        ret_list = True
    line = line.lower()
    p1 = re.compile(r'-\{.*?(zh-hans|zh-cn):([^;]*?)(;.*?)?\}-')
    p2 = re.compile(r'[(][: @ . , ？！\s][)]')
    p3 = re.compile(r'[「『]')
    p4 = re.compile(r'[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）0-9 , : ; \-\ \[\ \]\ ]')
    line = p1.sub(r' ', line)
    line = p2.sub(r' ', line)
    line = p3.sub(r' ', line)
    line = p4.sub(r' ', line)
    line = " ".join(line.split())
    if clear_stop:
        from nltk.corpus import stopwords
        english_stopwords = stopwords.words('english')
        line = " ".join([word for word in line.split() if not word in english_stopwords])

    # from nltk.stem.lancaster import LancasterStemmer
    # st = LancasterStemmer()
    # line = " ".join([st.stem(word) for word in line.split()])

    porter2 = nltk.stem.WordNetLemmatizer()
    line = [porter2.lemmatize(x) for x in nltk.word_tokenize(line)]
    line = [i for i in line if len(i)>1]
    if not ret_list:
        return " ".join(line)
    return line


class similarity:

    def cos(self,v_0, v_1):
        v1 = np.array(v_0)
        v2 = np.array(v_1)
        d = np.matmul(v1, v2)
        na = np.sum(np.square(v1))
        nb = np.sum(np.square(v2))
        return d / ((na * nb) ** 0.5)

    def Manhattan(self,vec1, vec2):
        npvec1, npvec2 = np.array(vec1), np.array(vec2)
        return np.abs(npvec1 - npvec2).sum()

    def Chebyshev(self,vec1, vec2):
        npvec1, npvec2 = np.array(vec1), np.array(vec2)
        return max(np.abs(npvec1 - npvec2))

    def Mahalanobis(self,vec1, vec2):
        npvec1, npvec2 = np.array(vec1), np.array(vec2)
        npvec = np.array([npvec1, npvec2])
        sub = npvec.T[0] - npvec.T[1]
        inv_sub = np.linalg.inv(np.cov(npvec1, npvec2))
        return math.sqrt(np.dot(inv_sub, sub).dot(sub.T))

    def Pearson(self,p, q):
        same = 0
        for i in p:
            if i in q:
                same += 1
        n = same
        sumx = sum([p[i] for i in range(n)])
        sumy = sum([q[i] for i in range(n)])
        sumxsq = sum([p[i] ** 2 for i in range(n)])
        sumysq = sum([q[i] ** 2 for i in range(n)])
        sumxy = sum([p[i] * q[i] for i in range(n)])
        up = sumxy - sumx * sumy / n
        down = ((sumxsq - pow(sumxsq, 2) / n) * (sumysq - pow(sumysq, 2) / n)) ** .5
        if down == 0: return 0
        r = up / down
        return r

