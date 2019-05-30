import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import numpy as np
import networkx as nx
import random,logging
from gensim.models import Word2Vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Graph():
  def __init__(self, nx_G, is_directed, p, q):
    self.G = nx_G
    self.is_directed = is_directed
    self.p = p
    self.q = q

  def node2vec_walk(self, walk_length, start_node):

    G = self.G
    alias_nodes = self.alias_nodes
    alias_edges = self.alias_edges

    walk = [start_node]

    while len(walk) < walk_length:
      cur = walk[-1]
      cur_nbrs = sorted(G.neighbors(cur))
      if len(cur_nbrs) > 0:
        if len(walk) == 1:
          walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
        else:
          prev = walk[-2]
          next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
            alias_edges[(prev, cur)][1])]
          walk.append(next)
      else:
        break

    return walk

  def simulate_walks(self, num_walks, walk_length):

    G = self.G
    walks = []
    nodes = list(G.nodes())
    print ('Walk iteration:')
    for walk_iter in range(num_walks):
      print (str(walk_iter+1), '/', str(num_walks))
      random.shuffle(nodes)
      for node in nodes:
        walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

    return walks

  def get_alias_edge(self, src, dst):

    G = self.G
    p = self.p
    q = self.q

    unnormalized_probs = []
    for dst_nbr in sorted(G.neighbors(dst)):
      if dst_nbr == src:
        unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
      elif G.has_edge(dst_nbr, src):
        unnormalized_probs.append(G[dst][dst_nbr]['weight'])
      else:
        unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
    norm_const = sum(unnormalized_probs)
    normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

    return alias_setup(normalized_probs)

  def preprocess_transition_probs(self):

    G = self.G
    is_directed = self.is_directed

    alias_nodes = {}
    for node in G.nodes():
      unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
      norm_const = sum(unnormalized_probs)
      normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
      alias_nodes[node] = alias_setup(normalized_probs)

    alias_edges = {}
    triads = {}

    if is_directed:
      for edge in G.edges():
        alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
    else:
      for edge in G.edges():
        alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

    self.alias_nodes = alias_nodes
    self.alias_edges = alias_edges

    return


def alias_setup(probs):

  K = len(probs)
  q = np.zeros(K)
  J = np.zeros(K, dtype=np.int)

  smaller = []
  larger = []
  for kk, prob in enumerate(probs):
      q[kk] = K*prob
      if q[kk] < 1.0:
          smaller.append(kk)
      else:
          larger.append(kk)

  while len(smaller) > 0 and len(larger) > 0:
      small = smaller.pop()
      large = larger.pop()

      J[small] = large
      q[large] = q[large] + q[small] - 1.0
      if q[large] < 1.0:
          smaller.append(large)
      else:
          larger.append(large)

  return J, q

def alias_draw(J, q):

  K = len(J)

  kk = int(np.floor(np.random.rand()*K))
  if np.random.rand() < q[kk]:
      return kk
  else:
      return J[kk]


def start(**args):

    print("start node2vec on [%s] dataset"%args["experiment_data"])
    if args["weighted"]:
        G = nx.read_edgelist(args["input"], nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args["input"], nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args["directed"]:
        G = G.to_undirected()

    G = Graph(G, args["directed"], args["p"], args["q"])
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args["num_walks"], args["walk_length"])
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks,
                     size=args["dimensions"],
                     window=args["window_size"],
                     min_count=0,
                     sg=1,
                     workers=args["workers"],
                     iter=args["iter"],
                     hs=args["hs"])

    print("save node2vec vectors on [%s]"%args["output"])
    model.wv.save_word2vec_format(args["output"])
