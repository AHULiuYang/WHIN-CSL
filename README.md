# WHIN-CSL

Source code for ICME-2019 paper "Citation Recommendation Based on Weighted Heterogeneous Information Network Containing Semantic Linking"

## Requirements

1. Python 3.5
2. gensim=3.4.0
3. NLTK=3.2.1
4. numpy=1.11.3
5. networkx=1.11

## Usage

Scripts in main_test.py

	-test_w2v <method>
		Run word2vec on papers' abstracts
	-test_d2v <method>
		Run doc2vec on papers' abstracts
	-test_infer_abstract_vec <method>
		Run infer vectors of papers' abstracts
	-test_calculate_sim_on_abstract <method>
		Calculate semantic links based on similarity scores between papers' abstracts
	-test_build_graph <method>
		Construct weighted heterogeneous information network containing semantic linking
	-test_n2v <method>
		Run network embedding
	-test_citation_recommendation <method>
		Run citation recommendation based on the learned network embeddings
	-test_evaluete <method>
		Run evaluate for computing Recall and NDCG

​                                                                                                                                                                                                            Please open a issue or connect the email (liuy_anhui@163.com) if you have any questions/problems, I will try my best to assist you.

## WHIN-CSL bibtex information

    @inproceedings{WHIN-CSL:ICME19,
     title={Citation Recommendation Based on Weighted Heterogeneous Information Network Containing Semantic Linking},
     author = {Jie, Chen and Yang, Liu and Shu, Zhao and Yanping, Zhang},
     booktitle = {ICME},
     year = {2019},
    } 

