# WHIN-CSL

A python 3.5 implemention of WHIN-CSL proposed in paper "Citation Recommendation Based on Weighted Heterogeneous Information Network Containing Semantic Linking", ICME 2019 [paper](http://www.pdf-express.org/Conf/44524XP/versions/981188/PID5863305.pdf). The implementation is flexible enough for modifying the model or fit your own datasets.

Requirements: This package is developed with Python 3.5, please make sure all the demendencies are installed, which is specified in requirements.txt.

```
-ATTN: This package is free for academic usage. You can run it at your own risk. 
For other purposes, please contact Prof. Shu Zhao(zhaoshuzs2002@hotmail.com)
-ATTN2: This package was developed by Mr.Yang Liu(liuy_anhui@163.com). 
The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Liu.
-ATTN3: This package was provided by Lab of Intelligent computing and knowledge engineering, School of computer science and technology, Anhui University
```

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

