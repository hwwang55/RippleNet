# RippleNet

This repository is the implementation of RippleNet ([arXiv](https://arxiv.org/abs/1803.03467)):
> RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems  
Hongwei Wang, Fuzheng Zhang, Jialin Wang, Miao Zhao, Wenjie Li, Xing Xie, Minyi Guo  

![](https://github.com/hwwang55/RippleNet/blob/master/framework.jpg)

RippleNet is a deep end-to-end model that naturally incorporates the knowledge graph into recommender systems.
Ripple Network overcomes the limitations of existing embedding-based and path-based KG-aware recommendation methods by introducing preference propagation, which automatically propagates users' potential preferences and explores their hierarchical interests in the KG.



### Files in the folder

- `data/`
  - `book/`
    - `BX-Book-Ratings.csv`: raw rating file of Book-Crossing dataset;
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
  - `movie/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg_part1.txt` and `kg_part2.txt`: knowledge graph file;
    - `ratrings.dat`: raw rating file of MovieLens-1M;
- `src/`: implementations of RippleNet.



### Required packages
The code has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):
- tensorflow-gpu == 1.4.0
- numpy == 1.14.5
- sklearn == 0.19.1


### Running the code
```
$ cd src
$ python preprocess.py --dataset movie (or --dataset book)
$ python main.py --dataset movie (note: use -h to check optional arguments)
```