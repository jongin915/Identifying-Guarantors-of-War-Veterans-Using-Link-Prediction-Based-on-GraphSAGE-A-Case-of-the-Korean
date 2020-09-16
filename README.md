# Identifying-Guarantors-of-War-Veterans-Using-Link-Prediction-Based-on-GraphSAGE-A-Case-of-the-Korean

Python==3.6.8, Stellargraph==1.2.1, Tensorflow==2.1.0, linkpred==0.5.1.

Purpose
This study proposes a framework that recommends guarantors to veterans using a graph neural network-based link prediction method. Many veterans can't get various benefits and welfare due to loss of service record. The service record can be replaced by several documents. In such cases, a “buddy statement” can play an important role. However, many veterans have difficulty to find someone who can vouch for their participation in the war. To solve this problem, this study proposes a combined operations network in which veterans can find guarantors who participated in the same battle. As it is difficult to find a guarantor directly for reasons such as death, we designed a link prediction-based approach to identify highly relevant guarantors in this network. We applied our proposed approach to Korean War data to train the combined operations network with GraphSAGE by sampling neighbors and using various kinds of aggregation functions. Furthermore, we evaluated this approach by comparing with the various other approaches.

Dataset
A dataset was extracted from The War Memorial of the Korea(https://www.warmemo.or.kr/front/militaryInfo/warDeadSearch.do) 
and Cho, S. H.; Son, K. S.; Park, J. S.; Lee, S. H.; Bae, A. S.; Lee, G. S.; and Kim, H. G. 2017. The Korean War: Major Battles. Institute for Military History, Seoul, Korea.
Nodes represent each Army units that participated in operations.
Edges are the number of combined operations that neighbor nodes participated in together.
The meaning of node features as follows.
- division_cap : whether the node belongs to the captial division
- division_i : whether the node belongs to the division i
- regiment_i : whether the node belongs to the regiment i
- period_k : whether the node participated in period k
- KIA : the number of killed in action of the node

Experiment Setup
In our experiment, we implement our model in Tensorflow. To evaluate the link prediction task in our war dataset, we separate the dataset into training and validation datasets. 
Moreover, the same number of positive and negative node pairs are extracted. In detail, we separate the edges of the war dataset by 90:10. 
Ten percent of the edges are regarded as positive pairs in the validation dataset, while the rest of the edges are considered those in the training dataset. 
Negative pairs are randomly sampled, and the number of negative pairs is the same as the positive pairs for each training and validation dataset. 
For baselines, we divide the entire edges into the training set and validation set by 90:10. The GraphSAGE-based models are conducted with the Adam optimizer (Kingma and Ba 2014). 
We use binary cross-entropy loss for supervised learning. 
We also compare three variants of aggregator functions for the GraphSAGE as mean aggregator, mean-pool aggregator, and max-pool aggregator, except the LSTM aggregator because LSTM architecture processes inputs in sequence.
There are various hyperparameters that need to be set to use GraphSAGE. 
First, we set the depth K=2 as recommended by Hamilton, Ying, and Leskovec (2017) with a training batch size of 256. 
We perform parameter sweeps over learning rates {0.01, 0.001, 0.0001} as the step size for minimizing loss; 1-hop and 2-hop sample sizes are both {5, 10, 15}. 
As every node in our war dataset has at least 15 neighbors, we set the sample sizes as less than or equal to 15. Moreover, two cases of dropout probability are set as {0.3, 0.5}. 
The optimal hyperparameters are obtained by a grid search. We conduct each experiment 10 times with random seeds and compare the average performance. 
The best performance is obtained when learning rate = 0.01, 1-hop sample size = 15, 2-hop sample size = 15, and dropout = 0.3. 
For GCN, we use a softmax function for activation and train 300 epochs using Adam optimizer with the learning rate of 0.01 as Kipf and Welling (2016) recommended.


References
Adamic, L. A., and Adar, E. 2003. Friends and neighbors on the Web. Social Networks 25(3): 211-230
Casler, M.; Fosmire, A.; and Klein, D. 2019. Support for Veterans: Community Reintegration, Family and Mental Health Needs (Master’s Thesis). Utica College, ProQuest Dissertations & Theses Global 59.
Cho, S. H.; Son, K. S.; Park, J. S.; Lee, S. H.; Bae, A. S.; Lee, G. S.; and Kim, H. G. 2017. The Korean War: Major Battles. Institute for Military History, Seoul, Korea
Fout, A.; Byrd J.; Shariat, B.; and Ben-Hur, A. 2017. Protein Interface Prediction using Graph Convolutional Network. Proceedings of the 31st International Conference on Neural Information Processing Systems 30: 6533-6542.
Hamilton, W. L.; Ying, R.; and Leskovec, J. 2017. Inductive Representation Learning on Large Graphs. Neural Information Processing Systems 2017: 1024–1034.
Jaccard, P. 1901. Etude de la distribution florale dans une portion des Alpes et du Jura. Bulletin de la Societe Vaudoise des Sciences Naturelles 37(142): 547-579
Jeong, M. K., and Kim, S. Y. 2018. Korean War Veterans Experience of War and Meaning in Life. Critical Social Welfare Academy (58): 243-278.
Kang, H. K.; Natelson, B. H.; Clare M.; Mahan, C. M.; Lee, M. Y.; and Murphy, F.M. 2003. Post-Traumatic Stress Disorder and Chronic Fatigue Syndrome-like Illness among Gulf War Veterans: A Population-based Survey of 30,000 Veterans. American Journal of Epidemiology 157(2): 141-148.
Kim, T. S.; Lee, W. K.; and Sohn, S. Y. 2019. Graph convolutional network approach applied to predict hourly bike-sharing demands considering spatial, temporal, and global effects. PloS one 14(9). e0220782.
Kingma, D. P., and Ba, J. 2014. Adam: A Method for Stochastic Optimization. arXiv preprint arXiv:1412.6980.
Kipf, T. N., and Welling, M. 2016. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations (ICLR).
Min, H. K. 2009. Korean War History: Battle of Inchon and Counterattack. Institute for Military History, Seoul, Korea
Nam, K. K. 2013. The Necessity and It’s Meaning of Compensation for Irregulars Contributor in the Korean War. Korean Association of Unification Strategy Unification Strategy 13(3): 9-32.
Nelson, C. B.; Abraham, K. M.; Miller, E. M.; Kees, M. R.; Walters, H. M.; Valenstein, M.; and Zivin, K. 2015. Veteran Mental Health and Employment: The Nexus and Beyond. War and Family Life: 239-260.
Park, D. C. 2014. The Korean War in statistics. Institute for Military History, Seoul, Korea
Ra, M. K. 2017. Korean Veterans' Policy through the Case of the Korean War Veterans. Journal of Patriots and Veterans Affairs in the Republic of Korea 16(4): 37-62. 
Scarselli, F.; Gori, M.; Tsoi, A.C.; Hagenbuchner, M.; and Monfardii, G. 2009. The Graph Neural Network Model.  IEEE Trasactions on Neural Networks 20(1): 61–80.
Tao, Z.; Linyuan, L.; and Yi-Cheng, Z. 2009. Predicting missing links via local information. The European Physical Journal B 71: 623–630
Veličković, P.; Cucurull, G.; Casanova, A.; Romero, A.; Liò, P.; and Bengio, Y. 2018. Graph Attention Networks. In International Conference on Learning Representations (ICLR)
Walter, W. S., and Evans, W. 1974. The National Personnel Records Center (NPRC) Fire: A Study in Disaster. THE AMERICAN ARCHIVIST 37(4): 521-549.
Williamson, V.; Stevelink, S.; Greenberg, K.; and Greenberg, N. 2018. Prevalence of Mental Health Disorders in Elderly U.S. Military Veterans: A Meta-Analysis and Systematic Review. The American Journal of Geriatric Psychiatry 26(5): 534-545.
Xu, K.; Hu, W.; Leskovec, J.; and Jegelka, S. 2019. How Powerful are Graph Neural Networks. In International Conference on Learning Representations (ICLR)
Yao, L.; Mao, C.; and Luo, Y. 2018. Graph convolutional networks for text classification. In Association for the Advancement of Artificial Intelligence (AAAI).


