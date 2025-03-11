# MHGTMDA: molecular heterogeneous graph transformer based on biological entity graph for miRNA-disease associations prediction
Drug target identification is a fundamental step in drug discovery and plays a pivotal role in the development of new therapies. Existing computational methods focus on the direct interactions between drugs and targets, often ignoring the complex interrelationships among drugs, targets and various biomolecules in the human system. To address this limitation, we propose a novel prediction model named DTGHAT (Drug and Target Association Prediction using Heterogeneous Graph Attention Transformer based on Molecular Heterogeneous). DTGHAT utilizes a graph attention transformer to identify novel targets from 15 heterogeneous drug-gene-disease networks characterized by chemical, genomic, phenotypic, and cellular networks. In a 5-fold cross-validation study, DTGHAT achieved an area under the receiver operating characteristic curve (AUC) of 0.9634, which is at least 4% higher than current state-of-the-art methods. Characterization ablation experiments highlight the importance of integrating biomolecular data from multiple sources in revealing drug-target interactions. In addition, a case study on cancer drugs further validates the effectiveness of DTGHAT in predicting novel drug target identification.

![Image text](https://github.com/stella-007/DTGHAT/blob/main/IMG/DTGHAT_00.png)

Overall architecture of DTGHAT. A. Data sources and some symbols in this study. B. Multi-molecule correlation graph. C. Multiple heterogeneous graph construction and multi-view graph attention network for graph topology feature extraction of Drugs and Proteins. D.  Multi-layer perceptron for training and prediction with attribute and graph topology features of Drugs and Proteins.
## Table of Contents
- [Installation](#installation)
- [Quick start](#quick-start)
- [Contributing](#contributing)
- [Cite](#cite)
- [Contacts](#contacts)
- [License](#license)

# Data description

| File name  | Description |
| ------------- | ------------- |
| drugs_node.csv    | drugs name file  |
| disease_node.csv  | disease name file   |
| protein_node.csv  | protein name file   |
| drug_disease_association  | all drug protein association file   |
| drug_drug_association  | all drug drug association file   |
| drug_disease_association  | all drug disease association file   |
| protein_disease_association  | all protein disease association file   |
| protein_protein_association  | all protein protein association file   |
| all_sample_drug_protein.csv  | all drug-protein sample  |
| attr_drug_protein_matrix.csv | feature of drug and protein fused with GIP |


# Installation
DTGHAT is tested to work under:

Python == 3.7

pytorch == 1.10.2+cu113

scipy == 1.5.4

numpy == 1.19.5

sklearn == 0.24.2

pandas == 1.1.5

matplotlib == 3.3.4

networkx == 2.5.1

# Quick start
To reproduce our results:

1, Download the environment required by MHGTMDA
```
pip install pytorch == 1.10.2+cu113
```
2, Run embedding.py to generate miRNA and disease embedding features, Specifically, we constructed the graph utilizing the torch_geometric tool. First, We enter the collected biological entities into HeteroData as nodes (HeteroData is a PyG built-in data structure for representing heterogeneous graphs). Next, we constructed node mappings by different node types to construct edge indexes in HeteroData. Finally, we construct node type labels to represent the type of each node in HeteroData..the options are:
```
python ./src/drug_embedding.py
```
3, The specific code is run by referring to the following train.py to generate train_model and performance score, the options are:
```
python ./src/DTI_train.py

```
4, Ablation experiment： To further demonstrate the effectiveness of MHGTMDA, we conducted two sets of ablation experiments, removing attribute features and structural features respectively, to compare the effects with MHGTMDA under 5-fold cross-validation experiment. The specific code is run by referring to the following ../attributes_feature/train.py，../network_feature/train.py to generate performance score for everyone, the options are:
```
python ./ablation/attributes_feature/train.py

python ./ablation/network_feature/train.py
```
5, We use a 5-fold cross-validation strategy to evaluate the generalization ability of our model (MHGTMDA). In the results, we plot the receiver operating characteristic curves(ROCs) and precision-recall curves (PRCs). Furthermore, the area under the ROCs (AUC) was also used to measure the ability of MHGTMDA. The specific code is run by referring to the following ./5CV/train.py to generate 5-CV scores, the options are:
```
python ./5CV/train.py
```
6, we conducted differential expression analyses on breast cancer and lung cancer, using MHGTMDA to further validate differentially expressed miRNAs. The specific code is run by referring to the following ./src/casestudies.py to generate two disease predictions, the options are:
```
python  ./src/casestudies.py
```
# License
This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.



# Contributing

All authors were involved in the conceptualization of the proposed method. XCJ conceived and supervised the project. XCJ designed the study and developed the approach. XCJ implemented and applied the method to microbial data. XCJ analyzed the results. XCJ and LM contributed to the review of the manuscript before submission for publication. All authors read and approved the final manuscript.
# Cite



# Contacts
If you have any questions or comments, please feel free to email: 2796612230@qq.com.
