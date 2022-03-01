# SaMoE
Code for ACL2022 findings paper "Table-based Fact Verification with Self-adaptive Mixture of Experts"
# Introduction
**S**elf-**a**daptive **M**ixture-**o**f-**E**xperts **N**etwork (**SaMoE**) is a framework that aims at dealing with statement verification based on tables, which requires complicated numerical/logical/textual reasoning skills. The network is composed of multiple experts, each handling a specific part of the semantics for reasoning, whereas a management module is applied to decide the contribution of each expert network to the verification result. A self-adaptive method is developed to teach the management module combining results of different experts more efficiently without external knowledge. SaMoE achieves **85.1%** accuracy on the benchmark dataset TabFact, comparable with the previous state-of-the-art models. 
![SaMoE's architechture](https://github.com/Zhouyx17/SaMoE/blob/main/overview.png?raw=true)
***
# Requirement
>nltk==3.5
numpy==1.19.2
pandas==1.1.5
pytorch_pretrained_bert==0.6.2
scikit_learn==0.24.1
scipy==1.5.4
tensorboardX==2.1
torch==1.7.1
tqdm==4.59.0
transformers==4.10.2
ujson==5.1.0
Unidecode==1.2.0
# Quick Start
To be updated very soon. (We're preparing the checkpoints)
# Start From Scratch
To be updated very soon. (We're preparing the codes and tutorial for reproducing our results)
