# SaMoE
Code for ACL2022 findings paper "Table-based Fact Verification with Self-adaptive Mixture of Experts"
# Introduction
**S**elf-**a**daptive **M**ixture-**o**f-**E**xperts **N**etwork (**SaMoE**) is a framework that aims at dealing with statement verification based on tables, which requires complicated numerical/logical/textual reasoning skills. The network is composed of multiple experts, each handling a specific part of the semantics for reasoning, whereas a management module is applied to decide the contribution of each expert network to the verification result. A self-adaptive method is developed to teach the management module combining results of different experts more efficiently without external knowledge. SaMoE achieves **85.1%** accuracy on the benchmark dataset TabFact, comparable with the previous state-of-the-art models. 
![SaMoE's architechture](https://github.com/THUMLP/SaMoE/blob/master/overview.png?raw=true)
***
# Requirement
- nltk==3.5
- numpy==1.19.2
- pandas==1.1.5
- pytorch_pretrained_bert==0.6.2
- scikit_learn==0.24.1
- scipy==1.5.4
- tensorboardX==2.1
- torch==1.7.1
- tqdm==4.59.0
- transformers==4.10.2
- ujson==5.1.0
- Unidecode==1.2.0
# Preparation
1. Clone this repo.   
2. Download RoBERTa-Large model from [https://huggingface.co/roberta-large](https://huggingface.co/roberta-large). Place all the files into a new folder "roberta_large" under the root.
# Data Preprocessing
We follow the data preprocessing method proposed in [Chen et al.2019](https://github.com/wenhuchen/Table-Fact-Checking) with a slight modification on the table pruning algorithm:
```
cd code
python preprocess_data.py
python preprocess_BERT_revised.py
```
You may find in the root a new folder "processed_datasets_revised", which contains all the processed data.
# Multi-expert Training
To train the MoE:
```
python run_moe.py --do_train --do_eval
```
You can evaluate the MoE without self-adaptive learning by:
```
python run_moe.py --do_eval --do_simple_test --do_complex_test --do_small_test --load_dir [modeldir]
```
e.g., if you train the MoE with default setting, then the evaluation command should be:
```
python run_moe.py --do_eval --do_simple_test --do_complex_test --do_small_test --load_dir outputs_moe/saved_model
```
# Self-adaptive Learning
Before running self-adaptive learning, you have to get the expert ability information on the train set by:
```
python run_moe.py --do_train_eval --load_dir [modeldir]
```
Then conduct self-adaptive learning based on the expert ability on the train set:
```
python run_adaptive_learning.py --do_train --do_eval --load_dir [modeldir]
```
To evaluate the model:
```
python run_adaptive_learning.py --do_eval --do_simple_test --do_complex_test --do_small_test --load_dir [modeldir]
```
e.g., by default the command should be:
```
python run_adaptive_learning.py --do_eval --do_simple_test --do_complex_test --do_small_test --load_dir outputs_adaptive_learning/saved_model
```
# Loading checkpoints
We have released our well-trained models [here](https://drive.google.com/drive/folders/1vYDF3-c6XatPZkpBVrpX30GfvAfovWOg?usp=sharing). The moe.zip contains the MoE model before the self-adaptive learning, and the samoe.zip contains SaMoE that achieves the best performance in our paper. You may evaluate these models with the commands provided above.

