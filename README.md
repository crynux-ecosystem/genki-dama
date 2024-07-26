<div align="center">

# **Genki-Dama (元気玉)**
## Federated Learning subnet on BitTensor, powered by Crynux  <!-- omit in toc -->


[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)]([https://discord.gg/bittensor](https://discord.gg/vJZnq4ujvK))
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

![Genki-Dama](genki.gif)

</div>

---
- [Introduction](#introduction)
- [Federated Learning](#federated-learning)
- [Miner](#miner)
- [Validator](#validator)
- [Roadmap](#roadmap)
- [License](#license)

---

## Introduction

Genki-Dama, inspired by the iconic Dragon Ball technique, is the first open-source machine learning framework that utilizes decentralized data and harnesses decentralized computing resources. Built upon the incentive mechanism of BitTensor, and the computing network of Crynux, Genki-Dama empowers federated learning in a decentralized manner, shattering the limitation of centralized approaches.

It includes two parts:
* Genki: machine-learning SDK to utilize BitTensor incentive mechanism and Crynux decentralized computing resources
* Dama: open-sourced model checkpoints trained by Genki

## Federated Learning

[Federated learning](https://en.wikipedia.org/wiki/Federated_learning) is a machine learning approach designed to harness decentralized data and computing while safeguarding privacy. Its workflow can be summarized as follows:

1. Miners locally prepare data, train models, and upload weights to a central server.
2. The server aggregates gradients from miners and initiates a new round as needed.


Federated learning has been widely used in [Gboard](https://research.google/blog/federated-learning-collaborative-machine-learning-without-centralized-training-data/), [speech models](https://support.google.com/assistant/answer/11140942) without using centralized training data.

Presently, Large Language Models (LLMs) alignment demands significant [human effort](https://openai.com/index/instruction-following/) to ensure data quality. BitTensor offers a superior incentive mechanism for this process. By employing federated learning on the BitTensor subnet, miners are incentivized to contribute high-quality instruction datasets, which are invaluable assets for both OpenAI and Google, neither of whom has disclosed these datasets to date.

Furthermore, as LLM pretraining exhausts public accessible data reservoirs, federated learning emerges as a viable solution to tap into additional high-quality data for both pretraining and fine-tuning purposes.

![Genki-Dama Overview](overview.png)


## Miner


Utilizing Crynux decentralized computing layer, miners engage with validators by finetuning on a base model with a customized datasets. The process involves the following steps:

1. Data collect: provide a high-quality dataset for a particular finetune task.
2. Finetune: train the model with customized datasets.
3. Commit: public the model to huggingface

Miners can choose whether to share their data:
* If they share data, data will be available on marketplace to trade.
* If they keep data private, only gradients will be communicated over network.

## Validator

Validators are responsible for evaluating the performance from each miners, and set weights to each miner.

Although it is difficult to evaluate data quality directly, validators can evaluate the performance of finetuned model, and set weights considering these factors:
* Model quality: model will be evaluated by cross-validation with other datasets.
* Data similarity: validators set higher weights to data with less similarity to encourage miners provide diversified data.

Validators order miners by their commit timestamp, and weight them with the score: (0.95 - data_similarity) * ∆loss 


## Roadmap

### Proof-of-concept
We start with a simple finetune task for proof-of-concept:
1. Miners provide high-quality datasets to write like Shakespeare on Tiny-GPT
2. Validators evaluate generated output on a predefined instruction dataset, and evaluate the data similarity between different miners.

### Federated Instruction Finetune

We utilizes Genki-Dama to finetune LLMs that can perform state-of-the-art results on benchmarks.

This is the improvement over [Shepherd-7B](https://arxiv.org/pdf/2305.05644)

### Data Markeplace

Miners optionally choose whether to share their data. Data quality is measured by the model trained from these data.

Clients can buy data from the marketplace.

Genki-Dama earns fees from these transactions.

### Model for Applications

Genki-Dama will be used to train models in real-world applications.

Validators evaluate the model quality via real traffic of applications. 

Genki-Dama earns license fee from these models.


## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
