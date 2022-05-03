# Federated Learning Simulator (FLSim)

<p align="center">
  <img src="https://github.com/facebookresearch/FLSim/blob/main/assets/logo.png">
</p>

<!-- [![CircleCI](https://circleci.com/gh/pytorch/flsim.svg?style=svg)](https://circleci.com/gh/pytorch/flsim) -->

Federated Learning Simulator (FLSim) is a flexible, standalone library written in PyTorch that simulates FL settings with a minimal, easy-to-use API. FLSim is domain-agnostic and accommodates many use cases such as computer vision and natural text. Currently FLSim supports cross-device FL, where millions of clients' devices (e.g. phones) train a model collaboratively together.

FLSim is scalable and fast. It supports differential privacy (DP), secure aggregation (secAgg), and a variety of compression techniques.

In FL, a model is trained collaboratively by multiple clients that each have their own local data, and a central server moderates training, e.g. by aggregating model updates from multiple clients.

In FLSim, developers only need to define a dataset, model, and metrics reporter. All other aspects of FL training are handled internally by the FLSim core library.

## FLSim
### Library Structure

<p align="center">
  <img src="https://github.com/facebookresearch/FLSim/blob/main/assets/FLSim_Overview.png">
</p>

FLSim core components follow the same semantic as FedAvg. The server comprises three main features: selector, aggregator, and optimizer at a high level. The selector selects clients for training, and the aggregator aggregates client updates until a round is complete. Then, the optimizer optimizes the server model based on the aggregated gradients. The server communicates with the clients via the channel. The channel then compresses the message between the server and the clients. Locally, the client consists of a dataset and a local optimizer. This local optimizer can be SGD, FedProx, or a custom Pytorch optimizer.

## Installation
The latest release of FLSim can be installed via `pip`:
```bash
pip install flsim
```

You can also install directly from the source for the latest features (along with its quirks and potentially occasional bugs):
```bash
git clone https://github.com/facebookresearch/FLSim.git
cd FLSim
pip install -e .
```

## Getting started

To implement a central training loop in the FL setting using FLSim, a developer simply performs the following steps:

1. Build their own data pipeline to assign individual rows of training data to client devices (to simulate data distributed across client devices)
2. Create a corresponding `torch.nn.Module` model and wrap it in an FL model.
3. Define a custom metrics reporter that computes and collects metrics of interest (e.g. accuracy) throughout training.
4. Set the desired hyperparameters in a config.


## Usage Example

### Tutorials
* [Image classification with CIFAR-10](https://github.com/facebookresearch/FLSim/blob/main/tutorials/cifar10_tutorial.ipynb)
* [Sentiment classification with LEAF's Sent140](https://github.com/facebookresearch/FLSim/blob/main/tutorials/sent140_tutorial.ipynb)
* [Compression for communication efficiency](https://github.com/facebookresearch/FLSim/blob/main/tutorials/channel_feature_tutorial.ipynb)
* [Adding a custom communication channel](https://github.com/facebookresearch/FLSim/blob/main/tutorials/custom_channel_tutorial.ipynb)

To see the details, please refer to the [tutorials](https://github.com/facebookresearch/FLSim/tree/main/tutorials) that we have prepared.

### Examples
We have prepared the runnable examples for 2 of the tutorials above:
* [Image classification with CIFAR-10](https://github.com/facebookresearch/FLSim/blob/main/examples/cifar10_example.py)
* [Sentiment classification with LEAF's Sent140](https://github.com/facebookresearch/FLSim/blob/main/examples/sent140_example.py)


## Contributing
See the [CONTRIBUTING](https://github.com/facebookresearch/FLSim/blob/main/CONTRIBUTING.md) for how to contribute to this library.


## License
This code is released under Apache 2.0, as found in the [LICENSE](https://github.com/facebookresearch/FLSim/blob/main/LICENSE) file.
