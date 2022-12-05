# Hidden Poison: Machine Unlearning Enables Camouflaged Poisoning Attacks
A Python implementation of [Hidden Poison: Machine Unlearning Enables Camouflaged Poisoning Attacks](https://arxiv.org)

## Abstract:
We introduce camouflaged data poisoning attacks, a new attack vector that arises in the context of machine unlearning and other settings when model retraining may be induced. An adversary first adds a few carefully crafted points to the training dataset such that the impact on the model's predictions is minimal. The adversary subsequently triggers a request to remove a subset of the introduced points at which point the attack is unleashed and the model's predictions are negatively affected. In particular, we consider clean-label targeted attacks (in which the goal is to cause the model to misclassify a specific test point) on datasets including CIFAR-10, Imagenette, and Imagewoof. This attack is realized by constructing camouflage datapoints that mask the effect of a poisoned dataset.

## Dependencies

* PyTorch => 1.6.*
* torchvision > 0.5.*
* [Imagenette](https://github.com/fastai/imagenette): only if the Imagenette dataset is used
* [Imagewoof](https://github.com/fastai/imagenette): only if the Imagewoof dataset is used

## Usage

Use run_single.py to start a quick run with default parameters.

To use Imagenette or Imagewoof dataset, run data/fetch_dataset.sh to download and unzip the images. (You may also do it manually)
