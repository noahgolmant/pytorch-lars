# pytorch-lars
Layer-wise Adaptive Rate Scaling in PyTorch

This repo contains a PyTorch implementation of layer-wise adaptive rate scaling (LARS) from the paper "[Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888)" by You, Gitman, and Ginsburg.

I am currently running benchmarks for various batch sizes and will update this readme when I have results.

To run, do

`python train.py --optimizer LARS --cuda lars_results`

It uses [skeletor-ml](https://github.com/noahgolmant/skeletor) for experiment logging.

