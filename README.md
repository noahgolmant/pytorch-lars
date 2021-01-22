# pytorch-lars
Layer-wise Adaptive Rate Scaling in PyTorch

This repo contains a PyTorch implementation of layer-wise adaptive rate scaling (LARS) from the paper "[Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888)" by You, Gitman, and Ginsburg. Another version of this was recently included in [PyTorch Lightning](https://pytorch-lightning-bolts.readthedocs.io/en/stable/api/pl_bolts.optimizers.lars_scheduling.html).

To run, do

`python train.py --optimizer LARS --cuda lars_results`

It uses [skeletor-ml](https://github.com/noahgolmant/skeletor) for experiment logging. But the main optimizer file does not depend on that framework.

## Preliminary results

I just tested this using a ResNet18 on CIFAR-10. I used a standard [gradient accumulation trick](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255) to train on very large batch sizes.

![Alt text](images/lars_test_curves.jpg?raw=true "Title")

| Batch Size | Test Accuracy |
| ---------- | ------------- |
|    64      |    89.39      |
|    256     |    85.45      |
|    1024    |    81.2       |
|    4096    | 73.41         |
| 16384      | 64.13         |

As a comparison, using SGD with momentum, I am able to achieve about 93.5% test accuracy in 200 epochs using a geometric decay schedule (using [this](https://github.com/kuangliu/pytorch-cifar) implementation). I have not done extensive hyperparameter tuning, though -- I used the default parameters suggested by the paper. I had a base learning rate of 0.1, 200 epochs, eta .001, momentum 0.9, weight decay of 5e-4, and the polynomial learning rate decay schedule.

There are two likely explanations for the difference in performance. One is hyperparameter tuning. ResNet18 may have different optimal hyperparameters compared to ResNet50, or CIFAR-10 may have different ones compared to ImageNet. Or both. Plugging in a geometric schedule in place of (or in addition to) the polynomial decay schedule may be the main culprit. The other possibility is that the gradient accumulation trick mentioned above interacts in unexpected ways with batch normalization. Both options could cause a performance regression.
