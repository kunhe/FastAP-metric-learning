## PyTorch implementation of "Deep Metric Learning to Rank"

The PyTorch version of FastAP is being implemented within the framework of 
[Deep-Metric-Learning-Baselines](https://github.com/kunhe/Deep-Metric-Learning-Baselines).
In this repository we provide a standalone implementation of the loss layer.

#### NOTE
- Completely reproducing results in the paper also requires implementing the minibatch sampling method, which is underway.
- It is currently a direct port of the Matlab implementation. To investigate: better use of automatic differentiation.

**TODO**
- [ ] implement FastAP's minibatch sampling method
- [ ] reproduce results in the paper
