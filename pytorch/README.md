## PyTorch implementation of "Deep Metric Learning to Rank"

The PyTorch version of FastAP is being implemented within the framework of 
[Deep-Metric-Learning-Baselines](https://github.com/kunhe/Deep-Metric-Learning-Baselines).
In this repository we provide a standalone implementation of the loss layer.

#### NOTE
- To completely reproduce results reported in the paper, the FastAP loss needs to be used in conjunction with the proposed minibatch sampling method, which is implemented in the linked repo.
- It is currently a direct port of the Matlab implementation. To investigate: better use of automatic differentiation.

**TODO**
- [x] implement FastAP's minibatch sampling method
- [ ] reproduce results in the paper
