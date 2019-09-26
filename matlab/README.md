## Matlab implementation of "Deep Metric Learning to Rank"

### Requirements
* Matlab R2017b or newer
  * This is to use the built-in [mink](https://www.mathworks.com/help/matlab/ref/mink.html) function. 
  * Alternatively, for earlier Matlab versions, you can use [this implementation](https://www.mathworks.com/matlabcentral/fileexchange/23576-min-max-selection) of mink.
* [MatConvNet](http://www.vlfeat.org/matconvnet/) v1.0-beta25 (with [`vl_contrib`](http://www.vlfeat.org/matconvnet/mfiles/vl_contrib/))
* [mcnExtraLayers](https://github.com/albanie/mcnExtraLayers) via `vl_contrib setup mcnExtraLayers`
* [autonn](https://github.com/vlfeat/autonn) via `vl_contrib setup autonn`

### Preparation
* Install/symlink MatConvNet at `./matconvnet` under this directory
* Create or symlink a directory `./cachedir` under the this directory
* Create a subdirectory `./cachedir/data`, and create symlinks to the datasets
    * Stanford Online Products at `./cachedir/data/Stanford_Online_Products`
    * In-Shop Clothes Retrieval at `./cachedir/data/InShopClothes`
    * PKU VehicleID at `./cachedir/data/VehicleID_V1.0`
* Create `./cachedir/models` and download pretrained models
  * [GoogLeNet](http://www.vlfeat.org/matconvnet/models/imagenet-googlenet-dag.mat): download to `./cachedir/models/imagenet-googlenet-dag.mat`
  * [ResNet-18](http://www.robots.ox.ac.uk/~albanie/models/pytorch-imports/resnet18-pt-mcn.mat): download to `./cachedir/models/resnet18-pt-mcn.mat`
  * [ResNet-50](http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat): download to `./cachedir/models/imagenet-resnet-50-dag.mat`

### Usage
We provide a unified interface `run_demo.m` to run all experiments conducted in the paper. The general syntax is 
```
run_demo([dataset], [key-value pairs])
```
where 
* `dataset` is one of `'products', 'inshop', 'vid'`
* Various parameters are specified as key-value pairs. The full list can be found by inspecting `get_opts.m`. Some notable ones are:
  * `'gpus'` (int) 1-based GPU index. Current implementation only supports 1 GPU.
  * `'arch'` (string) network architecture. Available: `'googlenet', 'resnet18', 'resnet50'`
  * `'dim'` (int) embedding dimesionality, default 512
  * `'nbins'` (int) number of distance quantizations, default 10
  * `'solver'` (string) SGD optimizer. Default: `'adam'`. Others: `'sgd', 'adadelta', 'adagrad', 'rmsprop'`

### Notes
* Our experiments are mainly run on a Titan X Pascal GPU with 12GB memory. If your GPU has a different amount of memory, you may want to change the default value of `opts.maxGpuImgs` (in `+models/*.m`) accordingly for each model architecture.
