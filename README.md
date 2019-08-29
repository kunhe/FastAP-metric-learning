# FastAP: Deep Metric Learning to Rank
This repository contains implementation of the following paper:

[Deep Metric Learning to Rank](http://openaccess.thecvf.com/content_CVPR_2019/html/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.html)<br>
[Fatih Cakir](http://cs-people.bu.edu/fcakir/)\*, [Kun He](http://cs-people.bu.edu/hekun/)\*, [Xide Xia](https://xidexia.github.io), [Brian Kulis](http://people.bu.edu/bkulis/), and [Stan Sclaroff](http://www.cs.bu.edu/~sclaroff/) (*equal contribution)<br>
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019

## Requirements
* **Matlab version**
  * Matlab R2017b or newer
    * This is to use the built-in [mink](https://www.mathworks.com/help/matlab/ref/mink.html) function. 
    * Alternatively, for earlier Matlab versions, you can use [this implementation](https://www.mathworks.com/matlabcentral/fileexchange/23576-min-max-selection) of mink.
  * [MatConvNet](http://www.vlfeat.org/matconvnet/) v1.0-beta25 (with [`vl_contrib`](http://www.vlfeat.org/matconvnet/mfiles/vl_contrib/))
  * [mcnExtraLayers](https://github.com/albanie/mcnExtraLayers) via `vl_contrib setup mcnExtraLayers`
  * [autonn](https://github.com/vlfeat/autonn) via `vl_contrib setup autonn`
* **PyTorch version**
  * coming soon
  
## Datasets
* Stanford Online Products
  * Can be downloaded [here](http://cvgl.stanford.edu/projects/lifted_struct/)
* In-Shop Clothes Retrieval
  * Can be downloaded [here](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
* PKU VehicleID
  * Please request the dataset from the authors [here](https://pkuml.org/resources/pku-vehicleid.html)

## Usage 
* **Matlab**: see `matlab/README.md`
* **PyTorch**: coming soon

## Contact
For questions and comments, feel free to contact: kunhe26@gmail.com or fcakirs@gmail.com

## License
MIT
