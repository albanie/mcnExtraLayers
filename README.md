## mcnExtraLayers

A collection of common MatConvNet functions and DagNN layers which are shared 
across a number of classification and object detection frameworks.

### Layers:

* `vl_nnflatten` - flatten along a given dimension
* `vl_nnglobalpool` - global pooling
* `vl_nnhuberloss` - computation of the Huber (L1-smooth) loss
* `vl_nnreshape` -  tensor reshaping
* `vl_nnsoftmaxt` - softmax along a given dimension

### Utilities

* [findBestCheckpoint](https://github.com/albanie/mcnExtraLayers/blob/master/utils/findBestCheckpoint.m) - 
function to rank and prune network checkpoints saved during training (useful for saving space automatically at the end of a training run)


### Install

The module is easiest to install with the `vl_contrib` package manager:

```
vl_contrib('install', 'mcnExtraLayers') ;
vl_contrib('setup', 'mcnExtraLayers') ;
```
