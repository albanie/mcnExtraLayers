## mcnUtils

A collection of common MatConvNet functions and DagNN layers which are shared 
across a number of classification and object detection frameworks.

Common utilities for MatConvNet, which includes:

* [findBestCheckpoint](https://github.com/albanie/mcnUtils/blob/master/matlab/findBestCheckpoint.m) - 
function to rank and prune network checkpoints saved during training (useful for saving space automatically at the end of a training run)


### Install

The module is easiest to install with the `vl_contrib` package manager:

```
vl_contrib('install', 'mcnUtils', 'contribUrl', 'github.com/albanie/matconvnet-contrib-test/') ;
vl_contrib('setup', 'mcnUtils', 'contribUrl', 'github.com/albanie/matconvnet-contrib-test/') ;
```
