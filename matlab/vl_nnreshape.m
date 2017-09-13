function y = vl_nnreshape(x, shape, varargin)
% VL_NNRESHAPE Feature reshaping
%   Y = VL_NNRESHAPE(X, SHAPE) reshpaes the input data X to have
%   the dimensions specified by SHAPE. X is a SINGLE array of 
%   dimension H x W x D x N where (H,W) are the height and width of 
%   the map stack, D is the image depth (number of feature channels) 
%   and N the number of of images in the stack. SHAPE is a 1 x 3 cell 
%   array, the contents of which are passed in order to the MATLAB 
%   reshape function. As a consequence, `[]` an be used to specify a 
%   dimension which should be computed from the other two. The batch size
%   (the fourth dimension of the input) is left unchanged by this 
%   reshaping operation.
%
%   Example:
%       Inputs: X with shape [100 100 3 5] and SHAPE = { 100 3 [] } 
%       will produce an output Y with shape [100 3 100 5]
%
%   DZDX = VL_NNRESHAPE(X, SHAPE, DZDY) computes the derivatives of the
%   block projected onto DZDY. DZDX and DZDY have the same dimensions
%   as X and Y respectively.
%
% Copyright (C) 2017 Samuel Albanie and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  [~, dzdy] = vl_argparsepos(struct(), varargin) ;

  batchSize = size(x, 4) ;

  if isempty(dzdy)
    y = reshape(x, shape{1}, shape{2}, shape{3}, batchSize) ;
  else
    y = reshape(dzdy{1}, size(x)) ;
  end
