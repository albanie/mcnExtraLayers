function y = vl_nnsoftmaxceloss(x, p, varargin)
%VL_NNSOFTMAXCELOSS computes the cross entropy loss from logits
%   Y = VL_NNSOFTMAXCELOSS(X, P) computes the cross entropy loss between
%   and 1 x 1 x C x N array of predicted logits and an 1 x 1 x C x N array
%   of target probabilities. Note that the third dimension of P must form
%   valid probabilities (i.e. sum to 1, and 0<=p<=1). The output Y is a
%   SINGLE scalar.
%
%   The Cross Entropy Loss between X and P is as:
%
%     Loss = - sum_n sum_c p_{cn} log(softmax(x_n)_c)
%
% Copyright (C) 2018 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.tol = 1e-5 ;
  [opts, dzdy] = vl_argparsepos(opts, varargin) ;

  % check valid probability targets
  normCond = all(abs(sum(p,3) - 1) < opts.tol) ;
  assert(normCond, 'values of p are not correctly normalized') ;
  rangeCond = all((0 <= p(:)) & (p(:) <= 1)) ;
  assert(rangeCond, 'values of p must lie between 0 and 1') ;

  Xmax = max(x,[],3) ;
  ex = exp(bsxfun(@minus, x, Xmax)) ;

  if isempty(dzdy)
		t = p .* (Xmax + log(sum(ex,3)) - x) ;
    y = sum(t(:)) ;
  else
		q = bsxfun(@rdivide, ex, sum(ex,3)) ;
    dydx = q - p ;
		y = bsxfun(@times, dzdy{1}, dydx) ;
  end
