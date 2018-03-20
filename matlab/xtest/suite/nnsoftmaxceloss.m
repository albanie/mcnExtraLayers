classdef nnsoftmaxceloss < nntest
  methods (Test)

    function basic(test)
      x = test.randn([1 1 8 50]) ;
      p = abs(test.rand([1 1 8 50])) ;
      p = bsxfun(@rdivide, p, sum(p, 3)) ;
      y = vl_nnsoftmaxceloss(x, p) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnsoftmaxceloss(x, p, dzdy) ;
      test.der(@(x) vl_nnsoftmaxceloss(x, p), x, dzdy, dzdx, 1e-4*test.range) ;
    end

    function basicWithInstanceWeights(test)
      x = test.randn([1 1 8 50]) ;
      p = abs(test.rand([1 1 8 50])) ;
      p = bsxfun(@rdivide, p, sum(p, 3)) ;
      w = test.randn([100 1]) ;
      y = vl_nnsoftmaxceloss(x, p, 'instanceWeights', w) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnsoftmaxceloss(x, p, dzdy, 'instanceWeights', w) ;
      test.der(@(x) vl_nnsoftmaxceloss(x, p, 'instanceWeights', w), x, ...
                                        dzdy, dzdx, 1e-3*test.range) ;
    end

  end
end
