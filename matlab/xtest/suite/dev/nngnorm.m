classdef nngnorm < nntest

  properties (TestParameter)
    %rows = {2 8 13}
    %cols = {2 8 17}
    %numDims = {1 3 4}
    %batchSize = {2 7}
    rows = {2}
    cols = {2}
    numDims = {5}
    batchSize = {1}
  end

  methods (Test)
    function basic(test, rows, cols, numDims, batchSize)
      H = rows ;
      W = cols ;
      C = numDims ;
      bs = batchSize ;
      numGroups = 1 ;
      x = test.randn(H, W, C, bs) ;
      g = test.randn(1, 1, C/numGroups, bs) / test.range ;
      b = test.randn(1, 1, C/numGroups, bs) / test.range ;

      args = {'numGroups', numGroups} ;
      y = vl_nngnorm(x, g, b, args{:}) ;
      dzdy = test.randn(size(y)) ;
      [dzdx,dzdg,dzdb] = vl_nngnorm(x, g, b, dzdy, args{:}) ;
      test.der(@(x) vl_nngnorm(x, g, b, args{:}), x, dzdy, dzdx, test.range * 1e-3) ;
      test.der(@(g) vl_nngnorm(x, g, b, args{:}), g, dzdy, dzdg, test.range * 1e-3) ;
      test.der(@(b) vl_nngnorm(x, g, b, args{:}), b, dzdy, dzdb, test.range * 1e-3) ;
    end
  end
end
