classdef HuberLoss < dagnn.Loss

  properties
    sigma = 1.
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnhuberloss(inputs{1}, inputs{2}, ...
                       'instanceWeights', inputs{3}, ...
                       obj.opts{:}) ;

      n = obj.numAveraged ;
      m = n + 1 ; % averaging handled by instance weights
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnhuberloss(inputs{1}, inputs{2}, derOutputs{1}, ...
                                'instanceWeights', inputs{3}, ...
                                obj.opts{:}) ;
      derInputs{2} = [NaN] ;
      derInputs{3} = [NaN] ;
      derParams = {} ;
    end

    function obj = LossSmoothL1(varargin)
      obj.load(varargin) ;
      obj.loss = 'smoothl1';
    end
  end
end
