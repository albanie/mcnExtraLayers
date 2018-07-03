classdef GlobalPooling < dagnn.Filter
  properties
    method = 'avg'
  end

  methods
    function outputs = forward(obj, inputs, params) %#ok
      outputs{1} = vl_nnglobalpool(inputs{1}, 'method', obj.method) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok
      derInputs{1} = vl_nnglobalpool(inputs{1}, derOutputs{1}, 'method', obj.method) ;
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes) %#ok
      assert(~any(isnan(inputSizes{1})), ...
           ['Found NaNs in the input variable dimensions. Was the ' ...
             'dagnn.print() function called with input sizes?']) ;
      outputSizes{1}(3:4) = inputSizes{1}(3:4) ;
      outputSizes{1}(1:2) = [1 1] ; % collapse spatial dims
    end

    function obj = GlobalPooling(varargin)
      obj.load(varargin) ;
    end
  end
end
