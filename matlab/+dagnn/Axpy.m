classdef Axpy < dagnn.Filter

  methods
    function outputs = forward(obj, inputs, params) %#ok
      outputs{1} = vl_nnaxpy(inputs{1}, inputs{2}, inputs{3}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok
      derInputs = vl_nnaxpy(inputs{1}, inputs{2}, inputs{3}, derOutputs{1}) ;
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes) %#ok
      % The output size of (A * X) + Y should be equal to
      % both X and Y
      assert(~any(isnan(inputSizes{2})), ...
           ['Found NaNs in the input variable dimensions. Was the ' ...
             'dagnn.print() function called with input sizes?']) ;
      assert(all(inputSizes{2} == inputSizes{3}), 'mismatch') ;
      outputSizes{1} = inputSizes{2} ;
    end

    function obj = Axpy(varargin)
      obj.load(varargin) ;
    end
  end
end
