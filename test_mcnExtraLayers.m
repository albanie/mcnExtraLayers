function test_mcnExtraLayers
% --------------------------------
% run tests for ExtraLayers module
% --------------------------------

% add tests to path
addpath(fullfile(fileparts(mfilename('fullpath')), 'matlab/xtest')) ;
addpath(fullfile(vl_rootnn, 'matlab/xtest/suite')) ;

% test network layers
run_extra_layers_tests('command', 'nn') ;
