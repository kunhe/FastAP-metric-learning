% util
addpath('util');

% mink()
if ~exist('mink', 'builtin')
    warning('Please use Matlab R2017b or newer, or follow the README to download the MinMaxSelection implementation.');
end

% matconvnet
logInfo('setting up [MatConvNet]');
run ./matconvnet/matlab/vl_setupnn

% mcn extra layers
logInfo('setting up [mcnExtraLayers]');
vl_contrib setup mcnExtraLayers

% autonn
logInfo('setting up [autonn]');
vl_contrib setup autonn

logInfo('Done!');
