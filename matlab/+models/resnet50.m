function [net, opts, in_name, in_dim] = resnet50(opts)
opts.imageSize = 224;
opts.maxGpuImgs = 90;  % # of images that a 12GB GPU can hold

net = load(fullfile(opts.localDir, 'models', 'imagenet-resnet-50-dag.mat'));
net = dagnn.DagNN.loadobj(net) ;

% remove softmax, fc
net.removeLayer('prob');
net.removeLayer('fc1000');

% freeze pretrained layers
if opts.lastLayer
    for i = 1:numel(net.params)
        net.params(i).learningRate = 0;
        net.params(i).weightDecay  = 0;
    end
end

in_name = 'pool5';
in_dim = 2048;
end
