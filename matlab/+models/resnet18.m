function [net, opts, in_name, in_dim] = resnet18(opts)
opts.imageSize = 224;
opts.maxGpuImgs = 256;  % # of images that a 12GB GPU can hold

net = load(fullfile(opts.localDir, 'models', 'resnet18-pt-mcn.mat'));
net = dagnn.DagNN.loadobj(net) ;

% remove softmax, fc
net.removeLayer('classifier_0');

% freeze pretrained layers
if opts.lastLayer
    for i = 1:numel(net.params)
        net.params(i).learningRate = 0;
        net.params(i).weightDecay  = 0;
    end
end

in_name = 'classifier_flatten';
in_dim = 512;
end
