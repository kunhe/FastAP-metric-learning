function [net, opts, in_name, in_dim] = googlenet(opts)
opts.imageSize = 224;
opts.maxGpuImgs = 320;  % # of images that a 12GB GPU can hold

% finetune GoogLeNet
net = load(fullfile(opts.localDir, 'models', 'imagenet-googlenet-dag.mat'));
net = dagnn.DagNN.loadobj(net) ;

% remove softmax, fc
net.removeLayer('softmax');
net.removeLayer('cls3_fc');

net.removeLayer('cls2_pool');
net.removeLayer('cls2_reduction');
net.removeLayer('relu_cls2_reduction');
net.removeLayer('cls2_fc1');
net.removeLayer('relu_cls2_fc1');
net.removeLayer('cls2_fc2');

net.removeLayer('cls1_pool');
net.removeLayer('cls1_reduction');
net.removeLayer('relu_cls1_reduction');
net.removeLayer('cls1_fc1');
net.removeLayer('relu_cls1_fc1');
net.removeLayer('cls1_fc2');

% freeze pretrained layers
if opts.lastLayer
    for i = 1:numel(net.params)
        net.params(i).learningRate = 0;
        net.params(i).weightDecay  = 0;
    end
elseif isfield(opts, 'ft') && opts.ft >= 1
    assert(opts.ft>=1 && opts.ft<=9);
    names = {};
    for l = opts.ft : 9
        names{end+1} = sprintf('icp%d', l);
    end
    for i = 1:numel(net.params)
        pname = net.params(i).name;
        notfound = cellfun(@(x) isempty(strfind(pname, x)), names);
        if all(notfound)
            net.params(i).learningRate = 0;
            net.params(i).weightDecay  = 0;
        end
    end
end

in_name = 'cls3_pool';
in_dim = 1024;
end
