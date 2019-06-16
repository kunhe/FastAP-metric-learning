function [net, opts] = get_model(opts, addFC)
if nargin < 2, addFC = true; end

t0 = tic;
modelFunc = str2func(sprintf('models.%s', opts.arch));
[net, opts, in_name, in_dim] = modelFunc(opts);
logInfo('%s in %.2fs', opts.arch, toc(t0));

% + FC layer
if addFC
    convobj = dagnn.Conv('size', [1 1 in_dim opts.dim], ...
        'pad', 0, 'stride', 1, 'hasBias', true);
    params = convobj.initParams();
    net.addLayer('fc', convobj, {in_name}, {'logits'}, {'fc_w', 'fc_b'});
    p1 = net.getParamIndex('fc_w');
    p2 = net.getParamIndex('fc_b');
    net.params(p1).value = params{1};
    net.params(p2).value = params{2};
    net.params(p1).learningRate = opts.lrmult;
    net.params(p2).learningRate = opts.lrmult;

    in_name = 'logits';
    in_dim  = opts.dim;
end

% + l2 normalization layer
net.addLayer('L2norm', dagnn.LRN('param', [2*in_dim, 0, 1, 0.5]), ...
    {in_name}, {'feats_l2'});

% + loss layer
lossobj = str2func(opts.obj);
net.addLayer('loss', lossobj('opt', opts), {'feats_l2', 'labels'}, {'objective'});

% print
if 0
    net.print({'data', [opts.imageSize opts.imageSize 3 opts.batchSize]}, ...
        'MaxNumColumns', 4, 'Layers', [], 'Parameters', []);
end

end
