function evaluate_model(net, imdb, batchFunc, opts, varargin)

if strcmp(opts.dataset, 'inshop')
    % separate query set and database
    query_id   = find(imdb.images.set == 2);
    gallery_id = find(imdb.images.set == 3);
    Yquery     = imdb.images.labels(query_id, :);
    Ygallery   = imdb.images.labels(gallery_id, :);
    Fquery     = cnn_encode(net, imdb, batchFunc, query_id,   opts, varargin{:})';
    Fgallery   = cnn_encode(net, imdb, batchFunc, gallery_id, opts, varargin{:})';
    whos Fquery Fgallery

    logInfo('[%s] Recall ...', opts.dataset);
    evaluate_recall(Fquery, Yquery, Fgallery, Ygallery, opts, false);

elseif strcmp(opts.dataset, 'vid')
    % small, medium, large
    test_sets = {'Small', 'Medium', 'Large'};
    for s = 1:3
        test_id = find(imdb.images.set(:, s+1) == 1);
        Ytest = imdb.images.labels(test_id, :);
        Ftest = cnn_encode(net, imdb, batchFunc, test_id, opts, varargin{:})';
        logInfo('[%s] Recall on [%s] ...', dataset, test_sets{s});
        evaluate_recall(Ftest, Ytest, Ftest, Ytest, opts);
    end

else
    % query set == database
    test_id = find(imdb.images.set == 3);
    Ytest = imdb.images.labels(test_id, :);
    Ftest = cnn_encode(net, imdb, batchFunc, test_id, opts, varargin{:})';
    whos Ftest

    logInfo('[%s] Recall & NMI ...', opts.dataset);
    evaluate_recall(Ftest, Ytest, Ftest, Ytest, opts);
end

end

% ------------------------------------------------------------------------------
% ------------------------------------------------------------------------------

function evaluate_recall(Xq, Yq, Xdb, Ydb, opts, removeDiag)
% features: Nxd matrix
if ~exist('removeDiag', 'var'), removeDiag = true; end

[Nq, d] = size(Xq);
assert(Nq == size(Yq, 1));
assert(size(Yq, 2) == size(Ydb, 2));

% pairwise distances
tic;
distmatrix = 2 - 2 * Xq * Xdb';  % NxN
if removeDiag
    distmatrix(logical(eye(Nq))) = Inf;
end

% recall rate
Ks = sort(opts.Ks, 'ascend');
recallrate = zeros(Nq, numel(Ks));

for i = 1:Nq
    % get top max(K) results, in increasing distance
    [val, ind] = mink(distmatrix(i, :), Ks(end));

    % recall rates for all K's
    for j = 1:numel(Ks)
        ind_k = ind(1 : Ks(j));
        recallrate(i, j) = any(Ydb(ind_k, 1) == Yq(i, 1));
    end
end
toc;

for j = 1:numel(Ks)
    fprintf('K: %4d, Recall: %.3f\n', opts.Ks(j), mean(recallrate(:, j)));
end

end

% ------------------------------------------------------------------------------
% ------------------------------------------------------------------------------

function H = cnn_encode(net, imdb, batchFunc, ids, opts, varName)
if ~exist('varName', 'var')
    varName = 'feats_l2';
end

batch_size = 2 * opts.maxGpuImgs;
onGPU = numel(opts.gpus) > 0;

logInfo('Testing [%s] on %d -> %s', opts.arch, length(ids), varName);

net.mode = 'test';
ind = net.getVarIndex(varName);
net.vars(ind).precious = 1;
if onGPU, net.move('gpu'); end

assert(strcmp(varName, 'feats_l2'));
H = zeros(opts.dim, length(ids), 'single');

tic;
for t = 1:batch_size:length(ids)
    ed = min(t+batch_size-1, length(ids));
    inputs = batchFunc(imdb, ids(t:ed));
    net.eval(inputs);

    ret = net.vars(ind).value;
    ret = squeeze(gather(ret));
    H(:, t:ed) = ret;
end

if onGPU && isa(net, 'dagnn.DagNN')
    net.reset(); 
    net.move('cpu'); 
end
toc;

end
