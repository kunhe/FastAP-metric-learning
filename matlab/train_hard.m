function [net,stats] = train_hard(net, imdb, getBatch, varargin)
%CNN_TRAIN_DAG Demonstrates training a CNN using the DagNN wrapper
%    CNN_TRAIN_DAG() is similar to CNN_TRAIN(), but works with
%    the DagNN wrapper instead of the SimpleNN wrapper.

% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
addpath(fullfile(vl_rootnn, 'examples'));

%%%%%%%%%%%%% new fields %%%%%%%%%%%%%
opts.saveInterval = 2;
opts.batchesPerMetaPair = 5;
opts.maxGpuImgs = Inf;
%%%%%%%%%%%%% new fields %%%%%%%%%%%%%

opts.expDir = fullfile('data','exp') ;
opts.continue = true ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.epochSize = inf;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;

opts.solver = [] ;  % Empty array means use the default SGD solver
[opts, varargin] = vl_argparse(opts, varargin) ;
if ~isempty(opts.solver)
    assert(isa(opts.solver, 'function_handle') && nargout(opts.solver) == 2,...
        'Invalid solver; expected a function handle with two outputs.') ;
    % Call without input arguments, to get default options
    opts.solverOpts = opts.solver() ;
end

opts.momentum = 0.9 ;
opts.saveSolverState = true ;
opts.nesterovUpdate = false ;
opts.randomSeed = 0 ;
opts.profile = false ;
opts.parameterServer.method = 'mmap' ;
opts.parameterServer.prefix = 'mcn' ;

opts.derOutputs = {'objective', 1} ;
opts.extractStatsFn = @extractStats ;
opts.plotStatistics = true;
opts.postEpochFn = [] ;  % postEpochFn(net,params,state) called after each epoch; can return a new learning rate, 0 to stop, [] for no change
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isscalar(opts.train) && isnumeric(opts.train) && isnan(opts.train)
    opts.train = [] ;
end
if isscalar(opts.val) && isnumeric(opts.val) && isnan(opts.val)
    opts.val = [] ;
end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

evaluateMode = isempty(opts.train) ;
if ~evaluateMode
    if isempty(opts.derOutputs)
        error('DEROUTPUTS must be specified when training.\n') ;
    end
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
    fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
    % [KH] make sure to use the input opts, saved maybe outdated
    opt = net.layers(end).block.opt;
    [net, state, stats] = loadState(modelPath(start)) ;
    net.layers(end).block.opt = opt;
else
    state = [] ;
end

for epoch=start+1:opts.numEpochs

    % Set the random seed based on the epoch and opts.randomSeed.
    % This is important for reproducibility, including when training
    % is restarted from a checkpoint.

    rng(epoch + opts.randomSeed) ;
    prepareGPUs(opts, epoch == start+1) ;

    % Train for one epoch.
    params = opts ;
    params.epoch = epoch ;
    params.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
    params.train = opts.train; %(randperm(numel(opts.train))) ; % shuffle
    params.train = params.train; %(1:min(opts.epochSize, numel(opts.train)));
    params.val = opts.val; %(randperm(numel(opts.val))) ;
    params.imdb = imdb ;
    params.getBatch = getBatch ;

    if numel(opts.gpus) <= 1
        [net, state] = processEpoch(net, state, params, 'train') ;
        [net, state] = processEpoch(net, state, params, 'val') ;
        if ~evaluateMode && ~mod(epoch, opts.saveInterval)
            saveState(modelPath(epoch), net, state) ;
        end
        lastStats = state.stats ;
    else
        spmd
            [net, state] = processEpoch(net, state, params, 'train') ;
            [net, state] = processEpoch(net, state, params, 'val') ;
            if labindex == 1 && ~evaluateMode && ~mod(epoch, opts.saveInterval)
                saveState(modelPath(epoch), net, state) ;
            end
            lastStats = state.stats ;
        end
        lastStats = accumulateStats(lastStats) ;
    end

    stats.train(epoch) = lastStats.train ;
    stats.val(epoch) = lastStats.val ;
    clear lastStats ;
    if ~mod(epoch, opts.saveInterval)
        saveStats(modelPath(epoch), stats) ;
    end

    if opts.plotStatistics
        switchFigure(1) ; clf ;
        plots = setdiff(...
            cat(2,...
            fieldnames(stats.train)', ...
            fieldnames(stats.val)'), {'num', 'time'}) ;
        for p = plots
            p = char(p) ;
            values = zeros(0, epoch) ;
            leg = {} ;
            for f = {'train', 'val'}
                f = char(f) ;
                if isfield(stats.(f), p)
                    tmp = [stats.(f).(p)] ;
                    values(end+1,:) = tmp(1,:)' ;
                    leg{end+1} = f ;
                end
            end
            subplot(1,numel(plots),find(strcmp(p,plots))) ;
            plot(1:epoch, values','o-') ;
            xlabel('epoch') ;
            title(p) ;
            legend(leg{:}) ;
            grid on ;
        end
        drawnow ;
        print(1, modelFigPath, '-dpdf') ;
    end

    if ~isempty(opts.postEpochFn)
        if nargout(opts.postEpochFn) == 0
            opts.postEpochFn(net, params, state) ;
        else
            [lr, params] = opts.postEpochFn(net, params, state) ;
            if ~isempty(lr), opts.learningRate = lr; end
            if opts.learningRate == 0, break; end
            opts.train = params.train;
        end
    end
end
logInfo('Complete: %d epochs.', opts.numEpochs);
if ~isempty(opts.postEpochFn)
    params = opts;
    params.imdb = imdb;
    params.epoch = params.numEpochs;
    params.getBatch = getBatch;
    opts.postEpochFn(net, params, state) ;
end

% With multiple GPUs, return one copy
if isa(net, 'Composite'), net = net{1} ; end


% -------------------------------------------------------------------------
function [net, state] = processEpoch(net, state, params, mode)
% -------------------------------------------------------------------------
% Note that net is not strictly needed as an output argument as net
% is a handle class. However, this fixes some aliasing issue in the
% spmd caller.

% initialize with momentum 0
if isempty(state) || isempty(state.solverState)
    state.solverState = cell(1, numel(net.params)) ;
    state.solverState(:) = {0} ;
end

% move CNN  to GPU as needed
numGpus = numel(params.gpus) ;
if numGpus >= 1
    net.move('gpu') ;
    for i = 1:numel(state.solverState)
        s = state.solverState{i} ;
        if isnumeric(s)
            state.solverState{i} = gpuArray(s) ;
        elseif isstruct(s)
            state.solverState{i} = structfun(@gpuArray, s, 'UniformOutput', false) ;
        end
    end
end
if numGpus > 1
    parserv = ParameterServer(params.parameterServer) ;
    net.setParameterServer(parserv) ;
else
    parserv = [] ;
end

% profile
if params.profile
    if numGpus <= 1
        profile clear ;
        profile on ;
    else
        mpiprofile reset ;
        mpiprofile on ;
    end
end

num = 0 ;
epoch = params.epoch ;
subset = params.(mode) ;
adjustTime = 0 ;

stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;

% -----------------------------------------
% -----------------------------------------
N = numel(subset);
if N > 0
logInfo('Epoch %d/%d: %s LR=%d', epoch, params.numEpochs, ...
    func2str(params.solver), params.learningRate);

% [KH] preprocess: group by meta-class
metaC = params.imdb.metaclass;
Nmeta = numel(metaC);
metaC_idx = cell(1, Nmeta);
for i = 1:Nmeta
    G = metaC{i}.groups;
    G = G(randperm(numel(G)));
    metaC_idx{i} = cat(1, G{:});
end
batchSize = params.batchSize;
assert(mod(batchSize, 2) == 0);
metaPairs = [];
for m = 1:Nmeta
    %metaPairs = [metaPairs; m*ones(Nmeta-m+1, 1), (m:Nmeta)'];
    metaPairs = [metaPairs; m*ones(Nmeta-m, 1), (m+1:Nmeta)'];
end
numPairs = size(metaPairs, 1);
metaPairs = metaPairs(randperm(numPairs), :);  % opt
numBatches = numPairs * params.batchesPerMetaPair;
logInfo('%d meta-classes, %d batches per meta-class pair', Nmeta, ...
    params.batchesPerMetaPair);

netopts = net.layers(end).block.opt;
if strcmp(mode, 'train')
    if params.batchSize > netopts.maxGpuImgs
        mode = 'train_staged';
    else
        net.mode = 'normal' ;
        net.conserveMemory = true;
    end
else
    net.mode = 'test' ;
end

start = tic;  a = tic;
objs = [];
cur = zeros(1, Nmeta);
t = 0;
for p = 1:numPairs
    % get 2 meta-classes
    m1 = metaPairs(p, 1);
    m2 = metaPairs(p, 2);
    for k = 1:params.batchesPerMetaPair
        if m1 == m2
            I = cur(m1) + (1:batchSize);
            I = mod(I-1, metaC{m1}.num) + 1;
            cur(m1) = I(end);
            batch = metaC_idx{m1}(I);
        else
            % get equal # of imgs from both meta-class
            I1 = cur(m1) + (1:batchSize/2);
            I2 = cur(m2) + (1:batchSize/2);
            I1 = mod(I1-1, metaC{m1}.num) + 1;
            I2 = mod(I2-1, metaC{m2}.num) + 1;
            cur(m1) = I1(end);
            cur(m2) = I2(end);
            batch = [metaC_idx{m1}(I1); metaC_idx{m2}(I2)];
        end
        t = t + batchSize;

        % train 1 batch
        if strcmp(mode, 'train')
            % regular train mode (batchSize within GPU limit)
            inputs = params.getBatch(params.imdb, batch) ;
            net.eval(inputs, params.derOutputs) ;

            if ~isempty(parserv), parserv.sync() ; end
            state = accumulateGradients(net, state, params, batchSize, parserv) ;

        elseif strcmp(mode, 'train_staged')
            % staged backprop mode
            % stage 1: get embedding matrix, in chunks
            net.mode = 'test';
            ind = net.getVarIndex('feats_l2');
            net.vars(ind).precious = 1;

            labelMat = [];  % label matrix
            embeddingMat = zeros(1, 1, netopts.dim, batchSize);  % embedding matrix
            if numel(params.gpus) > 0
                embeddingMat = gpuArray(embeddingMat);
            end

            subBatchSize = netopts.maxGpuImgs;
            nSub         = ceil(batchSize / subBatchSize);
            inputsCache  = cell(1, nSub);
            for s = 1:nSub
                sub = (s-1)*subBatchSize+1 : min(batchSize, s*subBatchSize);
                subInputs = params.getBatch(params.imdb, batch(sub));
                inputsCache{s} = subInputs;

                net.eval(subInputs);
                embeddingMat(:, :, :, sub) = gather(net.vars(ind).value);
                labelMat = [labelMat; subInputs{4}];
            end

            % stage 2: compute gradient matrix
            [gradientMat, obj_staged] = net.layers(end).block.computeGrad(...
                embeddingMat, labelMat);

            % stage 3: accumulate gradients
            net.mode = 'normal';
            for s = 1:nSub
                sub = (s-1)*subBatchSize+1 : min(batchSize, s*subBatchSize);
                featsGrad = gradientMat(:, :, :, sub);
                net.eval(inputsCache{s}, {'feats_l2', featsGrad});

                if ~isempty(parserv), parserv.sync() ; end
                state = accumulateGradients(net, state, params, numel(sub), parserv);
            end
        else
            % eval mode
            net.eval(inputs) ;
        end

        % Get statistics.
        time = toc(start) + adjustTime ;
        batchTime = time - stats.time ;
        stats.num = num ;
        stats.time = time ;
        stats = params.extractStatsFn(stats,net) ;
        currentSpeed = batchSize / batchTime ;
        averageSpeed = t / time ;
        if strcmp(mode, 'train_staged')
            objs = [objs, obj_staged];
        else
            objs = [objs, stats.objective];
        end
        if toc(a) > 30
            %if strcmp(mode, 'train_staged'), fprintf('\n'); end
            fprintf('%s-%s ep%02d: %3d/%3d (%d) %.1fHz', ...
                net.layers(end).block.opt.dataset, mode, epoch, ...
                fix(t/batchSize), numBatches, batchSize, averageSpeed) ;
            if strcmp(mode, 'train_staged')
                fprintf(' obj: %.3f (%.3f)\n', obj_staged, mean(objs(~isnan(objs))));
            else
                fprintf(' obj: %.3f (%.3f)\n', stats.objective, mean(objs(~isnan(objs))));
            end
            a = tic;
        end
    end  % for k
end  % for p
if strcmp(mode, 'train_staged')
    mode = 'train';
end
logInfo('Epoch %d, Avg %s obj = %g', epoch, mode, mean(objs(~isnan(objs))));
end  % if N>0

% Save back to state.
state.stats.(mode) = stats ;
if params.profile
    if numGpus <= 1
        state.prof.(mode) = profile('info') ;
        profile off ;
    else
        state.prof.(mode) = mpiprofile('info');
        mpiprofile off ;
    end
end
if ~params.saveSolverState
    state.solverState = [] ;
else
    for i = 1:numel(state.solverState)
        s = state.solverState{i} ;
        if isnumeric(s)
            state.solverState{i} = gather(s) ;
        elseif isstruct(s)
            state.solverState{i} = structfun(@gather, s, 'UniformOutput', false) ;
        end
    end
end

net.reset() ;
net.move('cpu') ;

% -------------------------------------------------------------------------
function state = accumulateGradients(net, state, params, batchSize, parserv)
% -------------------------------------------------------------------------
numGpus = numel(params.gpus) ;
otherGpus = setdiff(1:numGpus, labindex) ;

for p=1:numel(net.params)

    if ~isempty(parserv)
        parDer = parserv.pullWithIndex(p) ;
    else
        parDer = net.params(p).der ;
    end

    switch net.params(p).trainMethod
        case 'average' % mainly for batch normalization
            thisLR = net.params(p).learningRate ;
            net.params(p).value = vl_taccum(...
                1 - thisLR, net.params(p).value, ...
                (thisLR/batchSize/net.params(p).fanout),  parDer) ;

        case 'gradient'
            thisDecay = params.weightDecay * net.params(p).weightDecay ;
            thisLR = params.learningRate * net.params(p).learningRate ;

            if thisLR>0 || thisDecay>0
                % Normalize gradient and incorporate weight decay.
                parDer = vl_taccum(1/batchSize, parDer, ...
                    thisDecay, net.params(p).value) ;

                if isempty(params.solver)
                    % Default solver is the optimised SGD.
                    % Update momentum.
                    state.solverState{p} = vl_taccum(...
                        params.momentum, state.solverState{p}, ...
                        -1, parDer) ;

                    % Nesterov update (aka one step ahead).
                    if params.nesterovUpdate
                        delta = params.momentum * state.solverState{p} - parDer ;
                    else
                        delta = state.solverState{p} ;
                    end

                    % Update parameters.
                    net.params(p).value = vl_taccum(...
                        1,  net.params(p).value, thisLR, delta) ;

                else
                    % call solver function to update weights
                    [net.params(p).value, state.solverState{p}] = ...
                        params.solver(net.params(p).value, state.solverState{p}, ...
                        parDer, params.solverOpts, thisLR) ;
                end
            end
        otherwise
            error('Unknown training method ''%s'' for parameter ''%s''.', ...
                net.params(p).trainMethod, ...
                net.params(p).name) ;
        end
    end

    % -------------------------------------------------------------------------
    function stats = accumulateStats(stats_)
    % -------------------------------------------------------------------------
    for s = {'train', 'val'}
        s = char(s) ;
        total = 0 ;

        % initialize stats stucture with same fields and same order as
        % stats_{1}
        stats__ = stats_{1} ;
        names = fieldnames(stats__.(s))' ;
        values = zeros(1, numel(names)) ;
        fields = cat(1, names, num2cell(values)) ;
        stats.(s) = struct(fields{:}) ;

        for g = 1:numel(stats_)
            stats__ = stats_{g} ;
            num__ = stats__.(s).num ;
            total = total + num__ ;

            for f = setdiff(fieldnames(stats__.(s))', 'num')
                f = char(f) ;
                stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;

                if g == numel(stats_)
                    stats.(s).(f) = stats.(s).(f) / total ;
                end
            end
        end
        stats.(s).num = total ;
    end

    % -------------------------------------------------------------------------
    function stats = extractStats(stats, net)
    % -------------------------------------------------------------------------
    sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
    for i = 1:numel(sel)
        if net.layers(sel(i)).block.ignoreAverage, continue; end;
        stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average ;
    end

    % -------------------------------------------------------------------------
    function saveState(fileName, net_, state)
    % -------------------------------------------------------------------------
    net = net_.saveobj() ;
    save(fileName, 'net', 'state') ;
    logInfo('saved: %s', fileName);

    % -------------------------------------------------------------------------
    function saveStats(fileName, stats)
    % -------------------------------------------------------------------------
    if exist(fileName)
        try, save(fileName, 'stats', '-append') ; end
    else
        try, save(fileName, 'stats') ; end
    end

    % -------------------------------------------------------------------------
    function [net, state, stats] = loadState(fileName)
    % -------------------------------------------------------------------------
    load(fileName, 'net', 'state', 'stats') ;
    net = dagnn.DagNN.loadobj(net) ;
    if isempty(whos('stats'))
        warning('Epoch ''%s'' was only partially saved. Delete this file and try again.', ...
            fileName) ;
    end

    % -------------------------------------------------------------------------
    function epoch = findLastCheckpoint(modelDir)
    % -------------------------------------------------------------------------
    list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
    tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
    epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
    epoch = max([epoch 0]) ;

    % -------------------------------------------------------------------------
    function switchFigure(n)
    % -------------------------------------------------------------------------
    if get(0,'CurrentFigure') ~= n
        try
            set(0,'CurrentFigure',n) ;
        catch
            figure(n) ;
        end
    end

    % -------------------------------------------------------------------------
    function clearMex()
    % -------------------------------------------------------------------------
    clear vl_tmove vl_imreadjpeg ;

    % -------------------------------------------------------------------------
    function prepareGPUs(opts, cold)
    % -------------------------------------------------------------------------
    numGpus = numel(opts.gpus) ;
    if numGpus > 1
        % check parallel pool integrity as it could have timed out
        pool = gcp('nocreate') ;
        if ~isempty(pool) && pool.NumWorkers ~= numGpus
            delete(pool) ;
        end
        pool = gcp('nocreate') ;
        if isempty(pool)
            parpool('local', numGpus) ;
            cold = true ;
        end

    end
    if numGpus >= 1 && cold
        fprintf('%s: resetting GPU\n', mfilename)
        clearMex() ;
        if numGpus == 1
            gpuDevice(opts.gpus)
        else
            spmd
                clearMex() ;
                gpuDevice(opts.gpus(labindex))
            end
        end
    end
