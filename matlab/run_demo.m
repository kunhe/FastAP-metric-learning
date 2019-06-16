function run_demo(dataset, varargin)

% ----------------------------------------
% init
% ----------------------------------------
ip = inputParser;

if strcmp(dataset, 'products')
    ip.addParameter('Ks', [1 10 100 1000]);
    hardTrainFn = @train_hard;
    BPMP = 5;

elseif strcmp(dataset, 'inshop')
    ip.addParameter('Ks', [1 10 20 30 40 50]);
    hardTrainFn = @train_hard;
    BPMP = 2;

elseif strcmp(dataset, 'vid')
    ip.addParameter('Ks', [1 5]);
    hardTrainFn = @train_hard_vid;
    BPMP = 0;

else
    error('dataset not yet supported');
end

ip.KeepUnmatched = true;
ip.parse(varargin{:});
opts = ip.Results;
opts = get_opts(opts, dataset, varargin{:});

% post-parsing
cleanupObj = onCleanup(@() cleanup(opts.gpus));

opts = get_opts(opts);  % carry out all post-processing on opts
record_diary(opts);
disp(opts);

% ----------------------------------------
% model & data
% ----------------------------------------
[net, opts] = get_model(opts);

global imdb
imdb = get_imdb(imdb, opts);
disp(imdb.images)

% use imagenet-pretrained model
imgSize = opts.imageSize;
meanImage = single(net.meta.normalization.averageImage);
batchFunc = @(I, B) batch_imagenet(I, B, imgSize, meanImage);

% ----------------------------------------
% train
% ----------------------------------------
% figure out solver & learning rate
if strcmp(opts.solver, 'sgd')
    solverFunc = [];
    if opts.lrdecay>0 & opts.lrdecay<1
        cur_lr = opts.lr;
        lrvec = [];
        while length(lrvec) < opts.epoch
            lrvec = [lrvec, ones(1, opts.lrstep)*cur_lr];
            cur_lr = cur_lr * opts.lrdecay;
        end
    elseif opts.lrdecay > 1
        % linear decay, lrdecay specifies the # epoch -> 0
        assert(mod(opts.lrdecay, 1) == 0);
        lrvec = linspace(opts.lr, 0, opts.lrdecay+1);
        opts.epoch = min(opts.epoch, opts.lrdecay);
    end
else
    solverFunc = str2func(['solver.' opts.solver]);
    lrvec = opts.lr;
end

if opts.ablation == 3
    % random minibatch sampling
    [net, info] = train_rand(net, imdb, batchFunc , ...
        'saveInterval'       , opts.testInterval , ...
        'plotStatistics'     , opts.plot         , ...
        'randomSeed'         , opts.randseed     , ...
        'gpus'               , opts.gpus         , ...
        'continue'           , opts.continue     , ...
        'expDir'             , opts.expDir       , ...
        'batchSize'          , opts.batchSize    , ...
        'weightDecay'        , opts.wdecay       , ...
        'numEpochs'          , opts.epoch        , ...
        'learningRate'       , lrvec             , ...
        'train'              , imdb.images.PV    , ...
        'val'                , NaN               , ...
        'solver'             , solverFunc        , ...
        'postEpochFn'        , @postepoch);
else
    % hard minibatch sampling
    [net, info] = hardTrainFn(net, imdb, batchFunc , ...
        'saveInterval'       , opts.testInterval , ...
        'plotStatistics'     , opts.plot         , ...
        'randomSeed'         , opts.randseed     , ...
        'gpus'               , opts.gpus         , ...
        'continue'           , opts.continue     , ...
        'expDir'             , opts.expDir       , ...
        'batchSize'          , opts.batchSize    , ...
        'weightDecay'        , opts.wdecay       , ...
        'numEpochs'          , opts.epoch        , ...
        'learningRate'       , lrvec             , ...
        'val'                , NaN               , ...
        'solver'             , solverFunc        , ...
        'postEpochFn'        , @postepoch        , ...
        'batchesPerMetaPair' , BPMP);
end

% ----------------------------------------
% done
% ----------------------------------------
net.reset();
net.move('cpu');
diary('off');

end

% ====================================================================
% get IMDB
% ====================================================================

function imdb = get_imdb(imdb, opts)
imdbName = sprintf('%s_%d', opts.dataset, opts.imageSize);
logInfo('IMDB: %s', imdbName);

if ~isempty(imdb) && strcmp(imdb.name, imdbName)
    return;
end

% load/compute
t0 = tic;
imdbFunc = str2func(['imdb.' opts.dataset]);
imdb = imdbFunc(['imdb_' imdbName], opts) ;
imdb.name = imdbName;

if opts.ablation == 3
    % random MBS: shuffle training instances
    imdb.images.PV = shuffle(imdb, opts);
else
    % hard MBS: group by meta-class
    if ~isfield(imdb, 'metaclass') || isempty(imdb.metaclass)
        logInfo('Analyzing meta-classes...'); tic;

        Ycls  = imdb.images.labels(imdb.images.set==1, 1);
        Ymeta = imdb.images.labels(imdb.images.set==1, 2);
        Umeta = unique(Ymeta);
        imdb.metaclass = cell(1, numel(Umeta));

        for i = 1:numel(Umeta)
            % group instances in this metaclass into cells
            Im = find(Ymeta == Umeta(i));  % images in this metaclass
            Ic = unique(Ycls(Im));         % instances in this metaclass
            G  = arrayfun(@(x) find(Ycls==x & Ymeta==Umeta(i)), Ic, 'uniform', 0);

            % store in imdb
            imdb.metaclass{i} = [];
            imdb.metaclass{i}.id = Umeta(i);
            imdb.metaclass{i}.num = numel(Im);
            imdb.metaclass{i}.groups = G;
        end
        toc;
    end
end

logInfo('%s in %.2f sec', imdbName, toc(t0));
end

% ====================================================================
% postprocessing after each epoch
% ====================================================================

function [lr, params] = postepoch(net, params, state)
lr = [];
if isa(net, 'composite')
    net = net{1};
end
opts = net.layers(end).block.opt;
epoch = params.epoch;
logInfo(opts.expID);
logInfo(opts.identifier);
logInfo(char(datetime));

if ~isempty(opts.gpus)
    [~, name] = unix('hostname');
    logInfo('GPU #%d on %s', opts.gpus, name);
end

% evaluate
if (epoch==1) | ~mod(epoch, opts.testInterval);
    evaluate_model(net, params.imdb, params.getBatch, opts);
end

% reshuffle
if isfield(opts, 'ablation') && opts.ablation == 3
    params.train = shuffle(params.imdb, opts);
end
diary off, diary on
end


function PV = shuffle(imdb, opts)
%  group by class label, then shuffle
Itrain = find(imdb.images.set == 1);
Ytrain = imdb.images.labels(Itrain, 1);  
C  = arrayfun(@(y) find(Ytrain==y), unique(Ytrain), 'uniform', false);
PV = [];

% go through classes in random order
for j = randperm(numel(C))
    % randomize the instances
    ind = C{j};
    ind = ind(randperm(length(ind)));
    PV  = [PV; ind];
end

assert(length(PV) == length(Itrain));
PV = Itrain(PV);
end

% ====================================================================
% misc
% ====================================================================

function cleanup(gpuid)
    diary('off');
end
