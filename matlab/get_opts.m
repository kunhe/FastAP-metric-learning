function opts = get_opts(opts, dataset, varargin)

if nargin == 1
    opts = process_opts(opts);
    return;
end

ip = inputParser;

% model params
ip.addParameter('obj'       , 'FastAP');
ip.addParameter('arch'      , 'resnet18');  % CNN model
ip.addParameter('dim'       , 512);      % embedding vector size
ip.addParameter('nbins'     , 10);       % quantization granularity

% SGD
ip.addParameter('solver'    , 'adam');
ip.addParameter('batchSize' , 256);
ip.addParameter('lr'        , 1e-5);   % base learning rate
ip.addParameter('lrmult'    , 10);     % learning rate multiplier for last layer
ip.addParameter('lrdecay'   , 0.1);    % [SGD only] LR decay factor
ip.addParameter('lrstep'    , 10);     % [SGD only] decay LR after this many epochs
ip.addParameter('wdecay'    , 0);      % weight decay

% train
ip.addParameter('testInterval', 3);   % test interval 
ip.addParameter('epoch'    , 30);     % num. epochs
ip.addParameter('gpus'     , 1);      % which GPU to use
ip.addParameter('continue' , true);   % resume from existing model
ip.addParameter('debug'    , false);
ip.addParameter('plot'     , false);

% misc
ip.addParameter('ablation', 0);  % 0: none, 1: batchSize, 2: nbins, 3: randMBS
ip.addParameter('randseed', 42);
ip.addParameter('prefix'  , []);

% parse input
ip.KeepUnmatched = true;
ip.parse(varargin{:});
opts = catstruct(ip.Results, opts);  % combine w/ existing opts
opts.dataset = dataset;

end

% =======================================================================
% =======================================================================

function opts = process_opts(opts)
% post-parse processing

opts.expID = sprintf('%s-%s-%d-%s', opts.dataset, opts.obj, opts.dim, opts.arch);

switch (opts.ablation)
    case 1
        opts.prefix = 'ablationM';
    case 2
        opts.prefix = 'ablationHist';
    case 3
        opts.prefix = 'randMBS';
    otherwise
        opts.prefix = '';
end

% for batch sizes > GPU mem limit: use chunks
switch (opts.arch)
    case 'resnet50'
        opts.maxGpuImgs = 90;
    case 'resnet18'
        opts.maxGpuImgs = 256;
    case 'googlenet'
        opts.maxGpuImgs = 320;
end

% --------------------------------------------
% identifier string for the current experiment
ID = sprintf('%dbins-batch%d-%s%.emult%g', opts.nbins, opts.batchSize, ...
    opts.solver, opts.lr, opts.lrmult);

if strcmp(opts.solver, 'sgd')
    assert(opts.lrdecay > 0 && opts.lrdecay < 1);
    assert(opts.lrstep > 0);
    ID = sprintf('%sD%gE%d', ID, opts.lrdecay, opts.lrstep);
end
ID = sprintf('%s-WD%.e', ID, opts.wdecay);

if isempty(opts.prefix)
    % prefix: timestamp
    [~, T] = unix(['git log -1 --format=%ci|cut -d " " -f1,2|cut -d "-" -f2,3' ...
        '|tr " " "."|tr -d ":-"']);
    opts.prefix = strrep(T, newline, '');
end
opts.identifier = [opts.prefix '-' ID];

% --------------------------------------------
% mkdirs
opts.localDir = fullfile(pwd, 'cachedir');  % use symlink on linux
if ~exist(opts.localDir, 'file')
    error('Please mkdir/symlink cachedir!');
end
opts.dataDir = fullfile(opts.localDir, 'data');
opts.imdbPath = fullfile(opts.dataDir, ['imdb_' opts.dataset]);

opts.expDir = fullfile(opts.localDir, opts.expID, opts.identifier);
if ~exist(opts.expDir, 'dir')
    logInfo(['creating opts.expDir: ' opts.expDir]);
    mkdir(opts.expDir);
end

% --------------------------------------------
% rng
rng(opts.randseed);

end
