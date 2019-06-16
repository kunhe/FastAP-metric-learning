function imdb = products(prefix, opts)
savefn = fullfile(opts.dataDir, [prefix '.mat']);
if exist(savefn, 'file')
    imdb = load(savefn);
    return;
end
stanforddir = fullfile(opts.dataDir, 'Stanford_Online_Products');

% train/test
[id, y1, y2, x] = textread([stanforddir '/Ebay_train.txt'], '%d %d %d %s', ...
    'headerlines', 1);
fn = x;
labels = [y1 y2];
set = ones(numel(id), 1);
logInfo('train: %d', numel(x));

[id, y1, y2, x] = textread([stanforddir '/Ebay_test.txt'], '%d %d %d %s', ...
    'headerlines', 1);
fn = [fn; x];
labels = [labels; y1 y2];
set = [set; 3*ones(numel(id), 1)];
logInfo(' test: %d', numel(x));

% resize to 256x256
a = tic; tic;
stanford256 = fullfile(stanforddir, 'images_256x256');
if ~exist(stanford256, 'dir'), mkdir(stanford256); end
fn256 = strrep(fn, '/', '_');
fn256 = cellfun(@(x) fullfile(stanford256, x), fn256, 'uniform', false);
for i = 1:numel(fn)
    if ~exist(fn256{i}, 'file')
        im = imread([stanforddir '/' fn{i}]);
        im = imresize(im, [256 256]);
        imwrite(im, fn256{i});
    end
    if toc > 30
        logInfo('%d/%d: %s', i, numel(fn), fn256{i}); tic;
    end
end
toc(a)

imdb.images.data = fn256;
imdb.images.labels = single(labels) ;
imdb.images.set = uint8(set) ;
imdb.meta.sets = {'train', 'val', 'test'} ;
save(savefn, '-struct', 'imdb');
end
