function imdb = inshop(prefix, opts)
savefn = fullfile(opts.dataDir, [prefix '.mat']);
if exist(savefn, 'file')
    imdb = load(savefn);
    return;
end
inshopdir = fullfile(opts.dataDir, 'InShopClothes');

% meta-classes
metaFile = fullfile(inshopdir, 'meta_classes.txt');
metaClasses = textread(metaFile, '%s');

% parse image names/labels
imgsFile = fullfile(inshopdir, 'Anno/list_bbox_inshop.txt');
[imgs, cType, pType, ~, ~, ~, ~] = textread(imgsFile, '%s %d %d %d %d %d %d', ...
    'headerlines', 2);
tmp  = cellfun(@(x) strsplit(x, '/'), imgs, 'uni', false);
cls  = cellfun(@(x) sscanf(x{4}, 'id_%d'), tmp);
meta = cellfun(@(x) [x{2} '/' x{3}], tmp, 'uni', false);
[~, meta] = ismember(meta, metaClasses);

% train/test
splitFile = fullfile(inshopdir, 'list_eval_partition.txt');
[img_sp, ~, sp] = textread(splitFile, '%s %s %s', 'headerlines', 2);
[~, loc] = ismember(imgs, img_sp);
[~, set] = ismember(sp(loc), {'train', 'query', 'gallery'});

% images are already 256x256, nothing to do

imdb.images.data = fullfile(inshopdir, imgs);
imdb.images.labels = single([cls meta cType pType]) ;
imdb.images.set = uint8(set) ;
imdb.meta.sets = {'train', 'query', 'gallery'} ;
save(savefn, '-struct', 'imdb');
end
