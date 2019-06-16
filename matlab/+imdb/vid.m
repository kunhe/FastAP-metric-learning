function imdb = vid(prefix, opts)
savefn = fullfile(opts.dataDir, [prefix '.mat']);
if exist(savefn, 'file')
    imdb = load(savefn);
    return;
end
viddir = fullfile(opts.dataDir, 'VehicleID_V1.0');

% imgs, vid
img2vid = fullfile(viddir, 'attribute', 'img2vid.txt');
[imgs, vid] = textread(img2vid, '%d %d');
n = numel(imgs);

% model info
mid = zeros(n, 1);
vid2model = fullfile(viddir, 'attribute', 'model_attr.txt');
[vid_m, mid_m] = textread(vid2model, '%d %d');
for i = 1:length(vid_m)
    mid(vid == vid_m(i)) = mid_m(i) + 1;
end

% color info
cid = zeros(n, 1);
vid2color = fullfile(viddir, 'attribute', 'color_attr.txt');
[vid_c, cid_c] = textread(vid2color, '%d %d');
for i = 1:length(vid_c)
    mid(vid == vid_c(i)) = cid_c(i) + 1;
end

% train/test
sets = zeros(n, 4);  % [train test_small test_medium test_large]

trainFile = fullfile(viddir, 'train_test_split', 'train_list.txt');
[~, vid_train] = textread(trainFile, '%d %d');
sets(ismember(vid, vid_train), 1) = 1;
logInfo('train: %d imgs', sum(sets(:, 1)));

testSmallFile = fullfile(viddir, 'train_test_split', 'test_list_800.txt');
[~, vid_s] = textread(testSmallFile, '%d %d');
sets(ismember(vid, vid_s), 2) = 1;
logInfo('test-small: %d imgs', sum(sets(:, 2)));

testMediumFile = fullfile(viddir, 'train_test_split', 'test_list_1600.txt');
[~, vid_m] = textread(testMediumFile, '%d %d');
sets(ismember(vid, vid_m), 3) = 1;
logInfo('test-medium: %d imgs', sum(sets(:, 3)));

testLargeFile = fullfile(viddir, 'train_test_split', 'test_list_2400.txt');
[~, vid_l] = textread(testLargeFile, '%d %d');
sets(ismember(vid, vid_l), 4) = 1;
logInfo('test-large: %d imgs', sum(sets(:, 4)));

% resize to 256x256
a = tic; tic;
vid256 = fullfile(viddir, 'image_256x256');
if ~exist(vid256, 'dir'), mkdir(vid256); end
fn256 = cell(n, 1);
for i = 1:n
    fn256{i} = sprintf('%s/%07d.jpg', vid256, imgs(i));
    if ~exist(fn256{i}, 'file')
        im = imread(sprintf('%s/image/%07d.jpg', viddir, imgs(i)));
        im = imresize(im, [256 256]);
        imwrite(im, fn256{i});
    end
    if toc > 20
        logInfo(fn256{i}); tic;
    end
end
toc(a)

% imdb
imdb.images.data = fn256;
imdb.images.labels = single([vid mid cid]) ;
imdb.images.set = uint8(sets) ;
imdb.meta.sets = {'train', 'test_small', 'test_medium', 'test_large'} ;
save(savefn, '-struct', 'imdb');
end
