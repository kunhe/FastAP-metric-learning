function record_diary(opts)
diary_path = @(i) sprintf('%s/diary_%03d.txt', opts.expDir, i);
ind = 1;
while exist(diary_path(ind), 'file')
    ind = ind + 1;
end
diary(diary_path(ind));
diary('on');
end
