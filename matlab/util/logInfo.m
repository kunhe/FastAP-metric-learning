function logInfo(str, varargin)
% get caller function ID and display log msg
if nargin > 1
    cmd = 'sprintf(str';
    for i = 1:length(varargin)
        cmd = sprintf('%s, varargin{%d}', cmd, i);
    end
    str = eval([cmd, ');']);
end 
[st, i] = dbstack();
caller = st(2).name;
fprintf('%s: %s\n', caller, str);
end
