% -------------------------------------------------------------------------
function prepareGPUs(params, cold)
% -------------------------------------------------------------------------
numGpus = numel(params.gpus) ;
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
    fprintf('%s: resetting GPU\n', mfilename) ;
    %clearMex() ;
    if numGpus == 1
        disp(gpuDevice(params.gpus)) ;
    else
        spmd
            %clearMex() ;
            disp(gpuDevice(params.gpus(labindex))) ;
        end
    end
end
