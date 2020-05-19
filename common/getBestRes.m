function [bestAcc, bestStd, bestParams, bestCorr] = getBestRes(ds,alg,nl, path_res)
    if(nargin == 3)
        path_res = '.\\Rpt';
    end
    dirName = [path_res, ds];
%     dirName = sprintf('.\\Rpt%s', ds);
    files = dir(dirName);
    filenames = string({files.name});
    ind = contains(filenames,alg,'IgnoreCase',true);
    filenames = filenames(ind);
    n = length(filenames);
    bestAcc = 0; bestStd = 0; bestCorr = []; bestParams = [];
    for i=1:n
       load([dirName,'\\',filenames{i}],'meanCorr', 'params', 'corr');
       if(isfield(params, 'nl'))
           noise_level = params.nl;
       else
           noise_level = params.lNoiseCoeff;
       end
       if(noise_level == nl && meanCorr > bestAcc)
           bestAcc = meanCorr;
           bestStd = std(corr);
           bestParams = params;
           bestCorr = corr;
       end
    end
end
   

