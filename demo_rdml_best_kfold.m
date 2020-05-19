clc;clearvars;rng('default'),rng(1);
addpath('.\common','.\data');

ds = 'yaleb32'; 
alg = 'rdml'; uAlg = upper(alg);
nl =0.15;
nlList = [0.0, .05, .1 .2];

evalAlg = 'knn';
saveflag = 1; 
CEIL_ACC = 100;

vsParams = {'C', 'eta', 'lr'};
CList = [.1 .01 .5 1  2 3 5 10 30 50 60 70 100, 150];
etaList = [.0001 .01 .03 .05 .05 .1 .5 1 2 3 5] ;
lrList = [.01 .05 .1 .3 0.5 .7 1 3 5]*1e-3 ;
maxIterList = [1,2,3,4,5];
maxIter2List = [100,1000, 1200];

%% load Data
[bestAcc, bestStd, params] = getBestRes(ds,alg, nl);
if(bestAcc == 0)
    params.normalize = 1; params.scale = 1;
    params.MaxIns = 1000; % to speed up computing "training kNN acc"
    params.margin = 1;
    params.kn = 3;
    params.nl = nl;
    params.eta = 1;
    params.lr = .001;
    params.solver = 'grad';
    params.maxIter = 3;
    params.maxIter2 = 1000;
    params.batchSize = 32;
    params.reducedDim = 0;
%     params.p = 120;
%     params.lnorm1 = 1;
end

cprintf('*blue','Best Accuracy:%0.2f+-%0.2f\n',bestAcc,bestStd); 
%% Specifying param values here, override the best setting for that params!!!!!!!!
params = override_best_params(params);

%%
options = gen_loadData_params(params);
[X,y,c,XTe,yTe] = loadData(ds, options);
[d,n] = size(X);
ind = randperm(n);
X = X(:,ind);
y = y(ind);

k = 5; params.k = k; %number of folds 
foldSize = floor(n / k); 

params.echo = 0;  
kn = params.kn;
showPlot = 1; 

testFlag = 0;
if(size(XTe,2) ~= 0)
   XTr = X; yTr = y;
   XTe_Orig = XTe;
   testFlag = 1;
end

for i = 1:length(vsParams)
    vsParam = vsParams{i};
    %% run && eval the algorithm
    results_vs = [];
    results_baseline_vs = [];
    vsList = [];
    eval(sprintf('vsList = %sList', vsParam));

    for vp = vsList
        corr = zeros(k,1); corrTr = zeros(k,1); 
        corr_baseline = zeros(k,1);
        runTime = zeros(k,1);
        % vsParam: name of the hyperparameter, vp: cur value of hyperparameter
        eval(sprintf('params.%s = %g', vsParam, vp));
        cprintf('*blue','optimize on parameter %s \n', vsParam);
        rng default;% for reproducibility of results
        rpt = cell(k,1);
        for t=1:k
            params.rng = 1;  rng(params.rng); % for reproducibility of results
            fprintf('processing fold #%d\n', t);
            XVal = X(:,(t-1)*foldSize+1:t*foldSize);
            yVal = y((t-1)*foldSize+1:t*foldSize);
            n_val = length(yVal);
            XTr = [X(:,1:(t-1)*foldSize),X(:,t*foldSize+1:end)];
            nTr = size(XTr,2);
            yTr = [y(1:(t-1)*foldSize),y(t*foldSize+1:end)];

            options.meanFlag =0; options.sampleReductionFlag = 0;
            options.additive = 1;
            yTrReal = yTr; params.yTrReal = yTrReal;
    
            if(params.nl > 0)
                [yTr, params.ind_labelNoise] = applyLabelNoise(yTr,c,params.nl);
            end
   
            [T,S,D] = genSDT(XTr,yTr,params.kn,params.margin);
            N = size(T,1);

            if(params.nl > 0)
                params.noisyTInd = find( (yTrReal(T(:,1)) ~= yTrReal(T(:,2))) | ...
                                     (yTrReal(T(:,1)) == yTrReal(T(:,3))) )';
                params.noiseFreeTInd = setdiff((1:size(T,1))',params.noisyTInd);
            end
            params.XTe = XVal; params.yTe = yVal; params.yTr = yTr;
            cprintf('*red','*************************run %d **************************\n',t);
            %% run algorithm
            tic;
            switch(uAlg)
                case 'RDML'
                    params.desc = sprintf('%s %s C=%g eta=%g lr=%g noisel=%g k=%d \n',...
                        alg,ds, params.C,params.eta,params.lr,params.nl,t);
                    cprintf('*comments',params.desc);
                    [M,W,~,rpt{t}] = rdml(XTr,T,params);
                case 'EUC'
                    W = eye(d,d);
            end
            runTime(t) = toc;

            %% eval algorithm
            [corr(t),corrTr(t)] = knn_test(W,XTr,yTr,XVal,yVal,kn,params.MaxIns);
            cprintf('blue','%s Test Correct Rate:%0.2f, t=%d\t\t',alg,corr(t),t);
            cprintf('red','%s Train Correct Rate:%0.2f, t=%d\n',alg,corrTr(t),t);
            corr_baseline(t) = knn_test(eye(d,d), XTr,yTr,XVal,yVal,params.kn,params.MaxIns);
            fprintf(2,'**********************************************************\n');

%             highCorr = (t*mean(corr(1:t)) + (k-t)*CEIL_ACC) / k;
%             if(highCorr < bestAcc)
%                 break;
%             end

        end %end for k
        corr = corr(1:t); corrTr = corrTr(1:t); runTime = runTime(1:t);
        corr_baseline = corr_baseline(1:t); 
        meanCorr = mean(corr); meanCorrTr = mean(corrTr);  meanStd = std(corr);
        meanRunTime = mean(runTime);
        cprintf('*comment','%s-%s %s num_folds =%0.2f noiseCoeff=%g C=%g eta=%g \n',...
             evalAlg,alg,ds,params.k, params.nl,params.C,params.eta);
        cprintf('*blue','Test Mean Correct Rate:%0.2f+-%0.2f  Run Time:%0.2f\n',...
                    meanCorr,meanStd, meanRunTime); 
        cprintf('*red','Train Mean Correct Rate:%0.2f\n',meanCorrTr);
        fprintf(2,'**********************************************************\n');
        results_vs = [results_vs,meanCorr];
        results_baseline_vs = [results_baseline_vs, mean(corr_baseline)];

        %% save results
        if(saveflag && meanCorr > bestAcc)
            bestAcc = meanCorr; bestStd = meanStd;
            fname = sprintf('.\\Rpt%s\\%s_%s_%g_%0.2f_%d.mat',...
                ds,alg,ds,params.nl,meanCorr,k);
            params = rmfield(params, {'XTe', 'yTe', 'yTr'});
            if(params.nl > 0)
                params = rmfield(params, {'noisyTInd','noiseFreeTInd'});
            end

            save(fname,'meanCorr', 'meanCorrTr', 'corr', 'corrTr', 'params', 'runTime');
        end
    end % for vp

    figure,
    lineWidth = 1.5;
    plot(1:length(vsList), results_vs,'LineWidth',lineWidth);
    hold on;
    plot(1:length(vsList), results_baseline_vs,'--','LineWidth',lineWidth);
    xticks(1:length(vsList))
    xticklabels(strsplit(num2str(vsList)));
    legend({upper(alg), 'Eucledian'});
    title(sprintf('%s accuracy using learned metrics vs %s values on %s dataset, Label Noise=%g%%',...
        alg,vsParam, ds, params.nl*100));
    xlabel(vsParam);
    ylabel('Accuracy')

    cprintf('*blue','Best Accuracy:%0.2f+-%0.2f\n',bestAcc,bestStd); 
    [bestAcc, bestStd, params] = getBestRes(ds,alg, nl);
    %% Specifying param values here, override the best setting for that params!!!!!!!!
    params = override_best_params(params);
end % for vsparams

function options = gen_loadData_params(params)
    options = struct();
    if(isfield(params,'biasFlag'))
        options.biasFlag = params.biasFlag;
    end
    if(isfield(params,'normalize'))
        options.normalize = params.normalize;
    end
    if(isfield(params,'normalize'))
        options.normalize = params.normalize;
    end
    if(isfield(params,'reducedDim'))
        options.reducedDim = params.reducedDim;
    end
    if(isfield(params,'lnorm1'))
        options.lnorm1 = params.lnorm1;
    end
end

function params = override_best_params(params)
     params.normalize = 1;
    %params.reducedDim = 20; % 0 means do not reduce dim using pca
%     params.lnorm1 = 1;
end



