function [M,W,weight,rpt] = rdml(XTr,T,params)
    global I J K
    I = T(:,1); J = T(:,2); K = T(:,3);
    d = size(XTr,1);
    M = eye(d); W = eye(d);
    N = size(T,1);
    weight = ones(N,1);
    beta = 1/(1-exp(-params.eta));
    if(isfield(params,'plotFunc'))
        weight = beta*params.C*params.eta*exp(-params.eta*l_hinge(XTr,T,W,ones(N,1),params.margin));
        params.plotFunc(XTr, params.yTr,T,weight,params.noisyInd);
    end
    echo = params.echo;
    params.echo = params.echo; % grad
    params.dispProgress = echo;
    rpt = [];
    
    if(~echo)
        textprogressbar('Percent:','init');
    end
        
    for i=1:params.maxIter
        % optimize weights
        weight = beta*params.C*params.eta*exp(-params.eta*l_hinge(XTr,T,W,ones(N,1),params.margin));
        % optimize metric
        if(strcmpi(params.solver,'grad'))
            [M,W] = whml(XTr,T,weight,params);
        else
            [M,W] = wTripletSVM(XTr,T,weight,params);
        end
        if(isfield(params,'warmStart') && params.warmStart)
            params.W0 = W; params.M0 = M;
        end
        
        if (echo)
            objFunc = beta* (N - sum(exp(-params.eta*l_hinge(XTr,T,W,ones(N,1),params.margin))) );
            fprintf('obj func = %0.2f, mean weight:%0.3f, mean noisy:%0.3f, mean weight noise free:%0.3f\n'...
              ,objFunc,mean(weight), mean(weight(params.noisyTInd)), mean(weight(params.noiseFreeTInd)) );
                  nWeight = 1e5* weight./sum(weight) ;
            fprintf('mean normal weight:%0.3f, mean normal noisy:%0.3f, mean normal noise free:%0.3f\n\n'...
              ,mean(nWeight), mean(nWeight(params.noisyTInd)), mean(nWeight(params.noiseFreeTInd)) );
        
            [corr,corrTr] = knn_test(W,XTr,params.yTr,params.XTe,params.yTe,params.kn);
            cprintf('blue','RML Test Correct Rate:%0.2f, iter=%d\t\t',corr,i);
            cprintf('red','RML Train Correct Rate:%0.2f, iter=%d\n',corrTr,i);
            rpt(i,:) = [corr, corrTr];
        else
             textprogressbar(100*i/params.maxIter);
        end
    end
    if(~echo)
        textprogressbar('finished!!!','stop');
    end
end