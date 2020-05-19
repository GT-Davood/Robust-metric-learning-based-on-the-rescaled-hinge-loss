function [M,W] = whml(X,T,weight,params)
    % Weighted Hinge Metric Learning
    % mininmize reg/2*normFro(M) + sum(wi*l_hinge(xi,x'i,x"i))
    if(isfield(params,'maxIter2'))
        maxIter = params.maxIter2;
    else
        maxIter = 1000;
    end
    
    if (isfield(params,'dispProgress'))
        dispProgress = params.dispProgress;
    else 
        dispProgress = 0;
    end

    [d,~] = size(X); N=size(T,1);
    I = T(:,1); J = T(:,2); K = T(:,3);
    srw = sqrt(weight)';
    Dij = bsxfun(@times,X(:,I) - X(:,J),srw);
    Dik = bsxfun(@times,X(:,I) - X(:,K),srw);
   if(isfield(params,'M0'))
       M = params.M0;
       W = params.W0;
   else
      M = eye(d);
      W = eye(d);
   end

    if(params.echo)
        [~,err,~,nImp] = compLoss(X,T,M,W,weight,params);
        displayRes(0,err,nImp,inf);
    elseif(dispProgress)
        textprogressbar('Percent:','init');
    end
    
    maxTrial = 5;
    for iter=1:maxIter
        
        if(params.batchSize < N)
             selInd = randi(N,1,params.batchSize);
        else
            selInd = 1:N;
        end
        
        % compute loss and find active constraints
        [~,err,activeInd,nImp] = compLoss(X,T(selInd,:),M,W,weight(selInd),params);
        activeInd = selInd(activeInd);

        % compute gradient
        G = zeros(d,d);
        B = 3000;
        for i=1:B:nImp
            BB=min(B-1,nImp-i);
            ind = activeInd(i:i+BB);
            G = G + (Dij(:,ind)*Dij(:,ind)' - Dik(:,ind)*Dik(:,ind)');
        end
        G = eye(d) + G;
%         G = M + G;

        %update metric
        lr = params.lr;
        for j=1:maxTrial
            M_new = M - lr* G;
            if(isfield(params,'p'))
                [W,M_new] = getProjection(M_new,params.p);
            else
                [W,M_new] = getProjection(M_new);
            end
            [~,err_new,~,~] = compLoss(X,T(selInd,:),M,W,weight(selInd),params);
            if(err_new < err)
                break;
            else
                lr = lr /2;
            end
        end

%         if(err_new > err || abs(err_new - err)/err < params.tresh)
%             if(params.echo)
%                 displayRes(iter,err,nImp,norm(G,'fro'));
%             end
%             break;
%             fprintf('break\n');
%         end
        M = M_new;
        if(params.echo)
            if(mod(iter,20) == 0)
                [loss,~,~,nImp] = compLoss(X,T(selInd,:),M,W,weight(selInd),params);
                displayRes(iter,err,nImp,norm(G,'fro'));
%                 w = params.C*exp(-loss);
%                 w = 1e3* w./sum(w);
%                 fprintf('mean weight:%0.3f, mean noisy:%0.3f, mean weight noise free:%0.3f\n\n'...
%                     ,mean(w), mean(w(params.noisyTInd)), mean(w(params.noiseFreeTInd)) );
            end
        elseif(dispProgress)
 
            textprogressbar(100*iter/maxIter);
        end

    end
    if(~params.echo && dispProgress)
        textprogressbar('finished!!!','stop');
    end
end

function [loss,err,activeInd,nImp] = compLoss(X,T,M,W,weight,params)
    loss = l_hinge(X,T,W,weight,params.margin);
    err =  params.C* sum(loss) + .5*norm(M,'fro');
    activeInd = find(loss > 0);
    nImp = size(activeInd,1);
end

function displayRes(iter,err,nImp,nG)
    fprintf('iter=%d, err=%g, nImp=%d, norm G=%0.3f \n',iter,err,nImp,nG);
end





