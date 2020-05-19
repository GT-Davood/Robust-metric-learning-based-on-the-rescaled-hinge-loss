function [corr,corrTr,cp,cpTr,yhatTe] = knn_test(W,XTr,yTr,XTe,yTe,kn,MaxIns)
    W = real(W);
    Z = XTr'*W;
% %     mdl = fitcknn(Z,yTr,'NumNeighbors',kn);
    nTe = size(XTe,2);
    yhatTe = zeros(1,nTe);
    ZTe = XTe'*W;
    B = 3000;
    for i=1:B:nTe
        BB=min(B-1,nTe-i);
%         yhatTe(i:i+BB) = predict(mdl, ZTe(i:i+BB,:));
        dist = pdist2(ZTe(i:i+BB,:),Z);
        [~,ind] = sort(dist,2);
        ind = ind(:,1:kn);
        yhatTe(i:i+BB)=mode(yTr(ind),2);
    end
    cp = classperf(yTe, yhatTe);
    corr = cp.CorrectRate*100;
    
    N = size(XTr,2);
    if(nargin >= 7 && N > MaxIns)
        indSel = randi(N,1,MaxIns);
    else
        indSel = 1:N;
    end
    nSel = length(indSel);
    classoutTr = zeros(nSel,1);

    for i=1:B:nSel
        BB=min(B-1,nSel-i);
%         classoutTr(i:i+BB) = predict(mdl, Z(indSel(i:i+BB),:));
        dist = pdist2(Z(indSel(i:i+BB),:), Z);
        [~,ind] = sort(dist,2);
        ind = ind(:,1:kn);
        classoutTr(i:i+BB)=mode(yTr(ind),2);
    end
    
    cpTr = classperf(yTr(indSel), classoutTr);
    corrTr = cpTr.CorrectRate*100;
end