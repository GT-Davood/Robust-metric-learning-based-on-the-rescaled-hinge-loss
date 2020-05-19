function [loss,nImp] = l_hinge(X,T,W,weight,margin)
    I = T(:,1); J = T(:,2); K = T(:,3);
    Z = W'*X;
    Zi = Z(:,I); Zj = Z(:,J); Zk = Z(:,K);
    
    distij =  sum((Zi-Zj).* (Zi-Zj),1);
    distik =  sum((Zi-Zk).* (Zi-Zk),1);
    loss = margin + (distij - distik)';
    loss(loss < 0) = 0;
    loss = weight.* loss;
    nImp = sum(loss>0);
end