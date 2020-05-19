function [P,A] = getProjection(A, p)
% get the p-dimensional projection matrix P induced by A
A = (A+A')/2;
[V,D] = eig(A);
[eval,ind] = sort(diag(D),'descend');
if(nargin < 2)
    p = sum(eval > 0);
end

V=V(:,ind(1:p));
eval=eval(1:p);
P=V*diag(sqrt(eval)); 
P = real(P);
A = P*P';
end