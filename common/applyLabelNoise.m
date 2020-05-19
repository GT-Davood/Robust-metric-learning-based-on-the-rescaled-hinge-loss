function [y,noisyInd] = applyLabelNoise(y,c,coeff)
    N = length(y);
    m = round(coeff*N); % number of examples to apply noise
    ind = randperm(N);
    noisyInd = ind(1:m);
    noisyLabels = mod(y(noisyInd)-1 + randi([1,c-1],1,m),c) + 1; 
    y(noisyInd) = noisyLabels;
end