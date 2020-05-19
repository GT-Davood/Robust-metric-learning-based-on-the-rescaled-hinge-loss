%Parameters:
% dsName: name of dataset
% params: rowFlag=1: each row represents an observation
%         biasFlag=1: bias term 1 will be added as the last feature
%         biasFlag=2: bias term 1 will be added as the first feature 
%         normalize=1 each feature normalized to have zero mean and std=1  
function [X,y,c,XTe,yTe] = loadData(dsName, params)
    X = []; XTe = []; y = []; yTe = [];
    rowFlag = 0; biasFlag = 0; normalize = 0; l2norm1 = 0; lnorm1 = 0;
    if(nargin >= 2)
        if(isfield(params,'rowFlag'))
            rowFlag = params.rowFlag;
        end
        if(isfield(params,'biasFlag'))
            biasFlag = params.biasFlag;
        end
        if(isfield(params,'normalize'))
            normalize = params.normalize;
        end
        if(isfield(params,'l2norm1'))
            l2norm1 = params.l2norm1;
        end
        if(isfield(params,'lnorm1'))
            lnorm1 = params.lnorm1;
        end
        
    end
    
    switch lower(dsName)
        case 'australian'
            load australian X y c
            y(y == -1) = 2;
        case {'synthetic_out1', 'synthetic_out2', 'synthetic_out3'}  
            eval(sprintf('func = @generate_synthetic_data%s', dsName(end)));  
            [X,y,~,XTe, yTe] = func(params.num_data, params.out_coeff, ...
                params.out_intensity, params.test_ratio);
            c = length(unique(y));
        case {'iris','wine','crab','glass','cancer','thyroid'}
            exp = sprintf('[X,l] = %s_dataset;',dsName);
            eval(exp);
            [~,N] = size(X);
            y = zeros(1,N);
            c = size(l,1); % number of classes
            for i=1:c
               y(l(i,:) == 1) = i;
            end
            
            if(strcmp(dsName,'thyroid'))
                [~,X,~] = pca(X', 'NumComponents',18);
                X = X';
            end
        case 'diabetes' 
            load('diabetes.mat');
            y = y + 1;
            c = 2;
        case 'banana' 
            load banana.mat X y;
            c = 2;
        case 'wdbc'
            load wdbc.mat X y c
        case 'glass6'
            load glass6.mat X y c;
        case 'spambase'
            load spambase.mat X y c;
        case {'syn2_lnoise', 'syn2_lnoise_ver2', 'syn2_lnoise_ver3'}
            load(dsName,'X','y','c');
        case 'waveform'
            load waveform.mat X y c;
        case 'satimage'
            load satimage.mat X y c;
            [~,X,~] = pca(X', 'NumComponents',36);
            X = X';
        case 'vehicle'
            load vehicle.mat X y XTe yTe c;
        case 'ionosphere'
            load ionosphere.mat X y c
              
        case {'pima','germancredit','svmguide1','a5a', 'magic_gamma'}
            load(dsName,'X', 'y', 'c');
            y = y';
            y(y == -1) = 2;
        case 'kddcup99'
            load ('KDDCup99_Data.mat');
            X = [Normal;Dos;Rtol;Utor;Probe];
            [~,X,~] = pca(X, 'NumComponents',20);
            X = X';
            [~, N] = size(X);
            y = zeros(1,N);
            n = 0;
            y(n+1:n+size(Normal,1))=1; n = n + size(Normal,1);
            y(n+1:n+size(Dos,1))=2;    n = n + size(Dos,1);
            y(n+1:n+size(Rtol,1))=3;   n = n + size(Rtol,1);
            y(n+1:n+size(Utor,1))=4;   n = n + size(Utor,1);
            y(n+1:n+size(Probe,1))=5;  
            c = 5;
        case 'mnist'
            X = loadMNISTImages('train-images.idx3-ubyte');
            y = loadMNISTLabels('train-labels.idx1-ubyte')';
            y = y+1;
            m = mean(X,2);
            [W,X,~] = pca(X', 'NumComponents',100);
            X = X';
            XTe = loadMNISTImages('t10k-images.idx3-ubyte');
            yTe = loadMNISTLabels('t10k-labels.idx1-ubyte')';
            yTe = yTe + 1;
            XTe = bsxfun(@minus,XTe,m);
            XTe =W'*XTe;  

            [y,sortInd] = sort(y);
            X = X(:,sortInd);
            c = 10;
        case 'usps'
           load usps.mat xTe xTr yTe yTr
           y = yTr;
           m = mean(xTr,2);
           [W,X,~] = pca(xTr', 'NumComponents',100);
           X = X';
%            xTe = bsxfun(@minus,xTe,m);
           xTe = bsxfun(@minus,xTe,mean(xTe,2));
           XTe =W'*xTe;  
           [y,sortInd] = sort(y);
            X = X(:,sortInd);
           c = 10;
        case 'isolet'
            X = load('isolet1+2+3+4.data');
            y = X(:,end)';
            XTe = load('isolet5.data');
            yTe = XTe(:,end)';
%             [~,T,~] = pca([X(:,1:end-1);XTe(:,1:end-1)],'NumComponents',172);
%             T = bsxfun(@minus,T,mean(T));
%             X = T(1:n,:)';
%             XTe = T(n+1:end,:)';
            X = X'; XTe = XTe';
            [y,sortInd] = sort(y);
            X = X(:,sortInd);
            c = 26;
        case 'isolet_mixed'
            X = load('isolet1+2+3+4.data');
            XTe = load('isolet5.data');
            [~,T,~] = pca([X(:,1:end-1);XTe(:,1:end-1)],'NumComponents',172);
            y = [X(:,end)', XTe(:,end)'];
            X = T';
            [y,sortInd] = sort(y);
            X = X(:,sortInd);
            XTe = [];
        case 'letters'
            load('letter_dataset.mat');
            c = 26;
        case {'orl32','orl64','yaleb32'}
            fea = [];
            load(dsName);
            d = 200;
            if(strcmp(dsName,'orl64') == 1)
                d = 64;
            end
            if(strcmp(dsName,'yaleb32'))
                lnorm1 = 1;
            end

            [~,fea,~] = pca(fea, 'NumComponents',d);
            X = fea'; y = gnd';
            c = max(y);
        case {'caltech10','caltech20','caltech50'}
            c = str2double(dsName(end-1:end));
%             [X,y,XTe,yTe] = loadCaltech101(c);
%             c = max(y);
            if(c == 10)
                 load caltech10.mat X y XTe yTe c
            elseif (c == 20)
                 load caltech20.mat X y XTe yTe c  
            else
                load caltech50.mat X y XTe yTe c
            end
        case {'caltech101','caltech256','indoor','oxford_cats_dogs'}
             load(dsName) 
             eval('m = length(methods);');
             [y,ind] = sort(y);
             X = cell(1,m);
             for j=1:m
                eval(sprintf('X{%d} = X_%d(:,ind); clear X_%d',j,j,j));
             end
        case 'caltech101-resnet-152'
             load(dsName);
        case 'caltech256-resnet-152'
             load(dsName);
        case 'cifar10-resnet-152'
             load(dsName);
        case 'oxford_cats_dogs-resnet-152'
             load(dsName);
        case {'cifar10'}
             load(dsName) 
             eval('m = length(methods);');
             nTr = 50000;
             yTe = y(nTr+1:end);
             y = y(1:nTr);
             [y,ind] = sort(y);
             X = cell(1,m);
             XTe = cell(1,m);
             for j=1:m
                eval(sprintf('X{%d} = X_%d(:,ind); XTe{%d} = X_%d(:,nTr+1:end); clear X_%d',j,j,j,j,j));
             end
        case 'pavia'
            load('PaviaU.mat', 'paviaU')
            load('PaviaU_gt.mat');
            img = paviaU; gt = paviaU_gt; 
            c = max(unique(gt(:)));
            [X,y] = getPatterns(img, gt);
        case {'indiana','indiana12'}
            load indiana_imgreal; % original data
            sz = size(img);
            img = reshape(img, [prod(sz(1:2)) sz(3)]);   
            img = img';
            img([104:108 150:163 220],:) =[]; % remove noise bands
            img= reshape(img',[145 145 200]);
            % load ground truth data
            load Indiana_16class
            trainall = trainall';
            sz = size(img);
            gt=zeros(sz(1),sz(2));  % training image
            gt(trainall(1,:))=trainall(2,:);
            c = max(unique(gt(:)));
            [X,y] = getPatterns(img, gt);
            if(strcmp(dsName,'indiana12'))
                % delete four minor classes
                classCount = accumarray(y',1);
                [~,classInd] = sort(classCount);
                ind = ismember(y,classInd(5:end));
                y = y(ind); X = X(:,ind);
                classInd = sort(classInd(5:end));
                for i=1:12
                    y(y == classInd(i)) = i;
                end
                c = 12;
            end
    end
    
    if(normalize > 0)
        m = mean([X,XTe],2);
        X = bsxfun(@minus, X, m);
        s = std([X,XTe],0,2);
        X = bsxfun(@rdivide,X,s);

        if(~isempty(XTe))
            XTe = bsxfun(@minus, XTe, m);
            XTe = bsxfun(@rdivide,XTe,s);
        end
    end
    
    if(l2norm1 || lnorm1)
        X = convert_lnorm1(X);
        if(~isempty(XTe))
            XTe = convert_lnorm1(XTe);
        end
    end
    
    if(biasFlag == 1)
        X = [X; ones(1,size(X,2))]; % add constant feature to X for bias
        if(size(XTe,2) > 1)
            XTe = [XTe; ones(1,size(XTe,2))]; 
        end
    else
        if(biasFlag == 2)
            X = [ones(1,size(X,2));X];
            if(size(XTe,2) > 1)
                XTe = [ones(1,size(XTe,2));XTe]; 
            end
        end
    end
    
    if(isfield(params,'reducedDim') && params.reducedDim > 0)
        [~,X_pca,~] = pca(X', 'NumComponents',params.reducedDim);
        X = X_pca';
        if(~isempty(XTe))
            [~,XTe,~] = pca(XTe', 'NumComponents',params.reducedDim);
            XTe = XTe';
        end
    end
    
    if(isfield(params,'c_limit') && params.c_limit > 0)
%         c_list = unique(y);
%         c = length(c_list);
%         chosen_idx = randperm(c, params.c_limit);
%         chosen_classes = c_list(chosen_idx);
        chosen_classes = 1:params.c_limit;
        sel_ind = ismember(y, chosen_classes);
        X = X(:, sel_ind);
        y = y(:, sel_ind);
        c = params.c_limit;
    end
    
    
    
    if(rowFlag == 1)
        X = X';
        y = y';
        XTe = XTe';
        yTe = yTe';
    end
    
end



function X = convert_lnorm1(X)
    X = X';
    for i=1:700:size(X,1)
        BB=min(700,size(X,1)-i);
        X(i:i+BB,:)=diag((sum(X(i:i+BB,:).^2,2)+1e-20).^-0.5)*X(i:i+BB,:);
    end
    X = X';
end