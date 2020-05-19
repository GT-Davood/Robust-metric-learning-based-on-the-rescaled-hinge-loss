function [T,S,D] = genSDT(X,y,kn,margin)

[~,N] = size(X);
dist_tn = zeros(N,1); %distance to the farthest target neighbor
[ind,dist] = knnsearch(X', X','k', 2*kn+1);
neighbour_labels = y(ind);
tar_neighbours = repmat(y',1,2*kn+1) == neighbour_labels;
imp_neighbours = ~tar_neighbours;
n_S = 0; % number of similar pairs
n_D = 0; % number of disimilar pairs
n_T = 0; % number of triplets
S = zeros(kn*N,2);
D = zeros(kn*N,2);
T = zeros(kn*kn*N,3);
for i=1:N
    try
        target_neighbours = ind(i,tar_neighbours(i,:));
        n_target = min(kn, length(target_neighbours)-1);
        ind_tn = target_neighbours(2:n_target+1);
        dist_targets = dist(i,tar_neighbours(i,:));
        dist_tn(i) = dist_targets(n_target+1); 
        S(n_S+1:n_S+n_target,:) = [repmat(i,n_target,1),ind_tn'];
        n_S = n_S + n_target;
        ind_potentialImp = dist(i,:) <= repmat(margin + dist_tn(i),1,2*kn+1);
        ind_imp = ind(i,imp_neighbours(i,:) & ind_potentialImp);
        n_imp = length(ind_imp); 
        if(n_imp > 0)
            D(n_D+1:n_D+n_imp,:) = [repmat(i,n_imp,1),ind_imp'];
            n_D = n_D + n_imp;
            n_trip = n_imp*n_target;
            col3 = repmat(ind_imp, n_target,1);
            col3 = col3(:);
            T(n_T+1:n_T+n_trip,:) = [repmat(i,n_trip,1),repmat(ind_tn',n_imp,1),col3];
            n_T = n_T + n_trip;
        end
   catch
       fprintf(2,"%d",i);
   end
end

D = D(1:n_D,:);
S = S(1:n_S,:);
T = T(1:n_T,:);

end %end function

