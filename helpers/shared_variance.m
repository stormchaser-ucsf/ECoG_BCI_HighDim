function shared_var = shared_variance(dataTensor,dim)


shared_var_overall=[];
for i=1:size(dataTensor,3)
    
    Xa = squeeze(dataTensor(:,:,i));
    % compute manifold via PCA
    [c1,s1,l1] = pca(Xa,'Centered','on');
    
    
    for j=i+1:size(dataTensor,3)
        shared_var=[];
        
        Xb = squeeze(dataTensor(:,:,j));
        % compute manifold via PCA
        [c2,s2,l2] = pca(Xb,'Centered','on');
        
        
        % getting shared variance
        % projection 1
        [Q]= c1(:,1:dim);
        [Qm]= c2(:,1:dim);
        num = trace(Q*Q' * Qm*Qm' * Q*Q');
        den = trace(Qm*Qm');
        shared_var = [shared_var num/den];
        
        % projection 2
        [Q]= c2(:,1:dim);
        [Qm]= c1(:,1:dim);
        num = trace(Q*Q' * Qm*Qm' * Q*Q');
        den = trace(Qm*Qm');
        shared_var = [shared_var num/den];
        
        shared_var_overall = [shared_var_overall; mean(shared_var)];
    end
end

shared_var=shared_var_overall;