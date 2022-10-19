function [TrialZ,dist_val,mean_latent,var_latent,idx,acc] = get_latent_regression_procrustes(condn_data,net,imag)
%function [TrialZ,dist_val] = get_latent(files,net,imag)

idx=[];
TrialZ=[];
for i=1:length(condn_data)
    tmp=condn_data{i};
    Z = activations(net,tmp,'autoencoder');
    TrialZ = [TrialZ Z];
    idx=[idx repmat(i,1,size(Z,2))];
end

% plot the trial averaged activity in the latent space
Z=TrialZ;
%[c,s,l]=pca(Z');
%Z=s';
cmap = parula(length(unique(idx)));
figure;hold on
for i=1:size(cmap,1)
    %if i==1||i==6||i==7||i==4||i==2
    idxx = find(idx==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:),'MarkerSize',20);
    %end
end
xlabel('Latent 1')
ylabel('Latent 2')
zlabel('Latent 3')

if imag==1
    title('Imagined Latent Space')
else
    title('Proj. Online Data through Latent Space')
end
set(gcf,'Color','w')
set(gca,'LineWidth',1)
set(gca,'FontSize',12)



if imag==1
    % subsample in the case of imagined movement data to get mahab distance
    len = length(unique(idx));
    D = zeros(len);
    var_latent=[];
    mean_latent=[];
    %mean_latent=zeros(len);
    for i=1:len
        idxx = find(idx==i);
        A=Z(:,idxx);
        kk = randperm(size(A,2),round(size(A,2)/4));
        A = A(:,kk);
        mean_latent=[mean_latent;mean(A,2)'];
        var_latent = [var_latent;det(cov(A'))];
        for j=i+1:len
            idxx = find(idx==j);
            B=Z(:,idxx);
            kk = randperm(size(B,2),round(size(B,2)/4));
            B = B(:,kk);
            D(i,j) = mahal2(A',B',2);
            D(j,i) = D(i,j);

            %mean_latent(i,j)=mahal2(A',B',3);
        end
    end
    dist_val = squareform(D);
else
    % get pairwise mahalanbois distance
    len = length(unique(idx));
    D = zeros(len);
    var_latent=[];
    mean_latent=[];
    %mean_latent=zeros(len);
    for i=1:len
        idxx = find(idx==i);
        A=Z(:,idxx);
        mean_latent=[mean_latent;mean(A,2)'];
        var_latent = [var_latent;det(cov(A'))];
        for j=i+1:len
            idxx = find(idx==j);
            B=Z(:,idxx);
            D(i,j) = mahal2(A',B',2);
            D(j,i) = D(i,j);

            %mean_latent(i,j)=mahal2(A',B',3);
        end
    end
    dist_val = squareform(D);
end


% % compute decoding accuracy
% for i=1:length(acc)
%     acc(i,:) = acc(i,:)/sum(acc(i,:));
% end




end