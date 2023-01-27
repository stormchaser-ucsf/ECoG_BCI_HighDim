function [TrialZ,dist_val,mean_latent,var_latent,idx,acc,condn_data_recon] = get_latent_regression(files,net,imag)
%function [TrialZ,dist_val] = get_latent(files,net,imag)

idx=[];
TrialZ=[];
decodes=[];
acc=zeros(7);
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
for i=1:length(files)
    disp(i)
    file_loaded=1;
    try
        load(files{i});
    catch
        file_loaded=0;
    end
    if file_loaded
        features  = TrialData.SmoothedNeuralFeatures;
        kinax = TrialData.TaskState;
        kinax = [find(kinax==3)];
        if imag==0
            counter=TrialData.Params.ClickCounter;
            kinax=kinax(end-counter+1:end);
        end
        temp = cell2mat(features(kinax));
        chmap = TrialData.Params.ChMap;
        X = bci_pooling(temp,chmap);

        %2-norm the data
        for j=1:size(X,2)
            X(:,j)=X(:,j)./norm(X(:,j));
        end

        % feed it through the AE
        %X = X(1:96,:);
        X = X(3:3:96,:); % only hg
        %2-norm the data
        for j=1:size(X,2)
            X(:,j)=X(:,j)./norm(X(:,j));
        end
        Z = activations(net,X','autoencoder');
        out = predict(net,X');

        %         TrialZ = [TrialZ Z];
        %         idx=[idx repmat(TrialData.TargetID,1,size(Z,2))];

        if imag==0
            %if TrialData.SelectedTargetID == TrialData.TargetID
            %Z = Z(:,end-4:end);
            TrialZ = [TrialZ Z];
            idx=[idx repmat(TrialData.TargetID,1,size(Z,2))];
            %Z = mean(Z,2);
            %TrialZ = [TrialZ Z];
            %idx=[idx TrialData.TargetID];
            %end
        else
            TrialZ = [TrialZ Z];
            idx=[idx repmat(TrialData.TargetID,1,size(Z,2))];
        end

        if TrialData.TargetID == 1
            D1=[D1;out];
        elseif TrialData.TargetID == 2
            D2=[D2;out];
        elseif TrialData.TargetID == 3
            D3=[D3;out];
        elseif TrialData.TargetID == 4
            D4=[D4;out];
        elseif TrialData.TargetID == 5
            D5=[D5;out];
        elseif TrialData.TargetID == 6
            D6=[D6;out];
        elseif TrialData.TargetID == 7
            D7=[D7;out];
        end



    end

end


condn_data_recon{1} = D1;
condn_data_recon{2} = D2;
condn_data_recon{3} = D3;
condn_data_recon{4} = D4;
condn_data_recon{5} = D5;
condn_data_recon{6} = D6;
condn_data_recon{7} = D7;

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
        kk = randperm(size(A,2),round(size(A,2)/3));
        A = A(:,kk);
        mean_latent=[mean_latent;mean(A,2)'];
        var_latent = [var_latent;det(cov(A'))];
        for j=i+1:len
            idxx = find(idx==j);
            B=Z(:,idxx);
            kk = randperm(size(B,2),round(size(B,2)/3));
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