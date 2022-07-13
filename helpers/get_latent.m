function [TrialZ,dist_val] = get_latent(files,net,imag)
%function [TrialZ,dist_val] = get_latent(files,net,imag)

idx=[];
TrialZ=[];
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
        X = X(1:96,:);
        Z = activations(net,X','autoencoder');
        if imag==0
            if TrialData.SelectedTargetID == TrialData.TargetID
                %Z = Z(:,end-4:end);
                TrialZ = [TrialZ Z];
                idx=[idx repmat(TrialData.TargetID,1,size(Z,2))];
                %Z = mean(Z,2);
                %TrialZ = [TrialZ Z];
                %idx=[idx TrialData.TargetID];
            end
        else
            TrialZ = [TrialZ Z];
            idx=[idx repmat(TrialData.TargetID,1,size(Z,2))];
        end
    end
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

% get pairwise mahalanbois distance
len = length(unique(idx));
D = zeros(len);
for i=1:len
    idxx = find(idx==i);
    A=Z(:,idxx);
    for j=i+1:len
        idxx = find(idx==j);
        B=Z(:,idxx);
        D(i,j) = mahal2(A',B',2);
        D(j,i) = D(i,j);
    end
end
dist_val = squareform(D);


end