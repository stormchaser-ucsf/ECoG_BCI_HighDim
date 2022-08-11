function prin_angles = get_prin_angles(files,files1)

% concatentate state 3 data imagined data
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
for i=1:length(files)


    file_loaded=1;
    try
        load(files{i});
    catch
        file_loaded=0;
    end


    if file_loaded
        idx = find(TrialData.TaskState==3);
        features = TrialData.SmoothedNeuralFeatures;
        features = cell2mat(features(idx));
        % hg
        features = features(769:end,:);
        if TrialData.TargetID==1
            D1=[D1 features];

        elseif TrialData.TargetID==2
            D2=[D2 features];

        elseif TrialData.TargetID==3
            D3=[D3 features];

        elseif TrialData.TargetID==4
            D4=[D4 features];

        elseif TrialData.TargetID==5
            D5=[D5 features];

        elseif TrialData.TargetID==6
            D6=[D6 features];

        elseif TrialData.TargetID==7
            D7=[D7 features];
        end
    end

end

% concatentate state 3 data online data
D1o=[];
D2o=[];
D3o=[];
D4o=[];
D5o=[];
D6o=[];
D7o=[];
for i=1:length(files1)

    file_loaded=1;
    try
        load(files1{i});
    catch
        file_loaded=0;
    end


    if file_loaded
        idx = find(TrialData.TaskState==3);
        features = TrialData.SmoothedNeuralFeatures;
        features = cell2mat(features(idx));
        % hg
        features = features(769:end,:);
        if TrialData.TargetID==1
            D1o=[D1o features];

        elseif TrialData.TargetID==2
            D2o=[D2o features];

        elseif TrialData.TargetID==3
            D3=[D3 features];

        elseif TrialData.TargetID==4
            D4o=[D4o features];

        elseif TrialData.TargetID==5
            D5o=[D5o features];

        elseif TrialData.TargetID==6
            D6o=[D6o features];

        elseif TrialData.TargetID==7
            D7o=[D7o features];
        end
    end
end


% get a sense of the number of PCs
[c,s,l]=pca(D1o');
figure;stem(cumsum(l)./sum(l))

% are the manifolds more separate
len = min(size(D1,2),size(D1o,2));
clear dataTensor
dataTensor(:,:,1)=D1(:,1:len)';
dataTensor(:,:,2)=D1o(:,1:len)';
[prin_angles] = compute_prin_angles_manifold_bci(dataTensor,20);

% compare imagined angles to themselves
clear dataTensor
dataTensor(:,:,1)=D1(:,1:200)';
dataTensor(:,:,2)=D7(:,1:200)';
%dataTensor(:,:,3)=D3(:,1:300)';
%dataTensor(:,:,4)=D4(:,1:300)';
%dataTensor(:,:,5)=D5(:,1:300)';
%dataTensor(:,:,6)=D6(:,1:300)';
%dataTensor(:,:,7)=D7(:,1:300)';
[prin_angles] = compute_prin_angles_manifold_bci(dataTensor,20);
figure;
plot(prin_angles,'Color',[.5 .5 .5 1])

% compare online to themselves
clear dataTensor
dataTensor(:,:,1)=D1o(:,1:55)';
dataTensor(:,:,2)=D7o(:,1:55)';
%dataTensor(:,:,3)=D3(:,1:300)';
%dataTensor(:,:,4)=D4(:,1:300)';
%dataTensor(:,:,5)=D5(:,1:300)';
%dataTensor(:,:,6)=D6(:,1:300)';
%dataTensor(:,:,7)=D7(:,1:300)';
[prin_angles] = compute_prin_angles_manifold_bci(dataTensor,20);
hold on
plot(prin_angles,'Color',[.5 .5 .8 1])


% shared variance





