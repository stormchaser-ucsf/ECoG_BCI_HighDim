%% BUILDING MLP FOR THE HAND MODEL

clc;clear
close all

root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath('C:\Users\nikic\Documents\MATLAB')
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))

% GETTING DATA FROM THE HAND TASK, all but the last day's data
foldernames = {'20230427'};
cd(root_path)

hand_files=[];
for i=length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'HandImagined')
    %     if ~exist(folderpath)
    %         folderpath = fullfile(root_path, foldernames{i},'HandOnline')
    %     end
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'Imagined');
        tmp=dir(filepath);
        tmp=findfiles('',filepath)';
        hand_files = [hand_files;tmp(1:12)];
    end
end


% split into training and testing 
idx = randperm(length(hand_files),round(0.8*length(hand_files)));
train_files = hand_files(idx);
I = ones(length(hand_files),1);
I(idx)=0;
test_files = hand_files(find(I==1));


% load the data for the imagined files, if they belong to right thumb,
% index, middle, ring, pinky, pinch, tripod, power
D1=[];%thumb
D2=[];%index
D3=[];%middle
D4=[];%ring
D5=[];%pinky
D6=[];%power
D7=[];%pinch
D8=[];%tripod
D9=[];%wrist out
D10=[];%wrist in
D11=[];% wrist flex
D12=[];% wrist extend

for i=1:length(train_files)
    disp(i/length(train_files)*100)
    load(train_files{i})

    features  = TrialData.SmoothedNeuralFeatures;
    kinax = TrialData.TaskState;


    if length(TrialData.Time) ~= length(TrialData.NeuralTime)
        time_exp = TrialData.Time;
        time_neural = TrialData.NeuralTime;
        time_exp = time_exp-time_exp(1);
        time_neural = time_neural-time_neural(1);
        time_exp = time_exp(find(kinax==3));
        kinax_neural=[];diff_times=[];
        for k=1:length(time_exp) % find the index of closed neural time
            [aa bb]= min(abs(time_neural - time_exp(k)));
            kinax_neural= [kinax_neural bb];
        end
        kinax_neural=unique(kinax_neural);
        temp = cell2mat(features(kinax_neural));
    else
        temp = cell2mat(features(kinax));
    end

    % get delta, beta and hG removing bad channels
    temp = temp([257:512 1025:1280 1537:1792],:);
    bad_ch = [108 113 118];
    good_ch = ones(size(temp,1),1);
    for ii=1:length(bad_ch)
        bad_ch_tmp = bad_ch(ii)*[1 2 3];
        good_ch(bad_ch_tmp)=0;
    end
    temp = temp(logical(good_ch),:);


    if TrialData.TargetID == 1
        D1 = [D1 temp];
    elseif TrialData.TargetID == 2
        D2 = [D2 temp];
    elseif TrialData.TargetID == 3
        D3 = [D3 temp];
    elseif TrialData.TargetID == 4
        D4 = [D4 temp];
    elseif TrialData.TargetID == 5
        D5 = [D5 temp];
    elseif TrialData.TargetID == 6
        D6 = [D6 temp];
    elseif TrialData.TargetID == 7
        D7 = [D7 temp];
    elseif TrialData.TargetID == 8
        D8 = [D8 temp];
    elseif TrialData.TargetID == 9
        D9 = [D9 temp];
    elseif TrialData.TargetID == 10
        D10 = [D10 temp];
    elseif TrialData.TargetID == 11
        D11 = [D11 temp];
    elseif TrialData.TargetID == 12
        D12 = [D12 temp];
    end
end


clear condn_data
idx=1;
condn_data{1}=[D1(idx:end,:) ]'; 
condn_data{2}= [D2(idx:end,:)]'; 
condn_data{3}=[D3(idx:end,:)]'; 
condn_data{4}=[D4(idx:end,:)]'; 
condn_data{5}=[D5(idx:end,:)]'; 
condn_data{6}=[D6(idx:end,:)]'; 
condn_data{7}=[D7(idx:end,:)]'; 
condn_data{8}=[D8(idx:end,:)]'; 
condn_data{9}=[D9(idx:end,:)]'; 
condn_data{10}=[D10(idx:end,:)]'; 
condn_data{11}=[D11(idx:end,:)]'; 
condn_data{12}=[D12(idx:end,:)]'; 


% 2norm
for i=1:length(condn_data)
    tmp = condn_data{i};
    for j=1:size(tmp,1)
        tmp(j,:) = tmp(j,:)./norm(tmp(j,:));
    end
    condn_data{i}=tmp;
end


N=[];
T1=[];
for i=1:length(condn_data)
    tmp=condn_data{i};
    N = [N tmp'];
    T1 = [T1;i*ones(size(tmp,1),1)];
end


T = zeros(size(T1,1),length(condn_data));
for i=1:length(condn_data)
    [aa bb]=find(T1==i);
    T(aa(1):aa(end),i)=1;
end


% code to train a neural network
clear net_hand_mlp
net_hand_mlp = patternnet([64 64 64]) ;
net_hand_mlp.performParam.regularization=0.2;
net_hand_mlp = train(net_hand_mlp,N,T','UseGPU','yes');
%genFunction(net_hand_mlp,'MLP_Hand_03032022')
%save net_hand_mlp_0303 net_hand_mlp_0303

% using custom layers
layers = [ ...
    featureInputLayer(759)
    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.4)
    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.4)
    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.4)
    fullyConnectedLayer(12)
    softmaxLayer
    classificationLayer
    ];



X = N;
Y=categorical(T1);
idx = randperm(length(Y),round(0.8*length(Y)));
Xtrain = X(:,idx);
Ytrain = Y(idx);
I = ones(length(Y),1);
I(idx)=0;
idx1 = find(I~=0);
Xtest = X(:,idx1);
Ytest = Y(idx1);



%'ValidationData',{XTest,YTest},...
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',50, ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',32,...
    'ValidationFrequency',91,...
    'ValidationPatience',5,...
    'LearnRateSchedule','piecewise',...
    'ExecutionEnvironment','GPU',...
    'ValidationData',{Xtest',Ytest});

% build the classifier
net_mlp_hand = trainNetwork(Xtrain',Ytrain,layers,options);


% NOW TEST IT ON HELD OUT TRIALS
res_acc=zeros(length(condn_data));
for i=1:length(test_files)
    disp(i/length(test_files)*100)
    load(test_files{i})

    features  = TrialData.SmoothedNeuralFeatures;
    kinax = TrialData.TaskState;


    if length(TrialData.Time) ~= length(TrialData.NeuralTime)
        time_exp = TrialData.Time;
        time_neural = TrialData.NeuralTime;
        time_exp = time_exp-time_exp(1);
        time_neural = time_neural-time_neural(1);
        time_exp = time_exp(find(kinax==3));
        kinax_neural=[];diff_times=[];
        for k=1:length(time_exp) % find the index of closed neural time
            [aa bb]= min(abs(time_neural - time_exp(k)));
            kinax_neural= [kinax_neural bb];
        end
        kinax_neural=unique(kinax_neural);
        temp = cell2mat(features(kinax_neural));
    else
        temp = cell2mat(features(kinax));
    end

    % get delta, beta and hG removing bad channels
    temp = temp([257:512 1025:1280 1537:1792],:);
    bad_ch = [108 113 118];
    good_ch = ones(size(temp,1),1);
    for ii=1:length(bad_ch)
        bad_ch_tmp = bad_ch(ii)*[1 2 3];
        good_ch(bad_ch_tmp)=0;
    end
    temp = temp(logical(good_ch),:);

    decodes = net_hand_mlp(temp);
    %decodes = predict(net_mlp_hand,temp');
    [aa bb]=max(sum(decodes'));
    %decodes(decodes<=0.4)=0;
    %[aa bb]=max(decodes');
    %figure;stem(sum(decodes,2))
    decision = (bb);
    res_acc(TrialData.TargetID,decision)=res_acc(TrialData.TargetID,decision)+1;
end

for i=1:length(res_acc)
    res_acc(i,:)=res_acc(i,:)./sum(res_acc(i,:));
end
figure;imagesc(res_acc)


%% LDA on hand data

clc;clear

root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath('C:\Users\nikic\Documents\MATLAB')
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))

% GETTING DATA FROM THE HAND TASK, all but the last day's data
foldernames = {'20230427'};
cd(root_path)

imagined_files=[];
hand_files=[];
for i=length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'HandImagined')
    %     if ~exist(folderpath)
    %         folderpath = fullfile(root_path, foldernames{i},'HandOnline')
    %     end
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'Imagined');
        tmp=dir(filepath);
        tmp=findfiles('',filepath)';
        hand_files = [hand_files;tmp(1:12)];
    end
end
imagined_files=hand_files;

res_overall=[];
for iter=1:5
    disp(iter)


    % load the data for the imagined files, if they belong to right thumb,
    % index, middle, ring, pinky, pinch, tripod, power
    D1i={};
    D2i={};
    D3i={};
    D4i={};
    D5i={};
    D6i={};
    D7i={};
    D8i={};
    D9i={};
    D10i={};
    idx = randperm(length(imagined_files),round(0.8*length(imagined_files)));
    train_files = imagined_files(idx);
    I = ones(length(imagined_files),1);
    I(idx)=0;
    test_files = imagined_files(find(I==1));

    for i=1:length(train_files)
        %disp(i/length(train_files)*100)
        try
            load(train_files{i})
            file_loaded = true;
        catch
            file_loaded=false;
            disp(['Could not load ' files{j}]);
        end


        if file_loaded
            action = TrialData.TargetID;
            features  = TrialData.SmoothedNeuralFeatures;
            kinax = TrialData.TaskState;


            if length(TrialData.Time) ~= length(TrialData.NeuralTime)
                time_exp = TrialData.Time;
                time_neural = TrialData.NeuralTime;
                time_exp = time_exp-time_exp(1);
                time_neural = time_neural-time_neural(1);
                time_exp = time_exp(find(kinax==3));
                kinax_neural=[];diff_times=[];
                for k=1:length(time_exp) % find the index of closed neural time
                    [aa bb]= min(abs(time_neural - time_exp(k)));
                    kinax_neural= [kinax_neural bb];
                end
                kinax_neural=unique(kinax_neural);
                temp = cell2mat(features(kinax_neural));
            else
                temp = cell2mat(features(kinax));
            end

            % get delta, beta and hG removing bad channels
            temp = temp([257:512 1025:1280 1537:1792],:);
            bad_ch = [108 113 118];
            good_ch = ones(size(temp,1),1);
            for ii=1:length(bad_ch)
                bad_ch_tmp = bad_ch(ii)*[1 2 3];
                good_ch(bad_ch_tmp)=0;
            end
            temp = temp(logical(good_ch),:);
           
            data_seg = temp(1:end,:); % only high gamma
            %data_seg = mean(data_seg,2);

            if action ==1
                D1i = cat(2,D1i,data_seg);
                %D1f = cat(2,D1f,feat_stats1);
            elseif action ==2
                D2i = cat(2,D2i,data_seg);
                %D2f = cat(2,D2f,feat_stats1);
            elseif action ==3
                D3i = cat(2,D3i,data_seg);
                %D3f = cat(2,D3f,feat_stats1);
            elseif action ==4
                D4i = cat(2,D4i,data_seg);
                %D4f = cat(2,D4f,feat_stats1);
            elseif action ==5
                D5i = cat(2,D5i,data_seg);
                %D5f = cat(2,D5f,feat_stats1);
            elseif action ==6
                D6i = cat(2,D6i,data_seg);
                %D6f = cat(2,D6f,feat_stats1);
            elseif action ==7
                D7i = cat(2,D7i,data_seg);
                %D7f = cat(2,D7f,feat_stats1);
            elseif action ==8
                D8i = cat(2,D8i,data_seg);
                %D7f = cat(2,D7f,feat_stats1);
            elseif action ==9
                D9i = cat(2,D9i,data_seg);
            elseif action ==10
                D10i = cat(2,D10i,data_seg);
            end
        end
    end

    data=[];
    Y=[];
    data=[data cell2mat(D1i)]; Y=[Y;0*ones(size(cell2mat(D1i),2),1)];
    data=[data cell2mat(D2i)];  Y=[Y;1*ones(size(cell2mat(D2i),2),1)];
    data=[data cell2mat(D3i)];  Y=[Y;2*ones(size(cell2mat(D3i),2),1)];
    data=[data cell2mat(D4i)];  Y=[Y;3*ones(size(cell2mat(D4i),2),1)];
    data=[data cell2mat(D5i)];  Y=[Y;4*ones(size(cell2mat(D5i),2),1)];
    data=[data cell2mat(D6i)];  Y=[Y;5*ones(size(cell2mat(D6i),2),1)];
    data=[data cell2mat(D7i)];  Y=[Y;6*ones(size(cell2mat(D7i),2),1)];
    data=[data cell2mat(D8i)];  Y=[Y;7*ones(size(cell2mat(D8i),2),1)];
    data=[data cell2mat(D9i)];  Y=[Y;8*ones(size(cell2mat(D9i),2),1)];
    data=[data cell2mat(D10i)];  Y=[Y;9*ones(size(cell2mat(D10i),2),1)];
    data=data';

    % run LDA
    W = LDA(data,Y);

    % run it on the held out files and get classification accuracies
    acc=zeros(size(W,1));
    for i=1:length(test_files)
        %disp(i/length(test_files)*100)
        try
            load(test_files{i})
            file_loaded = true;
        catch
            file_loaded=false;
            disp(['Could not load ' files{j}]);
        end


        if file_loaded
            action = TrialData.TargetID;
            features  = TrialData.SmoothedNeuralFeatures;
            kinax = TrialData.TaskState;

            if length(TrialData.Time) ~= length(TrialData.NeuralTime)
                time_exp = TrialData.Time;
                time_neural = TrialData.NeuralTime;
                time_exp = time_exp-time_exp(1);
                time_neural = time_neural-time_neural(1);
                time_exp = time_exp(find(kinax==3));
                kinax_neural=[];diff_times=[];
                for k=1:length(time_exp) % find the index of closed neural time
                    [aa bb]= min(abs(time_neural - time_exp(k)));
                    kinax_neural= [kinax_neural bb];
                end
                kinax_neural=unique(kinax_neural);
                temp = cell2mat(features(kinax_neural));
            else
                temp = cell2mat(features(kinax));
            end

            % get delta, beta and hG removing bad channels
            temp = temp([257:512 1025:1280 1537:1792],:);
            bad_ch = [108 113 118];
            good_ch = ones(size(temp,1),1);
            for ii=1:length(bad_ch)
                bad_ch_tmp = bad_ch(ii)*[1 2 3];
                good_ch(bad_ch_tmp)=0;
            end
            temp = temp(logical(good_ch),:);

            data_seg = temp(1:end,:); % only high gamma
            %data_seg = mean(data_seg,2);
        end
        data_seg = data_seg';

        % run it thru the LDA
        L = [ones(size(data_seg,1),1) data_seg] * W';

        % get classification prob
        P = exp(L) ./ repmat(sum(exp(L),2),[1 size(L,2)]);

        %average prob
        decision = nanmean(P(1:end,:));
        %decision = P;
        [aa bb]=max(decision);

        % correction for online trials
%         if TrialData.TargetID==9
%             TrialData.TargetID = 7;
%         elseif TrialData.TargetID==10
%             TrialData.TargetID = 8;
%         end


        % store results
        if TrialData.TargetID <=10
            acc(TrialData.TargetID,bb) = acc(TrialData.TargetID,bb)+1;
        end
    end

    for i=1:length(acc)
        acc(i,:)= acc(i,:)/sum(acc(i,:));
    end
    %figure;imagesc(acc)
    %diag(acc)
    %mean(ans)

    res_overall(iter,:,:)=acc;

end

acc=squeeze(nanmean(res_overall,1));
figure;imagesc(acc)
diag(acc)
mean(ans)
colormap bone
caxis([0 1])
set(gcf,'Color','w')
title(['Av. Classif. Acc of ' num2str(mean(diag(acc))) '%'])
xticks(1:5)
yticks(1:5)
xticklabels({'Thumb','Index','Middle','Ring','Little'})
yticklabels({'Thumb','Index','Middle','Ring','Little'})
set(gca,'FontSize',14)



%% REGRESSION OF NEURAL FEATURES HG AND LMP ONTO SMOOTHED KINEMATICS

clc;clear
root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames={'20220624'};

imagined_files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'HandImagined');
    D=dir(folderpath);

    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'Imagined');
        tmp=dir(filepath);
        imagined_files = [imagined_files;findfiles('',filepath)'];
    end
end


lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',3,'PassbandRipple',0.2, ...
    'SampleRate',1e3);

lfoFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',5,'PassbandRipple',0.2, ...
    'SampleRate',1e3);


% collect the data
neural_data={};
kin_data={};
targetID=[];
for i=1:length(imagined_files)

    disp(i/length(imagined_files)*100)
    try
        load(imagined_files{i})
        file_loaded = true;
    catch
        file_loaded=false;
        disp(['Could not load ' files{j}]);
    end

    action = TrialData.TargetID;

    if file_loaded && action ==1

        % get times for state 3 from the sample rate of screen refresh
        time  = TrialData.Time;
        time = time - time(1);
        idx = find(TrialData.TaskState==3) ;
        task_time = time(idx);

        % get the kinematics and extract times in state 3 when trials
        % started and ended
        kin = TrialData.CursorState;

        %get the overall broadband data
        raw_data=cell2mat(TrialData.BroadbandData');

        % extract hG
        Params = TrialData.Params;
        filtered_data=zeros(size(raw_data,1),size(raw_data,2),8);
        k=1;
        for ii=9:16 %9:16 is hG, 4:5 is beta
            filtered_data(:,:,k) =  abs(hilbert(filtfilt(...
                Params.FilterBank(ii).b, ...
                Params.FilterBank(ii).a, ...
                raw_data)));
            k=k+1;
        end
        tmp_hg = squeeze(mean(filtered_data,3));
        tmp_hg = filtfilt(lfoFilt,tmp_hg);
        %         for j=1:size(tmp_hg,2)
        %             tmp_hg(:,j) = smooth(tmp_hg(:,j),100);
        %         end

        % get low gamma
        filtered_data=zeros(size(raw_data,1),size(raw_data,2),3);
        k=1;
        for ii=6:8 %9:16 is hG, 4:5 is beta
            filtered_data(:,:,k) =  abs(hilbert(filtfilt(...
                Params.FilterBank(ii).b, ...
                Params.FilterBank(ii).a, ...
                raw_data)));
            k=k+1;
        end
        tmp_lg = squeeze(mean(filtered_data,3));
        %tmp_lg = filtfilt(lfoFilt,tmp_lg);


        %extract LMP data
        tmp_lmp = ((filtfilt(lpFilt,raw_data)));


        % upsample the kinematic data for later extraction in ms
        xq = (0:size(raw_data,1)-1)*1e-3;
        kin_up = interp1(time',kin(1,:)',xq);
        kin_up_smoothed = smooth(kin_up,500);
        %I = isnan(kin_up);I_idx=find(I==1);
        %if sum(I)>0
        %     kin_up(I) = kin_up(I_idx(1)-1);
        %end
        %kin_up_filt = filtfilt(lpFilt,kin_up);

        % now find the start and stop signals in the kinematic data
        kin=kin(1,idx);
        kind = [0 diff(kin)];
        aa=find(kind==0);
        kin_st=[];
        kin_stp=[];
        for j=1:length(aa)-1
            if (aa(j+1)-aa(j))>1
                kin_st = [kin_st aa(j)];
                kin_stp = [kin_stp aa(j+1)-1];
            end
        end

        %getting start and stop times
        start_time = task_time(kin_st);
        stp_time = task_time(kin_stp);

        % extract the broadband data (Fs-1KhZ) and smoothed kinematics
        start_time_neural = round(start_time*1e3);
        stop_time_neural = round(stp_time*1e3);
        data_seg={};
        data_kin={};
        chmap=TrialData.Params.ChMap;
        for j=1:length(start_time_neural)
            tmp = (raw_data(start_time_neural(j):stop_time_neural(j),:));
            tmp1 = zscore(tmp_hg(start_time_neural(j):stop_time_neural(j),:));
            tmp2 = zscore(tmp_lmp(start_time_neural(j):stop_time_neural(j),:));
            tmp3 = zscore(tmp_lg(start_time_neural(j):stop_time_neural(j),:));
            %tmp1=tmp1(:,m1);
            %tmp2=tmp2(:,m1);
            %tmp3=tmp3(:,m1);
            %tmp=tmp(1:round(size(tmp,1)/2),:); % only flexion phase
            tmp = [tmp1 tmp2 ];
            data_seg = cat(2,data_seg,tmp);

            tmp = (kin_up_smoothed(start_time_neural(j):stop_time_neural(j)));
            %tmp=tmp(1:round(length(tmp)/2));  % only flexion phase
            tmp_kin=randn(length(tmp),1)*1e-6;
            tmp_kin(:,1) = tmp;
            % velocity
            %tmp_kin= zscore(diff(tmp_kin));
            %tmp_kin = [tmp_kin(1) ;tmp_kin];
            data_kin = cat(2,data_kin,tmp_kin);

            targetID=[targetID action];
        end
        neural_data=cat(2,neural_data,data_seg);
        kin_data=cat(2,kin_data,data_kin);
    end
end


% plot ERPs from channel 106
ch_data = [];
for i=1:length(neural_data)
    tmp=cell2mat(neural_data(i));
    tmp=tmp(1:2690,:);
    ch_data=cat(3,ch_data,tmp);
end
ch3=squeeze(ch_data(:,106+128,:));
figure;plot(ch3,'Color',[.5 .5 .5 .5])
hold on
plot(mean(ch3,2),'Color','k')

% split into training and testing -> single action 
overall_r2=[];
dim=30;
parfor iter=1:100
    %disp(iter)
    train_idx=  randperm(length(neural_data),round(0.7*length(neural_data)));
    test_idx = ones(length(neural_data),1);
    test_idx(train_idx)=0;
    test_idx=logical(test_idx);
    train_neural = neural_data(train_idx);
    train_kin = kin_data(train_idx);
    test_neural = neural_data(test_idx);
    test_kin = kin_data(test_idx);

    % build the regression model
    train_neural = cell2mat(train_neural');
    train_kin = cell2mat(train_kin');
    train_kin=train_kin(:,1);
    [c,s,l]=pca(train_neural(:,1:128));
    [c1,s1,l1]=pca(train_neural(:,129:end));
    %[c2,s2,l2]=pca(train_neural(:,257:384));
    train_neural = [s(:,1:dim) s1(:,1:dim) ];

    % regression
    x=train_neural;
    y=train_kin;
    for j=1:size(y,2)
        y(:,j)=smooth(y(:,j),100);
    end
    y=y-mean(y);
    % OLS or ridge or weiner etc
    bhat = pinv(x)*y;
    %bhat = (x'*x  + 0.01*eye(size(x,2)))\(x'*y);
    yhat = x*bhat;

    % test it out on held out data
    test_neural = cell2mat(test_neural');
    %[c,s,l]=pca(test_neural(:,1:128));
    %[c1,s1,l1]=pca(test_neural(:,129:256));
    s=test_neural(:,1:128)*c(:,1:dim);
    s1=test_neural(:,129:end)*c1(:,1:dim);
    %s2=test_neural(:,257:384)*c2(:,1:dim);
    test_neural = [s(:,1:dim) s1(:,1:dim) ];
    test_kin = cell2mat(test_kin');
    test_kin=test_kin(:,1);
    x=test_neural;
    y=test_kin;
    for j=1:size(y,2)
       y(:,j)=smooth(y(:,j),100);
    end
    y=y-mean(y);
    % get prediction
    yhat = x*bhat;

    % looking at r2
    %figure;plot(y(:,1));
    %hold on
    %plot(yhat(:,1))
    [r,p]=corrcoef(y,yhat);
    overall_r2(iter)=r(1,2);
end
figure;boxplot(overall_r2)



% split into training and testing, swapping weights from one action to
% another 
figure;
r2_finger={};
for i=1:5
    disp(i)
    r2_comparison=[];
    a_idx = find(targetID==i);
    neural_data_a = neural_data(a_idx);
    kin_data_a = kin_data(a_idx);    
    %[r2]=run_regression_bci(neural_data_a,kin_data_a,40);
    %r2_comparison = [r2_comparison r2'];
    for j=1:5
        b_idx = find(targetID==j);
        neural_data_b = neural_data(b_idx);
        kin_data_b = kin_data(b_idx);

        % run cross val regression
        [r2b]=run_regression_bci_swap(neural_data_b,kin_data_b,neural_data_a,kin_data_a,40);
        r2_comparison=[r2_comparison r2b'];
        r2_finger{i,j} = r2_comparison;
    end
    r2mean = sort(bootstrp(1000,@mean,r2_comparison));
    subplot(5,1,i);
    boxplot(r2mean)    
    box off
    %title(num2str(i))
end

save regression_hand_bci -v7.3

tmp=r2_finger{1,5};
r2mean = sort(bootstrp(1000,@mean,tmp));
figure;
yneg = r2mean(500,:)-r2mean(25,:);
ypos = r2mean(975,:)-r2mean(500,:);
errorbar(1:5,r2mean(500,:),yneg,ypos,'.','LineWidth',1)
xlim([0 6])
hold on
plot(1:5,r2mean(500,:),'o','MarkerSize',20,'LineWidth',1)
xticks(1:5)
xticklabels({'Thumb','Index','Middle','Ring','Index'})
ylabel('Corr with Kin')
title('Thumb submanifold')
set(gcf,'Color','w')
set(gca,'FontSize',14)
