%% COACTIVATION / MULTISTATE building a classifier (MAIN - B1)
% test on held out trials 
% the features are the output of the last layer of the LSTM. Collate these
% features at each time-step across the trial, and label trial either right
% hand, leg, head or coactivation RH+leg, RH+head. Train a MLP on training
% trials and test on held out trials. 

clc;clear
close all
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath 'C:\Users\nikic\Documents\MATLAB'

folders={'20221129','20221206','20221214','20221215','20230111','20230118',...
    '20230120'};

% load the files especially if robot3dArrow. If Imagined and if
% TrialData.Target is between 10 and 13, then store it 
filedata=[];
k=1;
for i=1:length(folders)
    disp(i)
    folderpath = fullfile(root_path,folders{i},'Robot3DArrow');
    D = dir(folderpath);
    for j=3:length(D)
        filepath = fullfile(folderpath,D(j).name);
        D1 =dir(filepath);        
        datapath = fullfile(filepath,D1(3).name);
        files = findfiles('',datapath)';
        for ii=1:length(files)
            load(files{ii});
            target = TrialData.TargetID;
            if sum(target == [1 2 4])>0
                filedata(k).TargetID = target;
                filedata(k).filename = files{ii};                
                filedata(k).filetype = 1;
                k=k+1;
            elseif (target >= 10) && (target <=11)                
                filedata(k).TargetID = target;
                filedata(k).filename = files{ii};                
                filedata(k).filetype = 0;
                k=k+1;
            end
        end
    end
end

% load the rnn
load net_bilstm_20220824
net_bilstm = net_bilstm_20220824;
% filter bank hg
Params=[];
Params.Fs = 1000;
Params.FilterBank(1).fpass = [70,77];   % high gamma1
Params.FilterBank(end+1).fpass = [77,85];   % high gamma2
Params.FilterBank(end+1).fpass = [85,93];   % high gamma3
Params.FilterBank(end+1).fpass = [93,102];  % high gamma4
Params.FilterBank(end+1).fpass = [102,113]; % high gamma5
Params.FilterBank(end+1).fpass = [113,124]; % high gamma6
Params.FilterBank(end+1).fpass = [124,136]; % high gamma7
Params.FilterBank(end+1).fpass = [136,150]; % high gamma8
Params.FilterBank(end+1).fpass = [30,36]; % lg1
Params.FilterBank(end+1).fpass = [36,42]; % lg2
Params.FilterBank(end+1).fpass = [42,50]; % lg3
% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end
% low pass filters
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',25,'PassbandRipple',0.2, ...
    'SampleRate',1e3);

% get the neural features per trial
for i=1:length(filedata)
    disp(i/length(filedata)*100)
    filepath = filedata(i).filename;
    [lstm_output]...
        = get_lstm_performance_multistate_features(filepath,net_bilstm,Params,lpFilt);
    filedata(i).lstm_output = lstm_output;
end

% now build a classifier to discriminate between the neural features
% a 7X10X10X5 MLP will do the trick
acc_overall=[];
for iter=1:10

    tid = unique([filedata(1:end).TargetID]);
    train_idx=[];
    test_idx=[];
    condn_data={};
    for i=1:length(tid)
        idx = find([filedata(1:end).TargetID]==tid(i));
        train_idx1 = randperm(length(idx),round(0.9*length(idx)));
        I = ones(length(idx),1);
        I(train_idx1)=0;
        test_idx1 =  find(I==1);
        train_idx = [train_idx idx(train_idx1)];
        test_idx = [test_idx idx(test_idx1')];
        % get the training data
        train_idx1 = idx(train_idx1);
        tmp_data=[];
        for j=1:length(train_idx1)
            %a = filedata(train_idx1(j)).TargetID;
            disp(filedata(train_idx1(j)).TargetID)
            a = filedata(train_idx1(j)).lstm_output;
            if size(a,2)>20
                a = a(:,5:15);
            end
            tmp_data = [tmp_data a];
        end
        condn_data{i}=tmp_data';
    end

    % now build a classifier
    A = condn_data{1};
    B = condn_data{2};
    C = condn_data{3};
    D = condn_data{4};
    E = condn_data{5};
    D = D(randperm(size(D,1),round(0.6*size(D,1))),:);
    E = E(randperm(size(E,1),round(0.6*size(E,1))),:);

    clear N
    N = [A' B' C' D' E' ];
    T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
        5*ones(size(E,1),1)];

    T = zeros(size(T1,1),5);
    [aa bb]=find(T1==1);[aa(1) aa(end)]
    T(aa(1):aa(end),1)=1;
    [aa bb]=find(T1==2);[aa(1) aa(end)]
    T(aa(1):aa(end),2)=1;
    [aa bb]=find(T1==3);[aa(1) aa(end)]
    T(aa(1):aa(end),3)=1;
    [aa bb]=find(T1==4);[aa(1) aa(end)]
    T(aa(1):aa(end),4)=1;
    [aa bb]=find(T1==5);[aa(1) aa(end)]
    T(aa(1):aa(end),5)=1;

    % code to train a neural network
    clear net
    net = patternnet([10 5 ]) ;
    net.performParam.regularization=0.2;
    %net.divideParam.trainRatio=0.85;
    %net.divideParam.valRatio=0.15;
    %net.divideParam.testRatio=0;
    net = train(net,N,T');


    % now run it through the test dataset
    acc=zeros(5);
    for i=1:length(test_idx)
        tmp = filedata(test_idx(i)).lstm_output;
        target = filedata(test_idx(i)).TargetID;
        if target==1
            target=1;
        elseif target==2
            target=2;
        elseif target==4
            target=3;
        elseif target==10
            target=4;
        elseif target==11
            target=5;
        end

        out=net(tmp);
        [aa bb]=max(out);
        decode = mode(bb);
        acc(target,decode) = acc(target,decode)+1;
    end

    for i=1:size(acc,1)
        acc(i,:) = acc(i,:)./sum(acc(i,:));
    end

    figure;
    imagesc(acc)
    set(gcf,'Color','w')
    set(gca,'FontSize',14)
    xticks(1:5)
    yticks(1:5)
    xticklabels({'Right Thumb','Left leg','Head','Rt. Thumb + Head','Rt. Thumb + Lt. Leg'})
    yticklabels({'Right Thumb','Left leg','Head','Rt. Thumb + Head','Rt. Thumb + Lt. leg'})
    colormap bone
    caxis([0 1])
    colorbar
    title([num2str(100*mean(diag(acc))) '% Accuracy'])
    acc_overall(iter,:,:)=acc;

end

acc = squeeze(mean(acc_overall,1));
figure;
imagesc(acc)
set(gcf,'Color','w')
set(gca,'FontSize',14)
xticks(1:5)
yticks(1:5)
xticklabels({'Right Thumb','Left leg','Head','Rt. Thumb + Head','Rt. Thumb + Lt. Leg'})
yticklabels({'Right Thumb','Left leg','Head','Rt. Thumb + Head','Rt. Thumb + Lt. leg'})
colormap bone
caxis([0 1])
colorbar
title([num2str(100*mean(diag(acc))) '% Accuracy'])


%save multistate_decoding_results -v7.3


%% COACTIVATION / MULTISTATE building a classifier (MAIN - B3)
% test on held out trials 
% the features are the output of the last layer of the LSTM. Collate these
% features at each time-step across the trial, and label trial either right
% hand, leg, head or coactivation RH+leg, RH+head. Train a MLP on training
% trials and test on held out trials. 

clc;clear
close all
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath 'C:\Users\nikic\Documents\MATLAB'

folders={'20240110','20240117','20240119'};

% load the files especially if robot3dArrow. If Imagined and if
% TrialData.Target is between 10 and 13, then store it
filedata=[];
k=1;
for i=1:length(folders)
    disp(i)
    folderpath = fullfile(root_path,folders{i},'Robot3DArrow');
    D = dir(folderpath);
    if i==1
        D=D(8:end);
    end
    if i==2 || i==3
        D=D(3:6);
    end
    
    for j=1:length(D)
        filepath = fullfile(folderpath,D(j).name);
        D1 =dir(filepath);
        datapath = fullfile(filepath,D1(3).name);
        if ~isempty(regexp(datapath,'Imagined'))
            files = findfiles('',datapath)';
            for ii=1:length(files)
                load(files{ii});
                target = TrialData.TargetID;
                if sum(target == [1 2 4])>0
                    if target==4
                        target=3;
                    end
                    filedata(k).targetID = target;
                    filedata(k).filename = files{ii};
                    filedata(k).filetype = 1;
                    k=k+1;
                elseif (target >= 10) && (target <=11)
                    if target ==10
                        target=4;
                    elseif target==11
                        target=5;
                    end
                    filedata(k).targetID = target;
                    filedata(k).filename = files{ii};
                    filedata(k).filetype = 0;
                    k=k+1;
                end
            end
        end
    end
end

% load the neural data, also pass it through the PnP decoder
load('net_B3_mlp_BadChannels')
net = net_B3_mlp_BadChannels;
for ii=1:length(filedata)
    disp(ii/length(filedata)*100)
    load(filedata(ii).filename)
    features  = TrialData.SmoothedNeuralFeatures;
    temp = cell2mat(features);
    kinax = find(TrialData.TaskState==3);
    temp = cell2mat(features(kinax));

    kinax1 = find(TrialData.TaskState==1);
    temp_state1 = cell2mat(features(kinax1));
    temp = temp([257:512 1025:1280 1537:1792],:); %delta, beta, hG
    %temp = temp([1025:1280],:);% only hG
    %bad_ch = [108 113 118];
    bad_ch = [14,15,21,22,108,113,118]; % based on new noise levels
    good_ch = ones(size(temp,1),1);
    for iii=1:length(bad_ch)
        %bad_ch_tmp = bad_ch(iii)*[1 2 3];
        bad_ch_tmp = bad_ch(iii)+(256*[0 1 2]);
        %bad_ch_tmp = bad_ch(iii)+(256*[0 ]);
        good_ch(bad_ch_tmp)=0;
    end
    temp = temp(logical(good_ch),:);
    % 2-norm
    for i=1:size(temp,2)
        temp(:,i) = temp(:,i)./norm(temp(:,i));
    end
    
    % store neural features
    %filedata(ii).neural = temp;

    % store decoder output
    filedata(ii).neural = predict(net,temp')';
end

condn_data_overall=filedata;
cv_acc=[];
for iter=1:10
    % split into training and testing trials, 15% test, 15% val, 70% test
    tidx=0;tidx_1=0;
    while tidx<5 || tidx_1<5
        test_prop=0.15;
        test_idx = randperm(length(condn_data_overall),round(test_prop*length(condn_data_overall)));
        test_idx=test_idx(:);
        I = ones(length(condn_data_overall),1);
        I(test_idx)=0;
        train_val_idx = find(I~=0);
        if test_prop==0
            prop=0.80;
        else
            prop = (0.7/0.85);
        end
        tmp_idx = randperm(length(train_val_idx),round(prop*length(train_val_idx)));
        train_idx = train_val_idx(tmp_idx);train_idx=train_idx(:);
        I = ones(length(condn_data_overall),1);
        I([train_idx;test_idx])=0;
        val_idx = find(I~=0);val_idx=val_idx(:);

        xx=[];
        for k=1:length(val_idx)
            xx(k) = condn_data_overall(val_idx(k)).targetID;
        end
        tidx=length(unique(xx));

        xx=[];
        for k=1:length(val_idx)
            xx(k) = condn_data_overall(test_idx(k)).targetID;
        end
        tidx_1=length(unique(xx));
        %disp([tidx_1 tidx])
    end

    % training options for NN
    [options,XTrain,YTrain] = ...
        get_options(condn_data_overall,val_idx,train_idx);

    % train network
    a=condn_data_overall(1).neural;
    s=size(a,1);
    layers = get_layers2(10,10,s,5);
    %layers = get_layers1(120,s,5);
    net = trainNetwork(XTrain,YTrain,layers,options);
    cv_perf = test_network(net,condn_data_overall,test_idx);
    cv_acc(iter) = cv_perf*100;
end
cv_acc_decodes = cv_acc;
figure;boxplot(cv_acc)
ylim([10 70])
hline(20,'r')

% figure;boxplot([cv_acc_decodes' cv_acc_neural'])
% ylim([10 70])
% title('Co-activation')

%%

% load the PnP MLP

% load the rnn
load net_bilstm_20220824
net_bilstm = net_bilstm_20220824;
% filter bank hg
Params=[];
Params.Fs = 1000;
Params.FilterBank(1).fpass = [70,77];   % high gamma1
Params.FilterBank(end+1).fpass = [77,85];   % high gamma2
Params.FilterBank(end+1).fpass = [85,93];   % high gamma3
Params.FilterBank(end+1).fpass = [93,102];  % high gamma4
Params.FilterBank(end+1).fpass = [102,113]; % high gamma5
Params.FilterBank(end+1).fpass = [113,124]; % high gamma6
Params.FilterBank(end+1).fpass = [124,136]; % high gamma7
Params.FilterBank(end+1).fpass = [136,150]; % high gamma8
Params.FilterBank(end+1).fpass = [30,36]; % lg1
Params.FilterBank(end+1).fpass = [36,42]; % lg2
Params.FilterBank(end+1).fpass = [42,50]; % lg3
% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end
% low pass filters
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',25,'PassbandRipple',0.2, ...
    'SampleRate',1e3);

% get the neural features per trial
for i=1:length(filedata)
    disp(i/length(filedata)*100)
    filepath = filedata(i).filename;
    [lstm_output]...
        = get_lstm_performance_multistate_features(filepath,net_bilstm,Params,lpFilt);
    filedata(i).lstm_output = lstm_output;
end

% now build a classifier to discriminate between the neural features
% a 7X10X10X5 MLP will do the trick
acc_overall=[];
for iter=1:10

    tid = unique([filedata(1:end).TargetID]);
    train_idx=[];
    test_idx=[];
    condn_data={};
    for i=1:length(tid)
        idx = find([filedata(1:end).TargetID]==tid(i));
        train_idx1 = randperm(length(idx),round(0.9*length(idx)));
        I = ones(length(idx),1);
        I(train_idx1)=0;
        test_idx1 =  find(I==1);
        train_idx = [train_idx idx(train_idx1)];
        test_idx = [test_idx idx(test_idx1')];
        % get the training data
        train_idx1 = idx(train_idx1);
        tmp_data=[];
        for j=1:length(train_idx1)
            %a = filedata(train_idx1(j)).TargetID;
            disp(filedata(train_idx1(j)).TargetID)
            a = filedata(train_idx1(j)).lstm_output;
            if size(a,2)>20
                a = a(:,5:15);
            end
            tmp_data = [tmp_data a];
        end
        condn_data{i}=tmp_data';
    end

    % now build a classifier
    A = condn_data{1};
    B = condn_data{2};
    C = condn_data{3};
    D = condn_data{4};
    E = condn_data{5};
    D = D(randperm(size(D,1),round(0.6*size(D,1))),:);
    E = E(randperm(size(E,1),round(0.6*size(E,1))),:);

    clear N
    N = [A' B' C' D' E' ];
    T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
        5*ones(size(E,1),1)];

    T = zeros(size(T1,1),5);
    [aa bb]=find(T1==1);[aa(1) aa(end)]
    T(aa(1):aa(end),1)=1;
    [aa bb]=find(T1==2);[aa(1) aa(end)]
    T(aa(1):aa(end),2)=1;
    [aa bb]=find(T1==3);[aa(1) aa(end)]
    T(aa(1):aa(end),3)=1;
    [aa bb]=find(T1==4);[aa(1) aa(end)]
    T(aa(1):aa(end),4)=1;
    [aa bb]=find(T1==5);[aa(1) aa(end)]
    T(aa(1):aa(end),5)=1;

    % code to train a neural network
    clear net
    net = patternnet([10 5 ]) ;
    net.performParam.regularization=0.2;
    %net.divideParam.trainRatio=0.85;
    %net.divideParam.valRatio=0.15;
    %net.divideParam.testRatio=0;
    net = train(net,N,T');


    % now run it through the test dataset
    acc=zeros(5);
    for i=1:length(test_idx)
        tmp = filedata(test_idx(i)).lstm_output;
        target = filedata(test_idx(i)).TargetID;
        if target==1
            target=1;
        elseif target==2
            target=2;
        elseif target==4
            target=3;
        elseif target==10
            target=4;
        elseif target==11
            target=5;
        end

        out=net(tmp);
        [aa bb]=max(out);
        decode = mode(bb);
        acc(target,decode) = acc(target,decode)+1;
    end

    for i=1:size(acc,1)
        acc(i,:) = acc(i,:)./sum(acc(i,:));
    end

    figure;
    imagesc(acc)
    set(gcf,'Color','w')
    set(gca,'FontSize',14)
    xticks(1:5)
    yticks(1:5)
    xticklabels({'Right Thumb','Left leg','Head','Rt. Thumb + Head','Rt. Thumb + Lt. Leg'})
    yticklabels({'Right Thumb','Left leg','Head','Rt. Thumb + Head','Rt. Thumb + Lt. leg'})
    colormap bone
    caxis([0 1])
    colorbar
    title([num2str(100*mean(diag(acc))) '% Accuracy'])
    acc_overall(iter,:,:)=acc;

end

acc = squeeze(mean(acc_overall,1));
figure;
imagesc(acc)
set(gcf,'Color','w')
set(gca,'FontSize',14)
xticks(1:5)
yticks(1:5)
xticklabels({'Right Thumb','Left leg','Head','Rt. Thumb + Head','Rt. Thumb + Lt. Leg'})
yticklabels({'Right Thumb','Left leg','Head','Rt. Thumb + Head','Rt. Thumb + Lt. leg'})
colormap bone
caxis([0 1])
colorbar
title([num2str(100*mean(diag(acc))) '% Accuracy'])

