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

