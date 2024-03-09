%% LOOKING AT REPRESENTATIONAL DRIFT
% is there a difference between any two days recordings

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data
addpath 'C:\Users\nikic\Documents\MATLAB'
pooling=1;
condn_data_day={};
session_data = session_data([1:9 11]);
for i=1:length(session_data)
   
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');

    % only AM folders
    if i~=6
        am = strcmp(session_data(i).AM_PM,'am');
        folders_imag = folders_imag.*am;
    end

    imag_idx = find(folders_imag==1);    
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);


    %%%%%% load imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end      
    
    tmp= load_data_for_MLP_TrialLevel(files,0,pooling);
    for j=1:length(tmp)
        tmp(j).targetID=i;
    end
    condn_data_day{i}=tmp;
end

% make them all into one giant struct
tmp=cell2mat(condn_data_day(1));
condn_data_overall=tmp;
for i=2:length(condn_data_day)
    tmp=cell2mat(condn_data_day(i));
    for k=1:length(tmp)
        condn_data_overall(end+1) =tmp(k);
    end
end

% paritioning the dataset
num_classes = length(unique([condn_data_overall.targetID]));
test_idx = randperm(length(condn_data_overall),round(0.15*length(condn_data_overall)));
test_idx=test_idx(:);
I = ones(length(condn_data_overall),1);
I(test_idx)=0;
train_val_idx = find(I~=0);
prop = (0.7/0.85);
tmp_idx = randperm(length(train_val_idx),round(prop*length(train_val_idx)));
train_idx = train_val_idx(tmp_idx);train_idx=train_idx(:);
I = ones(length(condn_data_overall),1);
I([train_idx;test_idx])=0;
val_idx = find(I~=0);val_idx=val_idx(:);

% training options for NN
[options,XTrain,YTrain] = ...
    get_options(condn_data_overall,val_idx,train_idx);
 
% design the neural net
aa=condn_data_overall(1).neural;
s=size(aa,1);
layers = get_layers1(64,s,num_classes);

% train the network
net = trainNetwork(XTrain,YTrain,layers,options);

% test performance on held out trials
[cv_perf,conf_matrix] = test_network(net,condn_data_overall,test_idx,num_classes);
conf_matrix

%%
% code to grid search to best get MLP parameters
% trying here for layer width and number of units

% get the data in trial format
clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data
addpath 'C:\Users\nikic\Documents\MATLAB'
condn_data={};
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);


    %%%%%% load imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    condn_data = [condn_data;load_data_for_MLP_TrialLevel(files,0)];

    %%%%%% load online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
    condn_data = [condn_data;load_data_for_MLP_TrialLevel(files,1)];

    %%%%%% load batch data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
    condn_data = [condn_data;load_data_for_MLP_TrialLevel(files,2)];
end

% make them all into one giant struct
tmp=cell2mat(condn_data(i));
condn_data_overall=tmp;
err_idx=[];
for i=2:length(condn_data)
    tmp=cell2mat(condn_data(i));
    for k=1:length(tmp)
        condn_data_overall(end+1) =tmp(k);
        if isempty(tmp(k))
            err_idx = [err_idx; [i k]];
        end
    end
end

%cv_acc_overall={};
%cv_acc2_overall={};
%cv_acc3_overall={};
cv_acc3={};
for iter=1:1

    % split into training and testing trials, 15% test, 15% val, 70% test
    test_idx = randperm(length(condn_data_overall),round(0.15*length(condn_data_overall)));
    test_idx=test_idx(:);
    I = ones(length(condn_data_overall),1);
    I(test_idx)=0;
    train_val_idx = find(I~=0);
    prop = (0.7/0.85);
    tmp_idx = randperm(length(train_val_idx),round(prop*length(train_val_idx)));
    train_idx = train_val_idx(tmp_idx);train_idx=train_idx(:);
    I = ones(length(condn_data_overall),1);
    I([train_idx;test_idx])=0;
    val_idx = find(I~=0);val_idx=val_idx(:);

    % training options for NN
    [options,XTrain,YTrain] = ...
        get_options(condn_data_overall,val_idx,train_idx);

    % grid search
    num_units = [64,96,128,256];
    num_layers = [1,2,3];
    %cv_acc={};
    %cv_acc2={};
    i3=1;

    for i=1:length(num_layers)
        if i==1
            % loop over number of units
            for j=1:length(num_units)
                % net = patternnet([num_units(j)]) ;
                % net.performParam.regularization=0.2;
                % net = train(net,N,T','useParallel','yes');
                % cv_acc{j} = cv_perf;
                layers = get_layers1(num_units(j),384);
                net = trainNetwork(XTrain,YTrain,layers,options);
                cv_perf = test_network(net,condn_data_overall,test_idx);
                if iter==1
                    cv_acc3(i3).cv_perf = cv_perf;
                    cv_acc3(i3).layers=[num_units(j),0,0];
                    i3=i3+1;
                else
                    cv_acc3(i3).cv_perf = [cv_acc3(i3).cv_perf cv_perf];
                    %cv_acc3(i3).layers=[num_units(j),0,0];
                    i3=i3+1;
                end

            end


        elseif i==2
            % loop over number of units
            for j=1:length(num_units)
                for k=1:length(num_units)
                    %net = patternnet([num_units(j) num_units(k)]) ;
                    %net.performParam.regularization=0.2;
                    %net = train(net,N,T','useParallel','yes');
                    %cv_acc2{j,k} = cv_perf;
                    layers = get_layers2(num_units(j),num_units(k),384);
                    net = trainNetwork(XTrain,YTrain,layers,options);
                    cv_perf = test_network(net,condn_data_overall,test_idx);
                    if iter==1
                        cv_acc3(i3).cv_perf = cv_perf;
                        cv_acc3(i3).layers=[num_units(j),num_units(k),0];
                        i3=i3+1;
                    else
                        cv_acc3(i3).cv_perf = [cv_acc3(i3).cv_perf cv_perf];
                        %cv_acc3(i3).layers=[num_units(j),num_units(k),0];
                        i3=i3+1;
                    end

                end
            end

        elseif i==3
            % loop over number of units
            for j=1:length(num_units)
                for k=1:length(num_units)
                    for l=1:length(num_units)
                        %net = patternnet([num_units(j) num_units(k) num_units(l)]) ;
                        %net.performParam.regularization=0.2;
                        %net = train(net,N,T','useParallel','yes');
                        layers = get_layers(num_units(j),num_units(k),num_units(l),384);
                        net = trainNetwork(XTrain,YTrain,layers,options);
                        cv_perf = test_network(net,condn_data_overall,test_idx);
                        if iter==1
                            cv_acc3(i3).cv_perf = cv_perf;
                            cv_acc3(i3).layers=[num_units(j),num_units(k),num_units(l)];
                            i3=i3+1;
                        else
                            cv_acc3(i3).cv_perf = [cv_acc3(i3).cv_perf cv_perf];
                            i3=i3+1;
                        end

                    end
                end
            end
        end
    end
    save B1_MLP_NN_Param_Optim_NoPooling cv_acc3 -v7.3
end


%% getting decoding accuracies for zero layer
i3=85;
cv_acc3(i3).layers=[0];
for iter=1:10
    % split into training and testing trials, 15% test, 15% val, 70% test
    test_idx = randperm(length(condn_data_overall),round(0.15*length(condn_data_overall)));
    test_idx=test_idx(:);
    I = ones(length(condn_data_overall),1);
    I(test_idx)=0;
    train_val_idx = find(I~=0);
    prop = (0.7/0.85);
    tmp_idx = randperm(length(train_val_idx),round(prop*length(train_val_idx)));
    train_idx = train_val_idx(tmp_idx);train_idx=train_idx(:);
    I = ones(length(condn_data_overall),1);
    I([train_idx;test_idx])=0;
    val_idx = find(I~=0);val_idx=val_idx(:);

    % training options for NN
    [options,XTrain,YTrain] = ...
        get_options(condn_data_overall,val_idx,train_idx);

    %train NN and get CV
    layers = get_layers0(96);
    net = trainNetwork(XTrain,YTrain,layers,options);
    cv_perf = test_network(net,condn_data_overall,test_idx);
    cv_acc3(i3).cv_perf = [ cv_acc3(i3).cv_perf cv_perf];
end

save B1_MLP_NN_Param_Optim cv_acc3 -v7.3

%% PLOTTING RESULTS

% plotting
acc=[];acc1=[];
for i=1:length(cv_acc3)
    acc(i) = mean(cv_acc3(i).cv_perf);
end

[aa bb]=max(acc)

figure;boxplot(acc(1:4))
ylim([.0 .9])
figure;boxplot(acc(5:20))
ylim([.0 .9])
figure;boxplot(acc(21:end))
ylim([.0 .9])

tmp=NaN(64,3);
tmp(1:4,1) = acc(1:4)';
tmp(1:16,2) = acc(5:20)';
tmp(1:end,3) = acc(21:end)';
figure;boxplot(tmp)
ylim([.5 .7])

% MAIN plotting
% plotting across all iterations to compare all layers
acc1=[];
for i=1:4
    acc1=[acc1;cv_acc3(i).cv_perf'];
end
acc2=[];
for i=5:20
    acc2=[acc2;cv_acc3(i).cv_perf'];
end
acc3=[];
for i=21:84
    acc3=[acc3;cv_acc3(i).cv_perf'];
end
acc0=cv_acc3(end).cv_perf';
acc0(end+1:length(acc3))=NaN;
acc1(end+1:length(acc3))=NaN;
acc2(end+1:length(acc3))=NaN;
acc=[acc0 acc1 acc2 acc3];
figure;
boxplot(acc,'notch','off')
set(gcf,'Color','w')
set(gca,'FontSize',12)
ylabel('Bin Level Decoding Acc')
xticks(1:4)
xticklabels({'0 Layers','1 Layer','2 Layer','3 Layer'})
box off
title('Cross. Valid for MLP width in B1')

% sign rank tests
clear p
[p(1),h,stats] = ranksum(acc0(~isnan(acc0)),acc1(~isnan(acc1)))
[p(2),h,stats] = ranksum(acc0(~isnan(acc0)),acc2(~isnan(acc2)))
[p(3),h,stats] = ranksum(acc0(~isnan(acc0)),acc3(~isnan(acc3)))
[p(4),h,stats] = ranksum(acc1(~isnan(acc1)),acc2(~isnan(acc2)))
[p(5),h,stats] = ranksum(acc1(~isnan(acc1)),acc3(~isnan(acc3)))
[p(6),h,stats] = ranksum(acc2(~isnan(acc2)),acc3(~isnan(acc3)))
[pfdr,pval]=fdr(p,0.01);

% non parametric tests of the median
a=acc(:,2);a=a(~isnan(a));
b=acc(:,3);b=b(~isnan(b));
stat = mean(a)-mean(b);
c=[a ;b];
c=c-mean(c);
boot=[];
l=length(a);
for i=1:1000
    c1=c(randperm(numel(c)));
    a1 = c1(1:l);
    b1 = c1(l+1:end);
    boot(i) = mean(a1)-mean(b1);
end
figure;
hist(abs(boot))
vline(abs(stat))
pval = sum(abs(boot)>abs(stat))/length(boot);
title(num2str(pval))


% plotting across all iterations



acc_128=[];
for i=1:5
    layers = get_layers1(128,759);
    net = trainNetwork(XTrain,YTrain,layers,options);
    cv_perf = test_network(net,condn_data_overall,test_idx);
    acc_128(i)  = cv_perf;
end


acc_150=[];
for i=1:5
    layers = get_layers1(150,759);
    net = trainNetwork(XTrain,YTrain,layers,options);
    cv_perf = test_network(net,condn_data_overall,test_idx);
    acc_150(i)  = cv_perf;
end


% having identified the fact that 1 layer is good, now going after the
% number of units, in steps of 10 from 100 to 250
num_units = [100:10:250];
cv_singleLayer={};
for iter=1:5
    test_idx = randperm(length(condn_data_overall),round(0.15*length(condn_data_overall)));
    test_idx=test_idx(:);
    I = ones(length(condn_data_overall),1);
    I(test_idx)=0;
    train_val_idx = find(I~=0);
    prop = (0.7/0.85);
    tmp_idx = randperm(length(train_val_idx),round(prop*length(train_val_idx)));
    train_idx = train_val_idx(tmp_idx);train_idx=train_idx(:);
    I = ones(length(condn_data_overall),1);
    I([train_idx;test_idx])=0;
    val_idx = find(I~=0);val_idx=val_idx(:);

    % training options for NN
    [options,XTrain,YTrain] = ...
        get_options(condn_data_overall,val_idx,train_idx)

    for j=1:length(num_units)
        layers = get_layers1(num_units(j),759);
        net = trainNetwork(XTrain,YTrain,layers,options);
        cv_perf = test_network(net,condn_data_overall,test_idx);
        if iter==1
            cv_singleLayer(i3).cv_perf = cv_perf;
            cv_singleLayer(i3).layers=[num_units(j),0,0];
            i3=i3+1;
        else
            cv_singleLayer(i3).cv_perf = [cv_acc3(i3).cv_perf cv_perf];
            %cv_acc3(i3).layers=[num_units(j),0,0];
            i3=i3+1;
        end

    end
end

%% EFFECT OF POOLING AND MLP PARAMS ON DAILY DECODING

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data
addpath 'C:\Users\nikic\Documents\MATLAB'
cv_perf_pooling=[];
cv_perf_no_pooling=[];
perf_online=[];
pooling=1;
for i=1:length(session_data)

    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);


    %%%%%% load imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end


    condn_data={};
    condn_data = [condn_data;load_data_for_MLP_TrialLevel(files,0,pooling)];

    % convert to a structure
    tmp=cell2mat(condn_data(1));
    condn_data_overall=tmp;
    for ii=2:length(condn_data)
        tmp=cell2mat(condn_data(ii));
        for k=1:length(tmp)
            condn_data_overall(end+1) =tmp(k);
        end
    end

    % get online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
    condn_data={};
    condn_data = [condn_data;load_data_for_MLP_TrialLevel(files,1,pooling)];
    % convert to a structure
    tmp=cell2mat(condn_data(1));
    condn_data_overall_online=tmp;
    for ii=2:length(condn_data)
        tmp=cell2mat(condn_data(ii));
        for k=1:length(tmp)
            condn_data_overall_online(end+1) =tmp(k);
        end
    end

    %get decoding accuracy from the actual online trial
    decoding_acc=[];
    for ii=1:length(condn_data_overall_online)
        out = condn_data_overall_online(ii).decodes;
        tid = condn_data_overall_online(ii).targetID;
        out=out(out>0);
        idx = out==tid;
        idx1 = out~=tid;
        out(idx)=1;
        out(idx1)=0;
        decoding_acc = [decoding_acc out];
    end
    perf_online(i)=mean(decoding_acc);

    for iter=1:1

        % split into training and testing trials
        tidx=0;
        while tidx<7
            test_prop=0;
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

            tidx=[];
            for k=1:length(val_idx)
                tidx(k) = condn_data_overall(val_idx(k)).targetID;
            end
            tidx=length(unique(tidx));
        end

        % training options for NN
        [options,XTrain,YTrain] = ...
            get_options(condn_data_overall,val_idx,train_idx);

        % test on imagined data itself
        a=condn_data{1}.neural;
        s=size(a,1);
        layers = get_layers2(64,64,s);
        net = trainNetwork(XTrain,YTrain,layers,options);
        if test_prop~=0
            cv_perf = test_network(net,condn_data_overall,test_idx);
        end



        % test on held out online data
        test_idx=1:length(condn_data_overall_online);
        cv_perf = test_network(net,condn_data_overall_online,test_idx);
        if pooling==0
            cv_perf_no_pooling = [cv_perf_no_pooling cv_perf];
        elseif pooling==1
            cv_perf_pooling = [cv_perf_pooling cv_perf];
        end
    end
end

save B1_res_pooling_indiv_days_Acc  cv_perf_no_pooling cv_perf_pooling -v7.3

figure;boxplot([cv_perf_no_pooling' cv_perf_pooling'])
mean([cv_perf_no_pooling' cv_perf_pooling'])
[P,H,STATS] = ranksum(cv_perf_no_pooling,cv_perf_pooling);


%% COMPARING HISTORICAL ONLINE PERFORMANCE WITH AND WITHOUT POOLING 

clc;clear;
close all
root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210616';

%20210615 are the dates to care about with online pooling performance
%folders={'113524','115703','144621','144814'}; %no pooling
%folders={'113909','114318','114537','115420','135913','140642','140904',...
%    '145829','150031','150224','150839'}; %pooling

folders={'112750','113117','113759','114449','140842','141045','141459','143736'};

files=[];
for i=1:length(folders)
    filepath=fullfile(root_path,'Robot3DArrow',folders{i},'BCI_Fixed');
    files=[files;findfiles('',filepath)'];
end

decodes=[];
num_targets=7;
trial_acc=zeros(num_targets);
for i=1:length(files)
    disp(i/length(files)*100)
    load(files{i});
    tid=TrialData.TargetID;
    out=TrialData.ClickerState;
    decodes_vote=[];
    for i=1:(num_targets)
        decodes_vote(i)=sum(out==i);
    end
    [aa bb]=max(decodes_vote);
    trial_acc(tid,bb)=trial_acc(tid,bb)+1;
    idx=(out==tid);
    idx1=(out~=tid);
    out(idx)=1;
    out(idx1)=0;
    decodes=[decodes out];
end

for i=1:length(trial_acc)
    trial_acc(i,:)=trial_acc(i,:)/sum(trial_acc(i,:));
end

diag(trial_acc)
mean(diag(trial_acc))
mean(decodes)



