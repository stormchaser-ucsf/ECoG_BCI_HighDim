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
    load('ECOG_Grid_8596_000067_B3.mat')
    condn_data = [condn_data;load_data_for_MLP_TrialLevel_B3(files,ecog_grid,0)];

    %%%%%% load online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
    condn_data = [condn_data;load_data_for_MLP_TrialLevel_B3(files,ecog_grid,1) ];

    %%%%%% load batch data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end
    condn_data = [condn_data;load_data_for_MLP_TrialLevel_B3(files,ecog_grid,2) ];
end

% make them all into one giant struct
tmp=cell2mat(condn_data(i));
condn_data_overall=tmp;
for i=2:length(condn_data)
    tmp=cell2mat(condn_data(i));
    for k=1:length(tmp)
        condn_data_overall(end+1) =tmp(k);
    end
end

%cv_acc_overall={};
%cv_acc2_overall={};
%cv_acc3_overall={};
cv_acc3={};
for iter=1:5

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
    num_units = [64,96,150,256];
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
                layers = get_layers1(num_units(j),759);
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
                    layers = get_layers2(num_units(j),num_units(k),759);
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
                        layers = get_layers(num_units(j),num_units(k),num_units(l),759);
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
    save B3_MLP_NN_Param_Optim cv_acc3 -v7.3
end

% plotting
acc=[];acc1=[];
for i=1:length(cv_acc3)
    acc(i) = mean(cv_acc3(i).cv_perf);
end

[aa bb]=max(acc)

figure;boxplot(acc(1:4))
ylim([.8 .9])
figure;boxplot(acc(5:20))
ylim([.8 .9])
figure;boxplot(acc(21:end))
ylim([.8 .9])

tmp=NaN(64,3);
tmp(1:4,1) = acc(1:4)';
tmp(1:16,2) = acc(5:20)';
tmp(1:end,3) = acc(21:end)';
figure;boxplot(tmp)

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




