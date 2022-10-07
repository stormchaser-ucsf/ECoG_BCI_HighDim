function [acc_lstm_sample,acc_mlp_sample,acc_lstm_trial,acc_mlp_trial] = ...
    get_lstm_performance_afterBatch(files,net_bilstm,Params,lpFilt)

% load the data, and run it through the classifier
decodes_overall=[];
data=[];
len=1000;
for i=1:length(files)
    %disp(i)

    % load
    load(files{i})

    % create buffer
    data_buffer = randn(len,128)*0.25;

    %get data
    raw_data = TrialData.BroadbandData;
    raw_data1=cell2mat(raw_data');

    % state of trial
    state_idx = TrialData.TaskState;
    decodes=[];
    trial_data={};
    for j=1:length(raw_data)
        %disp(j)
        tmp = raw_data{j};
        s=size(tmp,1);
        if s<len
            data_buffer = circshift(data_buffer,-s);
            data_buffer(end-s+1:end,:)=tmp;
        else
            data_buffer(1:end,:)=tmp(s-len+1:end,:);
        end

        % storing the data
        trial_data{j} = data_buffer;

        neural_features = extract_lstm_features_onlineTrials(data_buffer,Params,lpFilt);        

        % classifier output
        out=predict(net_bilstm,neural_features');
        [aa bb]=max(out);
        class_predict = bb;

        % store results
        if state_idx(j)==3
            decodes=[decodes class_predict];
        end
    end

%    decodes = decodes(1:25); % first 5s
    data(i).task_state = TrialData.TaskState ;
    %data(i).raw_data = trial_data;
    data(i).TargetID = TrialData.TargetID;
    %data(i).Task = 'Robot_Online_Data';
    data(i).Task = 'RobotBatch';
    decodes_overall(i).decodes = decodes;
    decodes_overall(i).tid = TrialData.TargetID;
end
%val_robot_data=data;
%save val_robot_data val_robot_data -v7.3

% looking at the accuracy of the bilstm decoder overall
acc=zeros(7,7);
for i=1:length(decodes_overall)
    tmp = decodes_overall(i).decodes;
    tid=decodes_overall(i).tid;
    for j=1:length(tmp)
        acc(tid,tmp(j)) =  acc(tid,tmp(j))+1;
    end
end
for i=1:length(acc)
    acc(i,:) = acc(i,:)/sum(acc(i,:));
end

% looking at accuracy in terms of max decodes
acc_trial=zeros(7,7);
for i=1:length(decodes_overall)
    tmp = decodes_overall(i).decodes;
    tid=decodes_overall(i).tid;
    acc1=zeros(7,7);
    for j=1:length(tmp)
        acc1(tid,tmp(j)) =  acc1(tid,tmp(j))+1;
    end
    acc1=sum(acc1);
    [aa bb]=max(acc1);
    acc_trial(tid,bb)=acc_trial(tid,bb)+1;
end
for i=1:length(acc_trial)
    acc_trial(i,:) = acc_trial(i,:)/sum(acc_trial(i,:));
end


%comparing to the mlp
acc_mlp=zeros(7,7);
acc_mlp_trial=zeros(7,7);
for i=1:length(files)
    %disp(i)

    % load
    load(files{i})

    decodes = TrialData.ClickerState;
    %decodes = decodes(1:25);
    tid=TrialData.TargetID;
    acc1=zeros(7,7);
    for j=1:length(decodes)
        if decodes(j)>0
            acc_mlp(tid,decodes(j))=acc_mlp(tid,decodes(j))+1;
            acc1(tid,decodes(j))=acc1(tid,decodes(j))+1;
        end
    end
    tmp=sum(acc1);
    [aa bb]=max(tmp);
    acc_mlp_trial(tid,bb)=acc_mlp_trial(tid,bb)+1;
end
for i=1:length(acc_mlp)
    acc_mlp(i,:) = acc_mlp(i,:)./sum(acc_mlp(i,:));
    acc_mlp_trial(i,:) = acc_mlp_trial(i,:)./sum(acc_mlp_trial(i,:));
end

% return outputs
acc_lstm_sample = acc;
acc_mlp_sample=acc_mlp;
acc_lstm_trial = acc_trial;
acc_mlp_trial = acc_mlp_trial;



