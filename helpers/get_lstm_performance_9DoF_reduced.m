function [acc_lstm_sample,acc_mlp_sample,acc_lstm_trial,acc_mlp_trial] = ...
    get_lstm_performance_9DoF_reduced(files,net_bilstm,Params,lpFilt,num_targets)


%
% % get the name of the files
% files = findfiles('mat',filepath,1)';
% files1=[];
% for i=1:length(files)
%     if regexp(files{i},'Data')
%         files1=[files1;files(i)];
%     end
% end
% files=files1;
% clear files1
%
%


% load the data, and run it through the classifier
decodes_overall=[];
data=[];
len=1000;
Trial_Data={};n=1;
k=1;
for i=1:length(files)
    disp(i/length(files)*100)

    % load
    files_loaded=true;
    try
        load(files{i})
    catch
        files_loaded=false;
    end

    if files_loaded

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
                Trial_Data(n).neural_features = neural_features;
                Trial_Data(n).TargetID = TrialData.TargetID;
                Trial_Data(n).TrialID = i;
                n=n+1;
            end
        end

        %    decodes = decodes(1:25); % first 5s
        data(k).task_state = TrialData.TaskState ;
        %data(k).raw_data = trial_data;
        data(k).TargetID = TrialData.TargetID;
        %data(k).Task = 'Robot_Online_Data';
        data(k).Task = 'RobotBatch';
        decodes_overall(k).decodes = decodes;
        if TrialData.TargetID==1
            decodes_overall(k).tid = TrialData.TargetID;
        elseif TrialData.TargetID==6
            decodes_overall(k).tid =2;
        elseif TrialData.TargetID==3
            decodes_overall(k).tid = TrialData.TargetID;
        elseif TrialData.TargetID==8
            decodes_overall(k).tid =4;
        elseif TrialData.TargetID==7
            decodes_overall(k).tid =5;
        else
            decodes_overall(k).tid = [];
        end

        k=k+1;
    end
end
%val_robot_data=data;
%save val_robot_data val_robot_data -v7.3

% looking at the accuracy of the bilstm decoder overall
acc=zeros(num_targets,num_targets);
for i=1:length(decodes_overall)
    if length(decodes_overall(i).tid)
        tmp = decodes_overall(i).decodes;
        tid=decodes_overall(i).tid;
        for j=1:length(tmp)
            acc(tid,tmp(j)) =  acc(tid,tmp(j))+1;
        end
    end
end
for i=1:length(acc)
    acc(i,:) = acc(i,:)/sum(acc(i,:));
end

% looking at accuracy in terms of max decodes
acc_trial=zeros(num_targets,num_targets);
for i=1:length(decodes_overall)
    if length(decodes_overall(i).tid)
        tmp = decodes_overall(i).decodes;
        tid=decodes_overall(i).tid;
        acc1=zeros(num_targets,num_targets);
        for j=1:length(tmp)
            acc1(tid,tmp(j)) =  acc1(tid,tmp(j))+1;
        end
        acc1=sum(acc1);
        [aa bb]=max(acc1);
        acc_trial(tid,bb)=acc_trial(tid,bb)+1;
    end
end
for i=1:length(acc_trial)
    acc_trial(i,:) = acc_trial(i,:)/sum(acc_trial(i,:));
end


%comparing to the mlp
acc_mlp=zeros(num_targets,num_targets);
acc_mlp_trial=zeros(num_targets,num_targets);
for i=1:length(files)
    %disp(i)

    % load
    files_loaded=true;
    try
        load(files{i})
    catch
        files_loaded=false;
    end

    if files_loaded

        decodes = TrialData.ClickerState;
        %decodes = decodes(1:25);
        tid=TrialData.TargetID;
        acc1=zeros(num_targets,num_targets);
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



