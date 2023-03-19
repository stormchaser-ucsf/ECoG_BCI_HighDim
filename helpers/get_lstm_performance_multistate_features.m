function [lstm_output] = ...
    get_lstm_performance_multistate_features(filepath,net_bilstm,Params,lpFilt)



file=filepath;

% load the data, and run it through the classifier
decodes_overall=[];
data=[];
len=1000;

% load
load(file)

% create buffer
data_buffer = randn(len,128)*0.25;

%get data
raw_data = TrialData.BroadbandData;
raw_data1=cell2mat(raw_data');

% state of trial
state_idx = TrialData.TaskState;
decodes=[];
decodes_value=[];
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

    % just raw output
    out=activations(net_bilstm,neural_features','fc_2');


    % store results
    if state_idx(j)==3
        decodes=[decodes class_predict];
        decodes_value = [decodes_value out];
    end
end

% return the output
lstm_output = decodes_value;

% 
% data.Task = 'MultiState';
% decodes_overall.decodes = decodes;
% decodes_overall.decodes_value = decodes_value;
% decodes_overall.tid = TrialData.TargetID;
% 
% 
% 
% % looking at the accuracy of the bilstm decoder overall
% targets = unique([decodes_overall(1:end).tid]);
% acc=zeros(length(targets),7);
% for iter=1:length(targets)
%     trials = find([decodes_overall(1:end).tid]==targets(iter));
%     for j=1:length(trials)
%         tmp = decodes_overall(trials(j)).decodes;
%         for k=1:length(tmp)
%             acc(iter,tmp(k)) = acc(iter,tmp(k))+1;
%         end
%     end
% end
% 
% for i=1:size(acc,1)
%     acc(i,:) = acc(i,:)./sum(acc(i,:));
% end
% 
% 
% 
% % looking at the accuracy of the bilstm decoder overall
% targets = unique([decodes_overall(1:end).tid]);
% lstm_output={};
% for iter=1:length(targets)
%     trials = find([decodes_overall(1:end).tid]==targets(iter));
%     tmp_out=[];
%     for j=1:length(trials)
%         tmp = decodes_overall(trials(j)).decodes_value;
%         tmp_out = [tmp_out tmp];
%     end
%     lstm_output{iter}=tmp_out;
% end
% 




%
% for i=1:length(acc)
%     acc(i,:) = acc(i,:)/sum(acc(i,:));
% end
%
% % looking at accuracy in terms of max decodes
% acc_trial=zeros(7,7);
% for i=1:length(decodes_overall)
%     tmp = decodes_overall(i).decodes;
%     tid=decodes_overall(i).tid;
%     acc1=zeros(7,7);
%     for j=1:length(tmp)
%         acc1(tid,tmp(j)) =  acc1(tid,tmp(j))+1;
%     end
%     acc1=sum(acc1);
%     [aa bb]=max(acc1);
%     acc_trial(tid,bb)=acc_trial(tid,bb)+1;
% end
% for i=1:length(acc_trial)
%     acc_trial(i,:) = acc_trial(i,:)/sum(acc_trial(i,:));
% end
% 
% 
% % return outputs
% acc_lstm_sample = acc;





