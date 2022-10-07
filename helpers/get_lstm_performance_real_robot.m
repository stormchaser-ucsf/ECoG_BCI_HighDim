function [acc_lstm_sample,acc_mlp_sample,acc_lstm_trial,acc_mlp_trial] = ...
    get_lstm_performance_real_robot(filepath,net_bilstm,Params,lpFilt)



% get the name of the files
files = findfiles('mat',filepath,1)';
files1=[];
for i=1:length(files)
    if regexp(files{i},'Data')
        files1=[files1;files(i)];
    end
end
files=files1;
clear files1




% load the data, and run it through the classifier
decodes_overall=[];
data=[];
len=1000;
for i=1:length(files)
    %disp(i)

    % load
    load(files{i})

    % create buffer
    data_buffer = randn(len,128)*0.005;

    %get data
    raw_data = TrialData.BroadbandData;
    raw_data1=cell2mat(raw_data');

    % get the output of the filtered clicker state
    filt_click_state = TrialData.FilteredClickerState;
    click_state = TrialData.ClickerState;

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



    % plotting
    click_state_freq=[];
    decodes_freq=[];
    for ii=1:7
        click_state_freq(ii) = sum(click_state==ii);
        decodes_freq(ii) = sum(decodes==ii);
    end

    tmp = [click_state_freq' decodes_freq'];
    % plotting hist
    figure;
    hold on
    for ii=1:7
        %idx = i:7:21;
        idx=ii;
        out = tmp(idx,:);
        h=bar(2*ii-0.25,mean(out(:,1)));
        h1=bar(2*ii+0.25,mean(out(:,2)));
        h.BarWidth=0.4;
        h.FaceColor=[0.2 0.2 0.7];
        h1.BarWidth=0.4;
        h1.FaceColor=[0.7 0.2 0.2];
        h.FaceAlpha=0.85;
        h1.FaceAlpha=0.85;

        %     s=scatter(ones(3,1)*2*i-0.25+0.05*randn(3,1),decodes(:,1),'LineWidth',2);
        %     s.CData = [0.2 0.2 0.7];
        %     s.SizeData=50;
        %
        %     s=scatter(ones(3,1)*2*i+0.25+0.05*randn(3,1),decodes(:,2),'LineWidth',2);
        %     s.CData = [0.7 0.2 0.2];
        %     s.SizeData=50;
    end
    xticks([2:2:14])
    xticklabels({'Right Thumb','Left Leg','Left Thumb','Head','Lips','Tongue','Both Middle'})
    ylabel('No. of Decodes')
    legend('Robot LSTM','Arrow LSTM')
    set(gcf,'Color','w')
    set(gca,'FontSize',14)
    set(gca,'LineWidth',1)

end
%val_robot_data=data;
%save val_robot_data val_robot_data -v7.3

%return outputs



