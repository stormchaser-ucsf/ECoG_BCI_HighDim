function [D1,D2,D3,D4,idx1,idx2,idx3,idx4] = load_erp_data_online_B2(files)


% load the data for each target
D1=[];
D2=[];
D3=[];
D4=[];
for i=1:length(files)    
    load(files{i});
    disp([i ])
    features  = TrialData.SmoothedNeuralFeatures;
    features = cell2mat(features);
    features = features(769:end,:);
    fs = TrialData.Params.UpdateRate;
    kinax = TrialData.TaskState;
    state1 = find(kinax==1);
    state2 = find(kinax==2);
    state3 = find(kinax==3);
    state4 = find(kinax==4);
    tmp_data = features(:,state3);
    idx1= ones(length(state1),1);
    idx2= 2*ones(length(state2),1);
    idx3= 3*ones(length(state3),1);
    idx4= 4*ones(length(state4),1);
    
    % interpolate
    tb = (1/fs)*[1:size(tmp_data,2)];
    t=(1/fs)*[1:10];
    tb = tb*t(end)/tb(end);
    tmp_data1 = interp1(tb,tmp_data',t,'spline')';
    idx3 = interp1(tb,idx3,t,'spline');
    
    % now stick all the data together
    trial_dur = (length(state3)-TrialData.Params.ClickCounter)*(1/fs);
    data = [features(:,[state1 state2]) tmp_data1 features(:,[state4])];
   
%     if length(state1)<8
%         data  =[data(:,1) data];
%     end
%     
%     % store the time to target data
%     if TrialData.TargetID <=6
%         time_to_target(2,TrialData.TargetID) = time_to_target(2,TrialData.TargetID)+1;
%         if trial_dur<=3
%             time_to_target(1,TrialData.TargetID) = time_to_target(1,TrialData.TargetID)+1;
%         end
%     end
%     
%     if size(data,2)==31
%         data=[data(:,1) data];
%     end
%     
    % now get the ERPs    
    %if TrialData.TargetID == TrialData.SelectedTargetID && trial_dur<=4
        if TrialData.TargetID == 1
            D1 = cat(3,D1,data);
        elseif TrialData.TargetID == 2
            D2 = cat(3,D2,data);
        elseif TrialData.TargetID == 3
            D3 = cat(3,D3,data);
        elseif TrialData.TargetID == 4
            D4 = cat(3,D4,data);        
        end
    %end
end

end

