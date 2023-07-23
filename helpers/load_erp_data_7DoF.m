function [D1,D2,D3,D4,D5,D6,D7,D8,D9,tim] = load_erp_data_7DoF(files)


% load the data for each target
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
D8=[];
D9=[];
for i=1:length(files)
    %disp(i/length(files)*100)
    files_loaded=1;

    try
        load(files{i});
    catch
        files_loaded=0;
    end

    if files_loaded
        features  = TrialData.SmoothedNeuralFeatures;
        features = cell2mat(features);
        %features = features(129:256,:); %delta        
        %features = features(769:end,:); %hG
        features = features(513:640,:);% beta
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
        t=(1/fs)*[1:15];
        tb = tb*t(end)/tb(end);
        tmp_data1 = interp1(tb,tmp_data',t,'spline')';
        idx3 = interp1(tb,idx3,t,'spline');


        % now stick all the data together
        %trial_dur = (length(state3)-TrialData.Params.ClickCounter)*(1/fs);
        data = [features(:,[state1 state2]) tmp_data1 features(:,[state4])];

        % correction
        if size(data,2)<35
            data=[data data(:,end)];
        end

        % correction
        %     if length(state1)<8
        %         data  =[data(:,1) data];
        %     end

        % store the time to target data
        %     time_to_target(2,TrialData.TargetID) = time_to_target(2,TrialData.TargetID)+1;
        %     if trial_dur<=3
        %         time_to_target(1,TrialData.TargetID) = time_to_target(1,TrialData.TargetID)+1;
        %     end

        % now get the ERPs
        % if TrialData.TargetID == TrialData.SelectedTargetID && trial_dur<=3
        if TrialData.TargetID == 1
            D1 = cat(3,D1,data);
        elseif TrialData.TargetID == 2
            D2 = cat(3,D2,data);
        elseif TrialData.TargetID == 3
            D3 = cat(3,D3,data);
        elseif TrialData.TargetID == 4
            D4 = cat(3,D4,data);
        elseif TrialData.TargetID == 5
            D5 = cat(3,D5,data);
        elseif TrialData.TargetID == 6
            D6 = cat(3,D6,data);
        elseif TrialData.TargetID == 7
            D7 = cat(3,D7,data);
        elseif TrialData.TargetID == 8
            D8 = cat(3,D8,data);
        elseif TrialData.TargetID == 9
            D9 = cat(3,D9,data);
        end
        %  end
    end
end

tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);
