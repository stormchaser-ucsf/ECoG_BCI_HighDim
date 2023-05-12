function [neural,kinematics] = get_neural_kinematics_KF(files,ch_pooling)



neural=[];
kinematics=[];
for i=1:length(files)

    % load the data
    disp(i/length(files)*100)
    load(files{i})
    idx=find(TrialData.TaskState==3);
    kin = TrialData.CursorState;
    kin = kin(:,idx);
    neural_features = TrialData.SmoothedNeuralFeatures;
    neural_features = cell2mat(neural_features(idx));

    % get only non-zero velocities
    idx=abs(sum(kin(4:6,:),1))>0;
    kin = kin(:,idx);
    neural_features = neural_features(:,idx);

    % rotate decoded velocity vector towards target
    target_pos = TrialData.TargetPosition;
    xint = [];
    for j=1:size(kin,2)

        % get current position and decoded velocity
        pos = kin(1:3,j);
        recorded_vel = TrialData.IntendedCursorState(:,j);
        %recorded_vel = kin(:,j);

        % if inside target, norm velocity = 0 else compute it
        Cursor.State = pos;
        Cursor.Center = TrialData.Params.Center;
        out = InTargetRobot3D(Cursor,target_pos,...
            TrialData.Params.RobotTargetRadius);
        if out>0
            vel_mag= 0;
        else
            vel_mag = norm(recorded_vel(4:6));
        end

        % ReFit -> rotate decoded vel vector towards target
        new_vel_vector = target_pos' - pos;
        new_vel_vector = vel_mag*((new_vel_vector)/norm(new_vel_vector));
        xint = [xint new_vel_vector];
    end

    % delta, beta and hG
    fidx = [129:256 513:640 769:896];
    neural_features = neural_features(fidx,:);

    % pooling
    if ch_pooling == true
        neural_features = pool_features(neural_features,TrialData.Params.ChMap);
    end

    % store
    neural = [neural neural_features];
    kinematics = [kinematics xint];
end

