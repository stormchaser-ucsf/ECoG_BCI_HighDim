function [kin_data] = get_kinematics(files)


D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
figure;
hold on
cmap=turbo(6);
kk=1;
kin_data={};
for ii=1:length(files)
    disp(ii)
    file_loaded=1;
    try
        load(files{ii});
    catch
        file_loaded=0;
    end
    if file_loaded
        features  = TrialData.SmoothedNeuralFeatures;
        kinax = TrialData.TaskState;
        kinax = [find(kinax==3)];
        temp = cell2mat(features(kinax));

        % get the kinematics data and the target data
        kindata = TrialData.CursorState;
        kindata = kindata(1:3,kinax);
        kindata = kindata(:,2:11) - kindata(:,1);        
        target_pos = TrialData.TargetPosition;
        col = cmap(TrialData.TargetID,:);
        plot3(kindata(1,:),kindata(2,:),kindata(3,:),'LineWidth',2,'Color',col)
        trial_error = sqrt(sum(sum((kindata - target_pos').^2)));
        kin_data(kk).TargetID = TrialData.TargetID;
        kin_data(kk).filename = files{ii};
        kin_data(kk).error = trial_error;
        kk=kk+1;
    end
end


set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('X-axis')
ylabel('Y-axis')
zlabel('Z-axis')


end