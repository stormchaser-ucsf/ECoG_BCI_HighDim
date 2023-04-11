function [br, acc, time2target,T,overall_acc] = compute_bitrate_net(files,num_target)


% look at the decodes per direction to get a max vote
T=zeros(num_target,num_target+1);
tim_to_target=[];
num_suc=[];
num_fail=[];
overall_acc=zeros(num_target);
for i=1:length(files)
    disp(i)
    indicator=1;
    try
        load(files{i});
    catch ME
        warning('Not able to load file, skipping to next')
        indicator = 0;
    end
    if indicator
        kinax = TrialData.TaskState;
        clicker_state = TrialData.FilteredClickerState;
        if sum(clicker_state>0)
            [aa bb]=find(clicker_state>0);
            if length(aa)>0
                selected_target = clicker_state(bb(1));
                tim_taken = bb(1)+0;
            else
                selected_target = num_target+1;
                tim_taken = length(clicker_state);
            end

            idx = TrialData.TargetID;
            %         t(1) = sum(clicker_state ==1);
            %         t(2) = sum(clicker_state ==2);
            %         t(3) = sum(clicker_state ==3);
            %         t(4) = sum(clicker_state ==4);
            %         t(5) = sum(clicker_state ==5);
            %         t(6) = sum(clicker_state ==6);
            %         t(7) = sum(clicker_state ==7);
            %         [aa bb]=max(t);
            %         T(idx,bb) = T(idx,bb)+1;
            T(idx,selected_target) = T(idx,selected_target)+1;
            if TrialData.TargetID == selected_target
                tim_to_target = [tim_to_target tim_taken];
                num_suc = [num_suc 1];
            else
                tim_to_target = [tim_to_target tim_taken];
                num_fail = [num_fail 1];
            end
            %         if TrialData.TargetID == TrialData.SelectedTargetID
            %             tim_to_target = [tim_to_target length(clicker_state)-TrialData.Params.ClickCounter];
            %             num_suc = [num_suc 1];
            %         elseif TrialData.TargetID ~= TrialData.SelectedTargetID %&& TrialData.SelectedTargetID~=0
            %             tim_to_target = [tim_to_target length(clicker_state)];
            %             num_fail = [num_fail 1];
            %         end
            clicker_state = clicker_state(clicker_state>0);
            chosen_target = mode(clicker_state);
            overall_acc(idx,chosen_target) = overall_acc(idx,chosen_target)+1;
        end
    end
end
T
for i=1:size(T)
    T(i,:) = T(i,:)./sum(T(i,:));
end
figure;imagesc(T)
colormap bone
caxis([0 1])
xticks([1:num_target+1])
yticks([1:num_target])
%xticklabels({'Rt Hand','Both Feet','Lt. Hand','Head', 'Mime up','Tong in','Both hands','Time Out'})
%yticklabels({'Rt Hand','Both Feet','Lt. Hand','Head', 'Mime up','Tong in','Both hands'})
set(gcf,'Color','w')
set(gca,'FontSize',12)
%title('Classif. using temporal history original action space')


% bit rate calculations
tim_to_target = [tim_to_target.*(1/TrialData.Params.UpdateRate)];
B = log2(7-1) * max(0,(sum(num_suc)-sum(num_fail))) / sum(tim_to_target);
title(['Accuracy of ' num2str(100*mean(diag(T))) '%' ' and bitrate of ' num2str(B)])
br=B;

time2target = tim_to_target;
acc = diag(T);

end