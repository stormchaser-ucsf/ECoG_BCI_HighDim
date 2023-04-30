function conf_matrix_overall = get_simulated_acc_PnP(net)
% function res_acc = get_simulated_acc(files,net)




%foldernames = {'20220601'};
foldernames = {'20210813','20210818','20210825','20210827','20210901','20210903',...
    '20210910','20210915','20210917','20210922','20210924'};
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';

folders={};
br_across_days={};
time2target_days=[];
acc_days=[];
conf_matrix_overall=[];
overall_trial_accuracy=zeros(7);
for i=1:10%length(foldernames)
    disp(['Processing Day ' num2str(i) ' of 10'])

    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    if i==1
        idx = [1 2 3 4];
        D = D(idx);
    elseif i==2
        idx = [1 2 3 4 6 7];
        D = D(idx);
    elseif i==3
        idx = [1 2 5 6];
        D = D(idx);
    elseif i==6
        idx = [1 2 3 4 5 6];
        D = D(idx);
    elseif i==8
        idx = [1 2 3 4 5 6 7];
        D = D(idx);
    elseif i==9
        idx = [1 2 5 6 7 9 10];
        D = D(idx);
    elseif i==11
        idx = [1 2 3  5 6 9 10 11];
        D = D(idx);
    elseif i == 10
        idx = [1 2 3 4 5  7 8];
        D = D(idx);
    end
    br=[];acc=[];time2target=[];
    for j=3:length(D)
        files=[];
        filepath=fullfile(folderpath,D((j)).name,'BCI_Fixed');
        if exist(filepath)            
            files = [files;findfiles('mat',filepath)'];
            folders=[folders;filepath];
        end
        if length(files)>0
            [acc] = compute_bitrate_net(files,7,net);
            %[b,a,t,T,ov_acc] = compute_bitrate(files,7);
            %[b,a,t,T] = compute_bitrate_constTime(files,7);
            conf_matrix_overall = cat(3,conf_matrix_overall,acc);
            %br = [br b];
            %acc = [acc mean(a)];
            %time2target = [time2target; mean(t)];
            %overall_trial_accuracy = overall_trial_accuracy + ov_acc;
            %[br, acc ,t] = [br compute_bitrate(files)];
        end
    end
    %close all
    %br_across_days{i}=br;
    %time2target_days{i} = time2target(:);
    %acc_days{i} = acc(:);
    %time2target_days = [time2target_days ;time2target(:)];
    %acc_days = [acc_days ;acc(:)];
end


