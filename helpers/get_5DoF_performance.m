
clc;clear

root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';

foldernames = {'20230915'};
cd(root_path)

files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'DiscreteArrow5D');
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D((j)).name,'BCI_Fixed');
        if exist(filepath)
            disp(filepath)
            files = [files;findfiles('',filepath)'];
        end
    end
end


% look at the decodes per direction to get a max vote
T=zeros(5);
Tbin=zeros(5);
tim_to_target=[];
num_suc=[];
num_fail=[];
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
        clicker_state = TrialData.ClickerState;

        %run it through a 3 bin mode filter
        mode_filter = clicker_state;  
        mode_bins=4;
        for j=mode_bins:length(mode_filter)
            bins = clicker_state(j-mode_bins+1:j);
            mode_filter(j)=mode(bins);            
        end        
        clicker_state = mode_filter ;

        idx = TrialData.TargetID;
        t(1) = sum(clicker_state ==1);
        t(2) = sum(clicker_state ==2);
        t(3) = sum(clicker_state ==3);
        t(4) = sum(clicker_state ==4);
        t(5) = sum(clicker_state ==5);        
        % get the bin level accuracy
        Tbin(idx,:)=Tbin(idx,:)+t;
        [aa bb]=max(t);
        T(idx,bb) = T(idx,bb)+1;
        if TrialData.TargetID == TrialData.SelectedTargetID
            tim_to_target = [tim_to_target length(clicker_state)-TrialData.Params.ClickCounter];
            num_suc = [num_suc 1];
        else%if TrialData.SelectedTargetID ==0%~= TrialData.SelectedTargetID %&& TrialData.SelectedTargetID~=0
            tim_to_target = [tim_to_target length(clicker_state)];
            num_fail = [num_fail 1];
        end
    end
end

% get acc. trial level 
for i=1:size(T)
    T(i,:) = T(i,:)./sum(T(i,:));
end
figure;imagesc(T)
colormap bone
caxis([0 1])
xticks([1:5])
yticks([1:5])
xticklabels({'Rt thumb','Tong','Lt. thumb','Rot Rt Wrist', 'Both middle'})
yticklabels({'Rt thumb','Tong','Lt. thumb','Rot Rt Wrist', 'Both middle'})
set(gcf,'Color','w')
set(gca,'FontSize',12)
colorbar
mean(diag(T))

% get acc. bin level 
for i=1:size(Tbin)
    Tbin(i,:) = Tbin(i,:)./sum(Tbin(i,:));
end
figure;imagesc(Tbin)
colormap bone
caxis([0 0.8])
xticks([1:5])
yticks([1:5])
xticklabels({'Rt thumb','Tong','Lt. thumb','Rot Rt Wrist', 'Both middle'})
yticklabels({'Rt thumb','Tong','Lt. thumb','Rot Rt Wrist', 'Both middle'})
set(gcf,'Color','w')
set(gca,'FontSize',12)
colorbar
mean(diag(Tbin))
