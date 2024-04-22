
clc;clear
close all
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')

foldername='20210615';
foldernames=dir(fullfile(root_path,foldername,'Robot3DArrow'));
foldernames=foldernames(3:end);

%non_pooling=[5,8,11]

pooling=[];non_pooling=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path,foldername,...
        'Robot3DArrow',foldernames(i).name,'BCI_Fixed');

    if exist(folderpath,'dir') > 0
        files=findfiles('',folderpath);
        if ~isempty(files)
            load(files{1})
            if TrialData.Params.ChPooling ==1
                pooling = [pooling i];
            elseif TrialData.Params.ChPooling ==0
                non_pooling = [non_pooling i];
            end
        end
    end
end


pooling_files=[];
for i=1:length(pooling)
      folderpath = fullfile(root_path,foldername,...
        'Robot3DArrow',foldernames(pooling(i)).name,'BCI_Fixed');
      
      pooling_files =[pooling_files;findfiles('',folderpath)'];
end

acc_pooling = accuracy_online_data(pooling_files);
figure;imagesc(acc_pooling*100)
colormap(brewermap(128,'Blues'))
clim([0 100])
set(gcf,'color','w')
% add text
for j=1:size(acc_pooling,1)
    for k=1:size(acc_pooling,2)
        if j==k
            text(j-0.35,k,num2str(round(100*acc_pooling(k,j),1)),'Color','w')
        else
            text(j-0.35,k,num2str(round(100*acc_pooling(k,j),1)),'Color','k')
        end
    end
end
box on
xticks(1:7)
yticks(1:7)
xticklabels({'Rt. Thumb','Left leg','Lt. Thumb','Head','Lips','Tongue','Both middle'})
yticklabels({'Rt. Thumb','Left leg','Lt. Thumb','Head','Lips','Tongue','Both middle'})
title([num2str(mean(diag(acc_pooling*100))) '% accuracy'])

non_pooling_files=[];
for i=1:length(non_pooling)
      folderpath = fullfile(root_path,foldername,...
        'Robot3DArrow',foldernames(non_pooling(i)).name,'BCI_Fixed');
      
      non_pooling_files =[non_pooling_files;findfiles('',folderpath)'];
end

acc_non_pooling = accuracy_online_data(non_pooling_files);
figure;imagesc(acc_non_pooling*100)
colormap(brewermap(128,'Blues'))
clim([0 100])
set(gcf,'color','w')
% add text
for j=1:size(acc_non_pooling,1)
    for k=1:size(acc_non_pooling,2)
        if j==k
            text(j-0.35,k,num2str(round(100*acc_non_pooling(k,j),1)),'Color','w')
        else
            text(j-0.35,k,num2str(round(100*acc_non_pooling(k,j),1)),'Color','k')
        end
    end
end
box on
xticks(1:7)
yticks(1:7)
xticklabels({'Rt. Thumb','Left leg','Lt. Thumb','Head','Lips','Tongue','Both middle'})
yticklabels({'Rt. Thumb','Left leg','Lt. Thumb','Head','Lips','Tongue','Both middle'})
title([num2str(mean(diag(acc_non_pooling*100))) '% accuracy'])


% plot individual block examples
pooling_acc_blks=[];
for i=1:length(pooling)
    folderpath = fullfile(root_path,foldername,...
        'Robot3DArrow',foldernames(pooling(i)).name,'BCI_Fixed');
    tmp = accuracy_online_data(findfiles('',folderpath)');
    pooling_acc_blks(i) = mean(diag(tmp));
end
 %pooling_acc_blks=pooling_acc_blks+0.001*randn(size(pooling_acc_blks));

non_pooling_acc_blks=[];
for i=1:length(non_pooling)
    folderpath = fullfile(root_path,foldername,...
        'Robot3DArrow',foldernames(non_pooling(i)).name,'BCI_Fixed');
    tmp = accuracy_online_data(findfiles('',folderpath)');
    non_pooling_acc_blks(i) = mean(diag(tmp));
end

if length(pooling_acc_blks)>length(non_pooling_acc_blks)
    non_pooling_acc_blks(end+1:length(pooling_acc_blks)) = NaN;
else
    pooling_acc_blks(end+1:length(non_pooling_acc_blks)) = NaN;
end

figure;boxplot(100*[non_pooling_acc_blks' pooling_acc_blks'])
xticks(1:2)
xticklabels({'No Pooling','Pooling'})
set(gcf,'Color','w')
box off
ylabel('Accuracy')
yticks([30:20:90])


 [P,H,STATS] = ranksum(non_pooling_acc_blks,pooling_acc_blks);


