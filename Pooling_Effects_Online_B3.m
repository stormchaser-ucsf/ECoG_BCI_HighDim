
clc;clear
close all
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')

%foldername={'20230223','20230301','20230308','20230309','20230315'};
foldername={'20230216'};

pooling_dir={};non_pooling_dir={};
for i=1:length(foldername)

    foldernames=dir(fullfile(root_path,foldername{i},'RadialTaskMultiStateDiscreteArrow'));
    foldernames=foldernames(3:end);

    pooling=[];non_pooling=[];

    for j=1:length(foldernames)
        %     folderpath = fullfile(root_path,foldername,...
        %         'RadialTaskMultiStateDiscreteArrow',foldernames(i).name,'BCI_Fixed');

        folderpath = fullfile(root_path,foldername(i),...
            'RadialTaskMultiStateDiscreteArrow',foldernames(j).name,'BCI_Fixed');

        if exist(folderpath{1},'dir') > 0
            files=findfiles('',folderpath{1});
            if ~isempty(files)
                load(files{1})                
                if TrialData.Params.ChPooling ==1
                    pooling = [pooling j];
                elseif TrialData.Params.ChPooling ==0
                    non_pooling = [non_pooling j];
                end
            end
        end
    end
    pooling_dir{i} = pooling;
    non_pooling_dir{i} = non_pooling;
end


pooling_files=[];
for i=1:length(pooling)
      folderpath = fullfile(root_path,foldername,...
        'RadialTaskMultiStateDiscreteArrow',foldernames(pooling(i)).name,'BCI_Fixed');
      
      pooling_files =[pooling_files;findfiles('',folderpath)'];
end

acc_pooling = accuracy_online_data(pooling_files,4);
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
xticks(1:4)
yticks(1:4)
xticklabels({'Rt. Thumb','Left leg','Lt. Thumb','Head'})
yticklabels({'Rt. Thumb','Left leg','Lt. Thumb','Head'})
title([num2str(mean(diag(acc_pooling*100))) '% accuracy'])

non_pooling_files=[];
for i=1:length(non_pooling)
      folderpath = fullfile(root_path,foldername,...
        'RadialTaskMultiStateDiscreteArrow',foldernames(non_pooling(i)).name,'BCI_Fixed');
      
      non_pooling_files =[non_pooling_files;findfiles('',folderpath)'];
end

acc_non_pooling = accuracy_online_data(non_pooling_files,4);
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
xticks(1:4)
yticks(1:4)
xticklabels({'Rt. Thumb','Left leg','Lt. Thumb','Head'})
yticklabels({'Rt. Thumb','Left leg','Lt. Thumb','Head'})
title([num2str(mean(diag(acc_non_pooling*100))) '% accuracy'])


% plot individual block examples
pooling_acc_blks=[];
for i=1:length(pooling)
    folderpath = fullfile(root_path,foldername,...
        'RadialTaskMultiStateDiscreteArrow',foldernames(pooling(i)).name,'BCI_Fixed');
    tmp = accuracy_online_data(findfiles('',folderpath)',4);
    pooling_acc_blks(i) = mean(diag(tmp));
end
 %pooling_acc_blks=pooling_acc_blks+0.001*randn(size(pooling_acc_blks));

non_pooling_acc_blks=[];
for i=1:length(non_pooling)
    folderpath = fullfile(root_path,foldername,...
        'RadialTaskMultiStateDiscreteArrow',foldernames(non_pooling(i)).name,'BCI_Fixed');
    tmp = accuracy_online_data(findfiles('',folderpath)',4);
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
yticks([20:20:100])


 [P,H,STATS] = ranksum(non_pooling_acc_blks,pooling_acc_blks);


