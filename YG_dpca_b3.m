%% Data Fomatting Continuous Robot

% Robot 3D
clc;
clear;close all
addpath('C:\Users\nikic\Documents\MATLAB')
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
root_path ='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
foldernames = {'20230322', '20230323', '20230329', '20230330',...
    '20230405', '20230419', '20230420'};
cd(root_path)


files=[]; 
for ii=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{ii},'Robot3D');
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name);        
        if ~exist(fullfile(folderpath,D(j).name, 'BCI_Fixed'), 'dir')
            continue
        else
            filepath=fullfile(folderpath,D(j).name, 'BCI_Fixed');
            files = [files;findfiles('',filepath)'];
        end
    end
end

firingRatesAverage = [];
firingRates = []; firingRates1 = []; firingRates2 =[];firingRates3=[];
firingRates4=[];firingRates5=[]; firingRates6=[]; firingRates7=[];
full_b=[];full_d=[];full_lg=[];full_hg=[];

% filter bank hg
Params=[];
Params.Fs = 1000;
Params.FilterBank(1).fpass = [70,77];   % high gamma1
Params.FilterBank(end+1).fpass = [77,85];   % high gamma2
Params.FilterBank(end+1).fpass = [85,93];   % high gamma3
Params.FilterBank(end+1).fpass = [93,102];  % high gamma4
Params.FilterBank(end+1).fpass = [102,113]; % high gamma5
Params.FilterBank(end+1).fpass = [113,124]; % high gamma6
Params.FilterBank(end+1).fpass = [124,136]; % high gamma7
Params.FilterBank(end+1).fpass = [136,150]; % high gamma8
Params.FilterBank(end+1).fpass = [30,36]; % lg1
Params.FilterBank(end+1).fpass = [36,42]; % lg2
Params.FilterBank(end+1).fpass = [42,50]; % lg3
% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end


load('ECOG_Grid_8596_000063_B3.mat', 'ecog_grid'); chMap = ecog_grid;

for i = 1:length(files)
    disp(i/length(files)*100)
    load(files{i})
    ind = 1;
    %features = TrialData.NeuralFeatures;
    %feats = cell2mat(features);
    features = TrialData.BroadbandData;
    features = cell2mat(features')';
    bad_ch = [108 113 118];
    good_ch = ones(size(features,1),1);
    good_ch(bad_ch)=0;
    if size(features,2)>=4e3
        features = features(logical(good_ch),1:4e3);
    end

    tmp=features';
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8 % only hg
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));
    tmp_hg=resample(tmp_hg,size(tmp_hg,1)/10,size(tmp_hg,1));
    tmp_hg= tmp_hg(20:end-20,:);
    features = tmp_hg';

    % throwaway first few PCs
%     [c,s,l]=pca(features');
%     aa = s(:,2:end)*c(:,2:end)';
%     features = aa';

    % B3 grid    

    % get raw data


    % get delta, beta and hG removing bad channels
%     feats = feats([257:512 1025:1280 1281:1536 1537:1792],:);
%     bad_ch = [108 113 118];
%     good_ch = ones(size(feats,1),1);
%     for ii=1:length(bad_ch)
%         bad_ch_tmp = bad_ch(ii)*[1 2 3 4];
%         good_ch(bad_ch_tmp)=0;
%     end
%     feats = feats(logical(good_ch),:);
% 
%     features_d = feats(1:253,:);
%     features_b = feats(254:506,:);
%     features_lg = feats(507:759,:);
%     features_hg = feats(760:1012,:);
% 
%     fs = TrialData.Params.UpdateRate;
    %fn = fs/2;

    fs=1e3;


    target = TrialData.TargetID;
    
    kinax = TrialData.TaskState;
    state1 = find(kinax==1);s1 = length(state1);
    state2 = find(kinax==2);s2 = length(state2);
    state3 = find(kinax==3);s3 = length(state3);
    state4 = find(kinax==4);s4 = length(state4);
    states(i, :) = [s1 s2 s3 s4];

    % z-score each trial's data to state 1 and state 2
%     bl_data = features_hg(:,[state1 state2]);
%     m = mean(bl_data,2);
%     s = std(bl_data',1)';
%     features_hg = (features_hg-m)./s;

    %     % Interpolate
    %     tmp_data_b = features_hg(:,state3); % Gathering the neural data from state 3
    %     tb = (1/fs)*[1:size(tmp_data_b,2)];
    %     t=(1/fs)*[1:50]; tb = tb*t(end)/tb(end);   %Why do this? What do we get from this? -> Normalize each trial's length to be the same prior to averaging
    %     new_state3_b = interp1(tb,tmp_data_b',t,'spline')';

    % take only the first 20bins of data from state 3

    %tmp_data_b = features_hg(:,state3); % Gathering the neural data from state 3
    tmp_data_b = features;
    
    if size(tmp_data_b,2)>20
        %new_state3_b = tmp_data_b(:,1:20);

%         if s4<10
%             features_hg = [features_hg repmat(features_hg(:,end),1,10-s4-1)];
%             state4 = [state4 state4(end)+1:size(features_hg,2)];
%         end
        %new_features_b = [features_hg(:, [state1 state2]) new_state3_b features_hg(:, state4)];
%         new_features_b = [features_hg(:, [state1 state2]) new_state3_b ];
% 
%         if size(new_features_b,2) == 30
%             new_features_b = new_features_b(:,2:end);
%         end

        % upsample by a factor of 2
        %new_features_b = resample(new_features_b',3,1)';
        new_features_b = features;




        full_b = cat(3, full_b, new_features_b);

        %Target Specific
        if target == 1
            firingRates1 = cat(3, firingRates1, new_features_b);
        elseif target == 2
            firingRates2 = cat(3, firingRates2, new_features_b);
        elseif target == 3
            firingRates3 = cat(3, firingRates3, new_features_b);
        elseif target == 4
            firingRates4 = cat(3, firingRates4, new_features_b);
        elseif target == 5
            firingRates5 = cat(3, firingRates5, new_features_b);
        elseif target == 6
            firingRates6 = cat(3, firingRates6, new_features_b);
        elseif target == 7
            firingRates7 = cat(3, firingRates7, new_features_b);
        end

    end
end

% % Target Specific
% firingRates_a= permute(cat(4,firingRates1(:,:,1:end), firingRates2(:,:,1:end),...
%     firingRates3(:,:,1:end), firingRates4(:,:,1:end), firingRates5(:,:,1:end), ...
%     firingRates6(:,:,1:end)), [1 4 2 3]);
% size(firingRates_a)
% firingRatesAverage = mean(firingRates_a,4); 

% Data format for dPCA: Channels X Targets X Time
% data for single trials extracted is Channels X Time X Epochs
firingRatesAverage=[];
k=1;
for i=1:6
    %if i~=6 
        varname = genvarname(['firingRates' num2str(i)]);
        tmp_avg = squeeze(nanmean(eval(varname),3));
        tmp_avg = zscore(tmp_avg')';
        [c,s,l]=pca(tmp_avg');
        figure;plot(s(:,1))
        aa = s(:,2:end)*c(:,2:end)';
        tmp_avg=aa';
        firingRatesAverage(:,k,:) = tmp_avg;k=k+1;
        %figure;imagesc(tmp_avg);colorbar;title(num2str(i))
    %end
end

% dPCA Analysis
addpath('C:\Users\nikic\Documents\GitHub\dPCA\matlab')
%addpath('C:\Users\ygraham\Documents\dPCA-master\matlab\')
% Single Task Analysis
dim = 20;
combinedParams = {{1, [1 2]}, {2}};
margNames = {'Target', 'Time'};

% Multiple Tasks
% combinedParams = {{1, [1 3]}, {2, [2 3]}, {3}, {[1 2], [1 2 3]}};
% margNames = {'Target', 'Task', 'Time', 'Target/Task'};
% margColours = [23 100 171; 187 20 25; 150 150 150; 114 97 171]/256;

% simple stuff
time=1:size(firingRatesAverage,3); % remember to change this when comparing tasks
time = 1/10*time; % Making time in seconds
%timeEvents = time(round(length(time)/2));
timeEvents = time(29);
margColours = [23 100 171; 187 20 25; 150 150 150]/256;


% noise correction
firingRatesAverage=firingRatesAverage+0.001*randn(size(firingRatesAverage));

[W,V,whichMarg] = dpca(firingRatesAverage, dim, ...
    'combinedParams', combinedParams);


explVar = dpca_explainedVariance(firingRatesAverage, W, V, ...
    'combinedParams', combinedParams);

dpca_plot(firingRatesAverage, W, V, @dpca_plot_default, ...
    'explainedVar', explVar, ...
    'marginalizationNames', margNames, ...
    'marginalizationColours', margColours, ...
    'whichMarg', whichMarg,                 ...
    'time', time,                        ...
    'timeEvents', timeEvents,               ...
    'timeMarginalization', 3, ...
    'legendSubplot', 16);


% plotting one by one the dPCA mode time series for movement
xx=find(whichMarg==1);
out=[];
for i=1:size(firingRatesAverage,2)
    tmp = squeeze(firingRatesAverage(:,i,:));
    out(:,i) = tmp'*W(:,xx(1));
end
figure;plot(out)

%%
load('ECOG_Grid_8596_000063_B3.mat', 'ecog_grid');
chMap = ecog_grid;

set(gcf, 'Color', 'w')
spatial_dpca_batch1 = V(:,1);
figure;imagesc(spatial_dpca_batch1(chMap))
colorbar
title('dPCA Component 1')
save('dpc1.mat', 'spatial_dpca_batch1')

set(gcf, 'Color', 'w')
spatial_dpca_batch2 = V(:,2);
figure;imagesc(spatial_dpca_batch2(chMap))
colorbar
title('dPCA Component 2')
save('dpc2.mat', 'spatial_dpca_batch2')

set(gcf, 'Color', 'w')
spatial_dpca_batch3 = V(:,3);
figure;imagesc(spatial_dpca_batch3(chMap))
colorbar
title('dPCA Component 3')
save('dpc3.mat', 'spatial_dpca_batch3')

set(gcf, 'Color', 'w')
spatial_dpca_batch4 = V(:,4);
figure;imagesc(spatial_dpca_batch4(chMap))
colorbar
title('dPCA Component 4')
save('dpc4.mat', 'spatial_dpca_batch4')

set(gcf, 'Color', 'w')
spatial_dpca_batch5 = V(:,5);
figure;imagesc(spatial_dpca_batch5(chMap))
colorbar
title('dPCA Component 5')
save('dpc5.mat', 'spatial_dpca_batch5')