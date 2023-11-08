%% DPCA using smoothed neural features and interpolating in the arrow task 

clc;clear
%close all
addpath('C:\Users\nikic\Documents\MATLAB')
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
root_path ='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
foldernames = {'20230301'};
cd(root_path)

imagined_files=[];
online_files=[];

for ii=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{ii},'Robot3DArrow');
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name);
        if exist(fullfile(folderpath,D(j).name, 'Imagined'), 'dir')
            filepath=fullfile(folderpath,D(j).name, 'Imagined');
            imagined_files = [imagined_files;findfiles('',filepath)'];
        elseif exist(fullfile(folderpath,D(j).name, 'BCI_Fixed'), 'dir')
            filepath=fullfile(folderpath,D(j).name, 'BCI_Fixed');
            online_files = [online_files;findfiles('',filepath)'];
        end
    end
end

% load all the  files to run dPCA
firingRatesAverage = [];
firingRates1 = []; firingRates2 =[];firingRates3=[];
firingRates4=[];firingRates5=[]; firingRates6=[]; firingRates7=[];

for i=1:length(online_files)

    disp(i/length(online_files)*100)
    files_loaded=true;
    try
        load(online_files{i});
    catch
        files_loaded=false;
    end

    if files_loaded
        features  = TrialData.SmoothedNeuralFeatures;
        features = cell2mat(features);
        %     if TrialData.TargetID==6
        %         disp(i)
        %         break
        %     end

        % extract the beta features
        features = features(1025:1280,:);% beta        
        bad_ch = [108 113 118];
        good_ch = ones(size(features,1),1);
        good_ch(bad_ch)=0;
        features = features(logical(good_ch),:);

        % get it to state 3
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

        % interpolation for online control to 25 bins
        fs = TrialData.Params.UpdateRate;
        tb = (1/fs)*[1:size(tmp_data,2)];
        t=(1/fs)*[1:25]; % increase it to 25
        tb = tb*t(end)/tb(end);
        tmp_data1 = interp1(tb,tmp_data',t,'pchip')';
        idx3 = interp1(tb,idx3,t);
        % now stick all the data together
        trial_dur = (length(state3)-TrialData.Params.ClickCounter)*(1/fs);
        data = [features(:,[state1 state2]) tmp_data1 features(:,[state4])];


        % for imagined control use as is
%         tmp_data1=tmp_data;
%         data = [features(:,[state1 state2]) tmp_data1 features(:,[state4])];

        % baseline each trial to state1 data
        m = nanmean(features(:,state1),2);
        s = nanstd(features(:,state1)',1)';
        %data = (data-m)./s; %z-score
        %data = (data-m); % demean

        % artifact correction: if abs value of any z-score is greater than
        % some threshold
        if max(max(abs(data')))<50
            % get the data
            if TrialData.TargetID == 1
                firingRates1 = cat(3,firingRates1,data);
            elseif TrialData.TargetID == 2
                firingRates2 = cat(3,firingRates2,data);
            elseif TrialData.TargetID == 3
                firingRates3 = cat(3,firingRates3,data);
            elseif TrialData.TargetID == 4
                firingRates4 = cat(3,firingRates4,data);
            elseif TrialData.TargetID == 5
                firingRates5 = cat(3,firingRates5,data);
            elseif TrialData.TargetID == 6
                firingRates6 = cat(3,firingRates6,data);
            elseif TrialData.TargetID == 7
                firingRates7 = cat(3,firingRates7,data);
            end
        end
    end
end

% 
% for i=1:size(firingRates1,3)
%     a=squeeze(firingRates1(:,:,i));
%     figure;imagesc(a)
%     title(num2str(i))
% end

firingRatesAverage=[];
k=1;
for i=1:7    
    varname = genvarname(['firingRates' num2str(i)]);
    tmp_avg = squeeze(nanmedian(eval(varname),3));
    %tmp_avg = zscore(tmp_avg')';
    %[c,s,l]=pca(tmp_avg');
    %figure;plot(s(:,1))
    %aa = s(:,2:end)*c(:,2:end)';
    %tmp_avg=aa';
    firingRatesAverage(:,k,:) = tmp_avg;k=k+1;    
end

%%%%%% DPCA ANALYSIS
addpath('C:\Users\nikic\Documents\GitHub\dPCA\matlab')
dim = 40;
combinedParams = {{1, [1 2]}, {2}};
margNames = {'Target', 'Time'};

fs=5;
time=1:size(firingRatesAverage,3); % remember to change this when comparing tasks
time = (1/fs)*time; % Making time in seconds
%timeEvents = time(round(length(time)/2));
timeEvents = [time(5) time(10) time(35)];
margColours = [23 100 171; 187 20 25; 150 150 150]/256;

% noise correction
%firingRatesAverage=firingRatesAverage+0.001*randn(size(firingRatesAverage));

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


% % plotting one by one the dPCA mode time series for movement
% xx=find(whichMarg==1);
% out=[];
% for i=1:size(firingRatesAverage,2)
%     tmp = squeeze(firingRatesAverage(:,i,:));
%     out(:,i) = tmp'*W(:,xx(1));
% end
% figure;plot(out)


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

for i = 1:20%length(files)
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
        features = features(logical(good_ch),1:4e3); % 1s state 1, approx 0.75-1s state 2, rest state 3
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
    tmp_hg=resample(tmp_hg,size(tmp_hg,1)/10,size(tmp_hg,1)); % downsample by 10
    tmp_hg= tmp_hg(20:end-20,:); %get rid of low-pass filtering  artifacts from resampling
    features = tmp_hg';
    figure;
    subplot(2,1,1)
    imagesc(features)
    ylabel('Channels')
    xlabel('Time')
    subplot(2,1,2)
    plot(mean(features,1))
    vline(80,'r') % when state 2 started
    vline(155,'g')% when state 3 started approx (1s state 1 and 0.75s state 2, check individual trials)
    axis tight
    ylabel('Mean')
    xlabel('Time')
    sgtitle(['hG Target number ' num2str(TrialData.TargetID)])
end

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