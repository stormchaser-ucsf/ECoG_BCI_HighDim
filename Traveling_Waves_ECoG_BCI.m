%% TRAVELING WAVES ECOG BCI 

% STEP 1a: state 3 of arrow task: first get a sense of the electrode clusters across
% trials that show a significant increase in spectral power compared to 1/f

% STEP 1b: Make plots showing which frequencies most prominent and across
% what electrodes

% STEP 2: to get planar traveling waves (and not rotating), use the
% relative phase method

% STEP 3: to get rotating waves method, get the gradient method from Muller

% STEP 4: experiment with different frequency bands, and especially use the
% Muller method for broadband phase

%% STEP 1A AND 1B

clc;clear
close all


%% ROBOT 3D ARROW TASK
% see if traveling waves emerge during state 3 and not during other states

clc;clear
addpath 'C:\Users\nikic\Documents\MATLAB'
% get all the files on this particular session for target 1
filepath = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220803\Robot3DArrow\110025\BCI_Fixed';
D=dir(filepath);
files={};
for j=3:length(D)
    filename = fullfile(filepath,D(j).name);
    load(filename)
    if TrialData.TargetID==1 || TrialData.TargetID==7 || TrialData.TargetID==3
        files=[files;filename];
    end
end

bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',16,'HalfPowerFrequency2',26, ...
    'SampleRate',1e3);
bpFilt2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);
col={'k','r','g'};
pow_stat2=[];
pow_stat3=[];
for i=1:length(files)
    load(files{i})
    raw_data=TrialData.BroadbandData(1:end);
    raw_data = cell2mat(raw_data');
    chmap = TrialData.Params.ChMap;
    raw_data = filtfilt(bpFilt,raw_data);
    %raw_data = abs(hilbert(filtfilt(bpFilt2,raw_data)));
    %raw_data = filtfilt(bpFilt,raw_data);
    task_state = TrialData.TaskState;
    idx = [0 diff(task_state)];
    idx=find(idx>0);

    state1 = raw_data(1:1000,:);
    state2 = raw_data(1001:2000,:);
    state3 = raw_data(2001:3400,:);
    state4 = raw_data(3401:end,:);

    %[P, f ,Phi, lambda, Xhat, z0, Z,rf]=dmd_alg(state1',1e3,0,200);
    [P1, f1 ,Phi, lambda, Xhat, z0, Z,rf]=dmd_alg(state2',1e3,0,200);
    [P2, f2 ,Phi, lambda, Xhat, z0, Z,rf]=dmd_alg(state3',1e3,5,9);
    %     figure;plot(f,P)
    %     hold on
    %     plot(f1,P1)
    %     plot(f2,P2)

    % normalize the data at each channel to be within 0 and 1
    for j=1:size(raw_data,2)
        raw_data(:,j) = rescale(raw_data(:,j));
    end
    
    % get power within 1Hz increments
    xx=0:500;
    P1x=[];
    P2x=[];
    for j=2:length(xx)
        ff = logical((f1>=xx(j-1)) .* (f1<xx(j)));
        P1x(j) = nanmean(P1(ff));

        ff = logical((f2>=xx(j-1)) .* (f2<xx(j)));
        P2x(j) = nanmean(P2(ff));
    end
    pow_stat2=[pow_stat2;P1x];
    pow_stat3=[pow_stat3;P2x];


    %
    figure;
    for j=1:size(raw_data,1)
        tmp=raw_data(j,:);
        imagesc(tmp(chmap))
        %caxis([0 1])
        colormap bone
        colorbar
        axis off
        textboxHandle = uicontrol('Style', 'text', 'Position', [0, 0, 200, 30]);
        newText = sprintf('Bin: %d', ceil( (j-200)/200 +1));
        set(textboxHandle, 'String', newText);
        pause(0.000001)
    end
end
figure;plot(nanmean(pow_stat2,1))
hold on
plot(nanmean(pow_stat3,1))

% filter it in the theta range and plot a movie

%fvtool(bpFilt)




%% SARAH'S STOP TASK


clc;clear
close all

% load the day's data for robot stop task and make a video of the robot
% kinematics, the stop signals and then a video of the traveling waves over
% cortex


% Sarah's code for viz:
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220624\RealRobotBetaStop\140342\BCI_Fixed\Data0001.mat')

filename = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220624\Python\20220624\140342\data000.csv';
P = readtable(filename);
P = table2array(P);

%%

x           = (P(:,3));  % Position
midpoint    = (-0.15 - 0.58)/2;
zc          = find(diff(sign(x-midpoint)));  % midpoint crossings based on position

figure;
subplot(4,1,1)
hold on
plot(TrialData.StopSignal, 'b')
plot(TrialData.Direction, 'r')
legend('StopSignal', 'Direction')

subplot(4,1,2)
hold on
plot(x)
xline(zc, 'r')
ylabel("x position")

subplot(4,1,3)
hold on
plot(TrialData.BetaScalar)
yline(TrialData.Params.BetaThreshold)
ylabel('beta scalar')

subplot(4,1,4)
hold on
plot(diff(x)*5)
ylabel("x velocity")
%%
figure(2)
hold on
plot(TrialData.BetaScalar)
yline(TrialData.Params.BetaThreshold)


%% PLOTTING TRAVELIN WAVES ACROSS CORTEX NOW

task_state = TrialData.TaskState;
idx= find(task_state>2);
idx= idx(1);

raw_data=TrialData.BroadbandData(idx:end);
raw_data = cell2mat(raw_data');
chmap = TrialData.Params.ChMap;

% filter it in the theta range and plot a movie
bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
         'HalfPowerFrequency1',12,'HalfPowerFrequency2',16, ...
         'SampleRate',1e3);
fvtool(bpFilt)
raw_data = filtfilt(bpFilt,raw_data);

figure;
for i=1:2:size(raw_data,1)
    tmp=raw_data(i,:);
    imagesc(tmp(chmap))        
    colormap bone        
    axis off
    textboxHandle = uicontrol('Style', 'text', 'Position', [0, 0, 200, 30]);
    newText = sprintf('Iteration: %d', ceil( (i-200)/200 +1));
    set(textboxHandle, 'String', newText);
    pause(0.001)    
end


%% TRAVELING WAVES FOR B3

clc;clear;
close all

load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\20231122\Robot3DArrow\144831\BCI_Fixed\Data0007.mat')
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data_B3
addpath 'C:\Users\nikic\Documents\MATLAB'
load('ECOG_Grid_8596_000067_B3.mat')

raw_data = cell2mat(TrialData.BroadbandData');

bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
         'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
         'SampleRate',1e3);
raw_data = filtfilt(bpFilt,raw_data);
chmap=ecog_grid';
TrialData.TaskState
figure;
for i=1:1:size(raw_data,1)
    tmp=raw_data(i,:);
    imagesc(tmp(chmap))        
    colormap bone        
    axis off
    %caxis([min(raw_data(:)) max(raw_data(:))])
    %colorbar
    textboxHandle = uicontrol('Style', 'text', 'Position', [0, 0, 200, 30]);
    newText = sprintf('Iteration: %d', ceil( (i-200)/200 +1));
    set(textboxHandle, 'String', newText);
    pause(0.001)    
end








