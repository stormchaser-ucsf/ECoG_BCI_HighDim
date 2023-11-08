%% BUILDING AN ENGAGEMENT DISENGAGEMENT CLASSIFIER

%% looking at traveling waves in and around disengegement 


clc;clear
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20230830\MultiTargets2D_2Stage\101957\BCI_Fixed\Data0001.mat')

data = TrialData.BroadbandData;
assist = TrialData.Assist;
idx = find(assist~=0);

beta_wave = cell2mat(data');
bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',12,'HalfPowerFrequency2',26, ...
    'SampleRate',1e3);
beta_wave = filtfilt(bpFilt,beta_wave);

% plot the beta traveling wave
%
beta_wave = (beta_wave - min(beta_wave(:)))./(max(beta_wave(:))-min(beta_wave(:)));
chmap=TrialData.Params.ChMap;
figure;
for j=0.5e4:size(beta_wave,1)
    tmp=beta_wave(j,:);
    imagesc(tmp(chmap))
    caxis([0 1])
    colormap bone
    colorbar
    axis off
    textboxHandle = uicontrol('Style', 'text', 'Position', [0, 0, 200, 30]);
    txt_ip=ceil( (j-200)/200 +1);
    if sum(txt_ip ==idx) >0
        p='disengage';
    else
        p='engage';
    end
    newText = sprintf('Bin: %s %d', p, txt_ip);
    set(textboxHandle, 'String', newText);
    pause(0.000001)
end
idx


%% EXTRACTING FEATURES ACROSS TRIALS DURING ENG/DISENG AND CROSS VALIDATING



clc;clear
addpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers')
addpath('C:\Users\nikic\Documents\MATLAB')
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
cd(root_path)

foldernames={'20230830','20230901','20230913','20230915','20230920','20230922',...
    '20230927','20230928','20231004','20231006','20231011','20231013','20231018'};


% load the files
MultiTargets2D_2Stage

% extract segments for engagement and disengagement based on assist
% parameter for individual trials


% look at 





