%% IMPORTANT NOTES
% important files
% imagined_data_new_spaces: has all the training data for the new imagined
%                           spaces miming vs tongue in/out up/down
% 4PrimaryDirections_TrainData: Training data for the 4 primary directions
%                               using only online data
% OK_training_data:           Training data from imagined and online for OK
% 6DOF_Online_Data:           Online training data from 3D Arrow Task

%% TESTING OUT THE PERFORMACE OF DISTANCE ESTIMATION

clc;clear
close all

foldernames = {'20210108','20210115'};
root_path = 'E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'RobotDistance');
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        files = [files;findfiles('',filepath)'];
    end
end

% load the data for each target
D1=[];
D2=[];
D3=[];
D4=[];
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    l=length(features);
    kinax = [1:l];
    temp = cell2mat(features(kinax));
    temp = temp(129:end,:);
    temp = temp(:,end-20:end);
    if TrialData.TargetID == 1
        D1 = [D1 temp];
    elseif TrialData.TargetID == 2
        D2 = [D2 temp];
    elseif TrialData.TargetID == 3
        D3 = [D3 temp];
    elseif TrialData.TargetID == 4
        D4 = [D4 temp];
    end
end


clear condn_data
% combing both onlien plus offline
idx=641;
condn_data{1}=[D1(idx:end,:) ]';
condn_data{2}= [D2(idx:end,:)]';
condn_data{3}=[D3(idx:end,:) ]';
condn_data{4}=[D4(idx:end,:) ]';
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')

A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};


clear N
N = [A' B' C' D' ];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1)];

T = zeros(size(T1,1),4);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;

% code to train a neural network
parpool(4)
net = patternnet([256 256 256 ]) ;
net.performParam.regularization=0.2;
net = train(net,N,T','useParallel','yes');
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
genFunction(net,'MLP_DistanceEstimation')

% multiple linear regression

%% TRAINING 6DOF CLASSIFIER FOR B1 USING ONLINE 3D DATA AS WELL

clc;clear
root_path = 'E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames = {'20201218','20210108','20210115','20210128','20210201',...
    '20210212','20210219','20210226',...
    '20210305','20210312','20210319','20210402','20210326','20210409',...
    '20210416'};
cd(root_path)

files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        files = [files;findfiles('',filepath)'];
    end
end


% load the data for each target
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
% baseline stats
baseline_data_var = [];
baseline_data_mean=[];
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    kinax = TrialData.TaskState;
    if kinax(1) == 0
        kinax=kinax(2:end);
    end
    %idx = find(kinax==2);
    %idx = idx(end-3:end);
    kinax = find(kinax==3);

    %%% only last correct bins
    %l=length(kinax)+1;
    %kinax = kinax(l-TrialData.Params.ClickCounter:end);

    %%% whenever correct decision
    idx=find(TrialData.ClickerState == TrialData.TargetID);
    kinax = kinax(idx);

    if TrialData.TargetID == TrialData.SelectedTargetID
        temp = cell2mat(features(kinax));
        temp = temp(129:end,:);
        if TrialData.TargetID == 1
            D1 = [D1 temp];
        elseif TrialData.TargetID == 2
            D2 = [D2 temp];
        elseif TrialData.TargetID == 3
            D3 = [D3 temp];
        elseif TrialData.TargetID == 4
            D4 = [D4 temp];
        elseif TrialData.TargetID == 5
            D5 = [D5 temp];
        elseif TrialData.TargetID == 6
            D6 = [D6 temp];
        elseif TrialData.TargetID == 7
            D7 = [D7 temp];
        end
    end
    baseline_data_var=[baseline_data_var;sqrt(TrialData.FeatureStats.Var(129:256))];
    baseline_data_mean=[baseline_data_mean;(TrialData.FeatureStats.Mean(129:256))];
end
% % clearvars -except baseline_data_var baseline_data_mean
% save baseline_data_6DoF -v7.3
temp=unique(baseline_data_mean,'rows');
figure;plot(temp','Color',[.2 .2 .8 .6])
axis tight
xlabel('Electrode')
ylabel('Mean')
title('Baseline mean from training data')
set(gcf,'Color','w')
set(gca,'FontSize',14)
box off

clear condn_data
% combing both onlien plus offline
%idx=641;
idx = [1:128 385:512 641:768];
condn_data{1}=[D1(idx,:) ]'; % right hand
condn_data{2}= [D2(idx,:)]'; % both feet
condn_data{3}=[D3(idx,:)]'; % left hand
condn_data{4}=[D4(idx,:)]'; % head
condn_data{5}=[D5(idx,:)]'; % mime up
condn_data{6}=[D6(idx,:)]'; % tongue in
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
save 6DOF_Online_Data_3Feat condn_data -v7.3

A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};
E = condn_data{5};
F = condn_data{6};


clear N
N = [A' B' C' D' E' F'];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1)];

T = zeros(size(T1,1),4);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;
[aa bb]=find(T1==5);[aa(1) aa(end)]
T(aa(1):aa(end),5)=1;
[aa bb]=find(T1==6);[aa(1) aa(end)]
T(aa(1):aa(end),6)=1;

% code to train a neural network
clear net
net = patternnet([128 128 128 ]) ;
net.performParam.regularization=0.2;
net = train(net,N,T','UseGPU','yes');
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
genFunction(net,'MLP_6DoF_Trained4mOnlineData_3Features')
save net net

%%%%% code to train only 4 directions
A = condn_data{1};% right hand is x right
B = condn_data{6};% tongue in is y down
C = condn_data{3};% left hand is x left
D = condn_data{5};% mime up is y up

clear N
N = [A' B' C' D'];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1)];

T = zeros(size(T1,1),4);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;

% code to train a neural network
net = patternnet([256 256 256 ]) ;
net.performParam.regularization=0.1;
net = train(net,N,T');
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
genFunction(net,'MLP_4DoF_Trained4mOnlineData_3Features_20210319')


%% NEW TRAINING 6DOF CLASSIFIER FOR B1 USING ALL THE ONLINE 3D DATA

clc;clear
root_path = 'E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames = {'20201218','20210108','20210115','20210128','20210201','20210212','20210219','20210226',...
    '20210305','20210312','20210319','20210326','20210402','20210409'};
cd(root_path)

files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        files = [files;findfiles('',filepath)'];
    end
end


% load the data for each target
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
for i=1:length(files)
    disp([num2str(i) ' of ' num2str(length(files))])
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    kinax = TrialData.TaskState;
    if kinax(1) == 0
        kinax=kinax(2:end);
    end
    idx0 = find(kinax==2);
    idx = find(kinax==3);
    %idx = idx(end-TrialData.Params.ClickCounter+1:end);

    %idx = [find(kinax==2) find(kinax==3)];
    l = idx(end)-5:idx(end);
    idx = unique([idx0 l]);
    %idx=l(l>0);
    %kinax =  unique([idx0 (idx)]);
    kinax =  [(idx)];

    temp = cell2mat(features(kinax));
    temp = temp(129:end,:);
    if TrialData.SelectedTargetID == TrialData.TargetID
        if TrialData.TargetID == 1
            D1 = [D1 temp];
        elseif TrialData.TargetID == 2
            D2 = [D2 temp];
        elseif TrialData.TargetID == 3
            D3 = [D3 temp];
        elseif TrialData.TargetID == 4
            D4 = [D4 temp];
        elseif TrialData.TargetID == 5
            D5 = [D5 temp];
        elseif TrialData.TargetID == 6
            D6 = [D6 temp];
        end
    end
end

clear condn_data
% combing both onlien plus offline
%idx=641;
idx = [1:128 385:512 641:768];
condn_data{1}=[D1(idx,:) ]'; % right hand
condn_data{2}= [D2(idx,:)]'; % both feet
condn_data{3}=[D3(idx,:)]'; % left hand
condn_data{4}=[D4(idx,:)]'; % head
condn_data{5}=[D5(idx,:)]'; % mime up
condn_data{6}=[D6(idx,:)]'; % tongue in
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
save 6DOF_ALL_Online_Data_3Feat condn_data -v7.3

%% TRAINING 6DOF CLASSIFIER FOR B1 USING ONLINE "TIME BASED" 3D DATA AS WELL

clc;clear
root_path = 'E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames = {'20201218','20210108','20210115','20210128','20210201','20210212','20210219','20210226',...
    '20210305','20210312','20210319','20210402','20210326','20210409'};
cd(root_path)

files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        files = [files;findfiles('',filepath)'];
    end
end


% load the data for each target
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    kinax = TrialData.TaskState;
    if kinax(1) == 0
        kinax=kinax(2:end);
    end
    idx = find(kinax==2);
    idx = idx(end-3:end);
    kinax = find(kinax==3);
    l=length(kinax)+1;
    kinax = kinax(l-TrialData.Params.ClickCounter:end);
    %kinax =[idx kinax];

    if TrialData.TargetID == TrialData.SelectedTargetID
        temp = cell2mat(features(kinax));
        temp = temp(129:end,:);
        temp = temp(641:end,:);
        if TrialData.TargetID == 1
            D1 = cat(3,D1,temp);
            %D1 = [D1 temp];
        elseif TrialData.TargetID == 2
            D2 = cat(3,D2,temp);
        elseif TrialData.TargetID == 3
            D3 = cat(3,D3,temp);
        elseif TrialData.TargetID == 4
            D4 = cat(3,D4,temp);
        elseif TrialData.TargetID == 5
            D5 = cat(3,D5,temp);
        elseif TrialData.TargetID == 6
            D6 = cat(3,D6,temp);
        end
    end
end
clear condn_data
% combing both onlien plus offline
%idx=641;
%idx = [1:128 ];
condn_data{1}=[D1]; % right hand
condn_data{2}= [D2]; % both feet
condn_data{3}=[D3]; % left hand
condn_data{4}=[D4]; % head
condn_data{5}=[D5]; % mime up
condn_data{6}=[D6]; % tongue in
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
save 6DOF_Online_Data_hG_Temporal condn_data -v7.3

A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};
E = condn_data{5};
F = condn_data{6};


clear N
N = [A' B' C' D' E' F'];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1)];

T = zeros(size(T1,1),4);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;
[aa bb]=find(T1==5);[aa(1) aa(end)]
T(aa(1):aa(end),5)=1;
[aa bb]=find(T1==6);[aa(1) aa(end)]
T(aa(1):aa(end),6)=1;

% code to train a neural network
net = patternnet([128 128 128 ]) ;
net.performParam.regularization=0.2;
net = train(net,N,T');
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
genFunction(net,'MLP_6DoF_Trained4mOnlineData_3Features_20210319')

%%%%% code to train only 4 directions
A = condn_data{1};% right hand is x right
B = condn_data{6};% tongue in is y down
C = condn_data{3};% left hand is x left
D = condn_data{5};% mime up is y up

clear N
N = [A' B' C' D'];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1)];

T = zeros(size(T1,1),4);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;

% code to train a neural network
net = patternnet([256 256 256 ]) ;
net.performParam.regularization=0.1;
net = train(net,N,T');
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
genFunction(net,'MLP_4DoF_Trained4mOnlineData_3Features_20210319')

%% GETTING ALL THE DATA WHEN B1 WAS 'CLICKING'

clc;clear;
close all

% % saying OK
root = {'E:\DATA\ecog data\ECoG BCI\GangulyServer\CursorControlGridTask\20190807\112500\BCI_Fixed',...
    'E:\DATA\ecog data\ECoG BCI\GangulyServer\CursorControlGridTask\20190807\113225\BCI_Fixed',...
    'E:\DATA\ecog data\ECoG BCI\GangulyServer\CursorControlGridTask\20190807\140450\BCI_Fixed',...
    'E:\DATA\ecog data\ECoG BCI\GangulyServer\CursorControlGridTask\20190809\114432\BCI_Fixed',...
    'E:\DATA\ecog data\ECoG BCI\GangulyServer\CursorControlGridTask\20190809\114630\BCI_Fixed',...
    'E:\DATA\ecog data\ECoG BCI\GangulyServer\CursorControlGridTask\20190809\114836\BCI_Fixed',...
    'E:\DATA\ecog data\ECoG BCI\GangulyServer\CursorControlGridTask\20190809\141115\BCI_Fixed',...
    'E:\DATA\ecog data\ECoG BCI\GangulyServer\CursorControlGridTask\20190813\114725\BCI_Fixed',...
    'E:\DATA\ecog data\ECoG BCI\GangulyServer\CursorControlGridTask\20190813\114930\BCI_Fixed',...
    'E:\DATA\ecog data\ECoG BCI\GangulyServer\CursorControlGridTask\20190813\153221\BCI_Fixed'};


files=[];
for i=1:length(root)
    cd(root{i})
    temp = findfiles('',pwd);
    files=[files;temp'];
end


% load the data
imagine_data=[];
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.NeuralFeatures;
    kin = TrialData.CursorState;
    kinax = diff(kin(3,:));
    kinax = [find(kinax==0) size(kin,2)];
    kinax = kinax(3:end);
    if length(kinax>=1)
        temp = cell2mat(features(kinax));
        temp = temp(769:end,:);
        imagine_data = [imagine_data temp];
    end
end



%%% NEW %%%
%%% also taking into account times when B1 was within the target area
%%% trying to click %%%%%%
files1=findfiles('~kf_params',...
    'E:\DATA\ecog data\ECoG BCI\GangulyServer\CursorControlGridTask\20190816',1);
files2=findfiles('~kf_params',...
    'E:\DATA\ecog data\ECoG BCI\GangulyServer\CursorControlGridTask\20190819',1);
files3=findfiles('~kf_params',...
    'E:\DATA\ecog data\ECoG BCI\GangulyServer\CursorControlGridTask\20190830',1);
files3=findfiles('~kf_params',...
    'E:\DATA\ecog data\ECoG BCI\GangulyServer\CursorControlGridTask\20190904',1);
files4=findfiles('~kf_params',...
    'E:\DATA\ecog data\ECoG BCI\GangulyServer\CursorControlGridTask\20190906',1);

files=[files1 files2 files3]';
for i=1:length(files)
    disp(i)
    load(files{i})
    kinpos = TrialData.CursorState(1:2,:);
    bounds=TrialData.Params.ReachTargetWindows(TrialData.TargetID,:);
    boundx= bounds([1 3]);
    boundy=bounds([2 4]);
    features  = TrialData.NeuralFeatures;
    for j=1:size(kinpos,2)
        temp = cell2mat(features(j));
        temp = temp(769:end,:);
        if (boundx(1)<= kinpos(1,j) && kinpos(1,j)<=boundx(2))...
                && (boundy(1)<=kinpos(2,j) && kinpos(2,j)<=boundy(2))
            imagine_data = [imagine_data temp];
        end
    end
end

OK_data = imagine_data;
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
save Ok_training_data OK_data -v7.3

%% TRAINING OK ALONG WITH 6 DOF JUST FROM ONLINE DATA

clc;clear
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
load Ok_training_data
load 6DOF_Online_Data
% now loading all data

A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};
E = condn_data{5};
F = condn_data{6};
G = OK_data';
idx = randperm(size(G,1),350);
G = G(idx,:);
condn_data{7} = G;

clear N
N = [A' B' C' D' E' F' G'];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1)];

T = zeros(size(T1,1),7);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;
[aa bb]=find(T1==5);[aa(1) aa(end)]
T(aa(1):aa(end),5)=1;
[aa bb]=find(T1==6);[aa(1) aa(end)]
T(aa(1):aa(end),6)=1;
[aa bb]=find(T1==7);[aa(1) aa(end)]
T(aa(1):aa(end),7)=1;
figure;
imagesc(T)


% code to train a neural network
net = patternnet([256 256 256 ]) ;
net.performParam.regularization=0.2;
net = train(net,N,T');
genFunction(net,'MLP_6DoF_Trained4mOnlineData_20210212')


% build an equivalent SVM classifier
% to separate the 6 DoF plus 'OK'
model = build_SVM_multiclass(condn_data,0.7,4);

% test it out on the NN....do random data themselves have a directional
% bias?
addpath('C:\Users\Nikhlesh\Documents\GitHub\for_BCI\bci\clicker')
iter=1000;
dec=zeros(6,iter);
for i=1:iter
    d=randn(128,1);
    idx=d<0;
    d(idx)=-d(idx)+15;
    temp = MLP_6DoF_Trained4mOnlineData_20210212(d);
    [aa bb]=max(temp);
    if aa>=0.7
        dec(bb,i) = 1;
    end
end
figure;imagesc(dec)
set(gcf,'Color','w')
set(gca,'FontSize',14)
xlabel('Iterations')
yticklabels({'Right','Down','Left','up','Zup','Zdown'})
title('In built bias for the NN classifier')


figure;stem(sum(dec')/size(dec,2),'LineWidth',1)
xticklabels({'Right','Down','Left','up','Zup','Zdown'})
ylabel('Proportion')
set(gcf,'Color','w')
set(gca,'FontSize',14)
title('In built bias for the NN classifier')
hh=hline(1/6);
set(hh,'LineWidth',2)

% pick the day and session when there was a large bias present
%20210226 is the day when there were baseline issues

% plotting the average data within classification window
figure;
ha=tight_subplot(3,2);
targets={'right','down','left','up','z-up','z-down'};
for i=1:length(condn_data)
    axes(ha(i))
    temp=condn_data{i};
    m = mean(temp,1);
    mb = sort(bootstrp(10000,@mean,temp));
    %plot(m,'b');
    %plot(mb(1,'--b'))
    %plot(mb(end,'--b'))
    t=1:128;
    [fillhandle,msg]=jbfill(t,(mb(end,:)),(mb(1,:)),[0.2 0.2 0.8],[0.2 0.2 0.8],1,.6);
    axis tight
    ylim([-1 3])
    hold on
    hh=hline(0);
    set(hh,'LineWidth',2)
    title(targets{i})
    set(gca,'LineWidth',1)
    %hold on;plot(t,(median(b,1)),'k','LineWidth',1)
    if i==7
        xlabel('Electrode')
        ylabel('z-scored hG')
    end
end
set(gcf,'Color','w')

% looking at the bad day i.e. 20210226
root_path=('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210226');
% the first folder
foldernames={'113057'};

files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, 'Robot',foldernames{i},'BCI_Fixed');
    files = [files;findfiles('',folderpath)'];
end


% load the data for each target
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
% baseline stats
baseline_mean = [];
baseline_var=[];
suc=[];
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    kinax = TrialData.TaskState;
    if kinax(1) == 0
        kinax=kinax(2:end);
    end
    kinax = find(kinax==3);
    l=length(kinax)+1;
    kinax = kinax(l-TrialData.Params.ClickCounter:end);
    if TrialData.TargetID == TrialData.SelectedTargetID
        suc=[suc;TrialData.TargetID];
        temp = cell2mat(features(kinax));
        temp = temp(129:end,:);
        if TrialData.TargetID == 1
            D1 = [D1 temp];
        elseif TrialData.TargetID == 2
            D2 = [D2 temp];
        elseif TrialData.TargetID == 3
            D3 = [D3 temp];
        elseif TrialData.TargetID == 4
            D4 = [D4 temp];
        elseif TrialData.TargetID == 5
            D5 = [D5 temp];
        elseif TrialData.TargetID == 6
            D6 = [D6 temp];
        elseif TrialData.TargetID == 7
            D7 = [D7 temp];
        end
    end
    baseline_mean=[baseline_mean;(TrialData.FeatureStats.Mean(769:end))];
    baseline_var=[baseline_var;sqrt(TrialData.FeatureStats.Var(769:end))];
end

figure;subplot(2,1,1)
temp=unique(baseline_data_mean,'rows');
plot(temp','Color',[.2 .2 .8 .5])
axis tight
xlabel('Electrode')
ylabel('Mean')
set(gcf,'Color','w')
set(gca,'FontSize',14)
box off
hold on
temp=unique(baseline_mean,'rows');
hold on
plot(temp','Color','r')
title('Mean')

subplot(2,1,2)
temp=unique(baseline_data_var,'rows');
plot(temp','Color',[.2 .2 .8 .5])
axis tight
xlabel('Electrode')
ylabel('Std')
set(gcf,'Color','w')
set(gca,'FontSize',14)
box off
hold on
temp=unique(baseline_var,'rows');
hold on
plot(temp','Color','r')
title('Std')


temp=unique(baseline_var,'rows');
figure;plot(temp)
axis tight


%% TRAINING OK ALONG WITH 6 DOF JUST FROM ALL DATA

clc;clear
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
load Ok_training_data
load 6DOF_Online_Data
% now loading all data
data1=load('4PrimaryDirections_TrainData');
data2=load('imagined_data_new_spaces');

A = [condn_data{1} ;data1.condn_data{1}];
B = [condn_data{2}  ;data1.condn_data{2}];
C = [condn_data{3} ;data1.condn_data{3}];
D = [condn_data{4}  ;data1.condn_data{4}];
E = [condn_data{5};data2.data_up(641:end,:)'];
F = [condn_data{6};data2.data_tongIn(641:end,:)'];
G = OK_data';
% resize to about 1500 each
idx = randperm(size(A,1),2500);
A=A(idx,:);
idx = randperm(size(B,1),2500);
B=B(idx,:);
idx = randperm(size(C,1),2500);
C=C(idx,:);
idx = randperm(size(D,1),2500);
D=D(idx,:);
idx = randperm(size(G,1),2500);
G=G(idx,:);


clear N
N = [A' B' C' D' E' F' G'];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1)];

T = zeros(size(T1,1),7);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;
[aa bb]=find(T1==5);[aa(1) aa(end)]
T(aa(1):aa(end),5)=1;
[aa bb]=find(T1==6);[aa(1) aa(end)]
T(aa(1):aa(end),6)=1;
[aa bb]=find(T1==7);[aa(1) aa(end)]
T(aa(1):aa(end),7)=1;



% code to train a neural network
net = patternnet([256 256 256 ]) ;
net.performParam.regularization=0.2;
net = train(net,N,T');

cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
genFunction(net,'MLP_6DoF_PlusOK_Trained4mAllData_20210212')

%% DECODING OF THE OBSERVED HAND MOVEMENT
% continuous decoding model

clc;clear
filepath='E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210312\Hand\';
cd(filepath)

% hand open close folders
%folders = {'134510','135726','140825'};
%folders = {'114603','114057','114842','115052','115254','145008',...
%    };
D=dir(pwd);
folders={};k=1;
for j=3:length(D)
    folders{k} = D(j).name;
    k=k+1;
end
files=[];
for i=1:length(folders)
    fullpath = [filepath folders{i} '\BCI_Fixed\'];
    files = [files findfiles('', fullpath)];
end

% get the trials

% load the data for each target
kin_data={};
neural_data={};
kin_traj=[];
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    cursor_state = TrialData.CursorState(3,:);
    kinax = TrialData.TaskState;
    if kinax(1) == 0
        kinax=kinax(2:end);
    end
    idx = find(kinax==3);
    %idx = idx(1)-3:idx(end)+3;
    kinax = idx;
    temp = cell2mat(features(kinax));
    temp = temp(129:end,:);
    neural_data{i} = (temp);
    cursor_state = cursor_state(kinax);
    kin_data{i} = (cursor_state);
    kin_traj = [kin_traj cursor_state];
end


y = cell2mat(kin_data(3:end));
y=smooth(y);
ytest = cell2mat(kin_data(1:2));
ytest =smooth(ytest);
x = cell2mat(neural_data(3:end));
x = x(641:end,:);
xtest = cell2mat(neural_data(1:2));
xtest = xtest(641:end,:);
%
% m=mean(x');
% [coeff,score,latent]=pca(x');
% %x = (score(:,1:20)*coeff(:,1:20)')';
% %x = x+m';
% x=score(:,1:30)';
%
% xtest=xtest';
% m=mean(xtest);
% xtest = xtest  - mean(xtest);
% xtest = (xtest*coeff(:,1:30))';

%xtest = (xtest*coeff(:,1:20)*coeff(:,1:20)')';
%xtest = xtest+m';

% build the wiener filter, with 7 lags
X=[];
lag_length=7;
for i=2:length(y)
    lag_data = zeros(1,lag_length*128);
    if i<lag_length
        lags = 1:rem(i,lag_length)-1;
    else
        lags = fliplr(i-1:-1:(i-lag_length));
    end
    lags=lags(lags>0);

    k=1:128:length(lag_data)+1;
    for j=1:length(lags)
        lag_data(k(j):k(j+1)-1) = x(:,lags(j))';
    end

    X = [X;lag_data];
end

y=y(2:end);

% build the model
% y = XA

A = (X'*X + 1e-3*eye(size(X,2))) \ (X'*y);

% now test the model on held out data
X_test=[];
for i=2:length(ytest)
    lag_data = zeros(1,lag_length*128);
    if i<lag_length
        lags = 1:rem(i,lag_length)-1;
    else
        lags = fliplr(i-1:-1:(i-lag_length));
    end
    lags=lags(lags>0);

    k=1:128:length(lag_data)+1;
    for j=1:length(lags)
        lag_data(k(j):k(j+1)-1) = xtest(:,lags(j))';
    end

    X_test = [X_test;lag_data];
end
figure;
plot((smooth(X_test*A)))
hold on
plot((ytest(2:end)))
corr(ytest(2:end),smooth(X_test*A))



% using a linear classifier
data_open = neural_data(1:2:end);
data_close = neural_data(2:2:end);

% sample by sample model building
res=[];
for iter=1:25
    disp(iter)

    % take 6 random trials for testing
    test_idx_open = randperm(length(data_open),10);
    test_idx_close = randperm(length(data_close),10);

    test_open = data_open(test_idx_open);
    test_close = data_close(test_idx_close);

    idx = ones(length(data_open),1);
    idx(test_idx_open)=0;
    train_open = data_open(logical(idx));

    idx = ones(length(data_close),1);
    idx(test_idx_close)=0;
    train_close = data_close(logical(idx));

    % training the classifiers
    A1 = cell2mat(train_open);
    A = A1(1:end,:);

    B1 = cell2mat(train_close);
    B = B1(1:end,:);

    % train the model
    [res_acc,model_overall,pval] = svm_linear(A,B,1,1);
    model_overall = mean(model_overall,1);

    %figure;stem(model_overall)

    % test the model on trial by trial basis
    res_open=[];
    res_close=[];
    for j=1:length(test_open)
        tmp = cell2mat(test_open(j));
        tmp = model_overall*tmp(1:end,:);
        res_open = [res_open; sum(tmp<=0)/length(tmp)];

        tmp = cell2mat(test_close(j));
        tmp = model_overall*tmp(1:end,:);
        res_close = [res_close; sum(tmp>=0)/length(tmp)];
    end

    res=[res; sum(res_open>=0.5)  sum(res_close>=0.5)];

    %
    %     test_open =  cell2mat(test_open);
    %     test_open = test_open(1:end,:);
    %     test_res = model_overall*test_open;
    %     res(iter) = sum(test_res<=0)/length(test_res);
    %
    %     test_close =  cell2mat(test_close);
    %     test_close = test_close(1:end,:);
    %     test_res = model_overall*test_close;
    %sum(test_res>=0)/length(test_res)

end
res
median(res)


% building model based on time-course of trial structure
% the temporal evolution of hG features over a fixed time-window
res=[];
for iter=1:10
    disp(iter)

    % take 10 random trials for testing
    test_idx_open = randperm(length(data_open),10);
    test_idx_close = randperm(length(data_close),10);

    test_open = data_open(test_idx_open);
    test_close = data_close(test_idx_close);

    idx = ones(length(data_open),1);
    idx(test_idx_open)=0;
    train_open = data_open(logical(idx));

    idx = ones(length(data_close),1);
    idx(test_idx_close)=0;
    train_close = data_close(logical(idx));

    % training the classifiers on a trial by trial level using hG features
    A=[];
    B=[];
    tidx=1:16;
    for i=1:length(train_open)
        tmp = cell2mat(train_open(i));
        tmp= tmp(1:end,tidx);
        A(:,i) = tmp(:);
        tmp = cell2mat(train_close(i));
        tmp= tmp(1:end,tidx);
        B(:,i) = tmp(:);
    end

    % train the model
    [res_acc,model_overall,pval] = svm_linear(A,B,4,1);
    model_overall = mean(model_overall,1);

    %figure;stem(model_overall)

    % test the model on trial by trial basis
    res_open=[];
    res_close=[];
    for j=1:length(test_open)
        tmp = cell2mat(test_open(j));
        tmp= tmp(1:end,tidx);
        tmp = model_overall*tmp(:);
        res_open = [res_open; sum(tmp<=0)];

        tmp = cell2mat(test_close(j));
        tmp= tmp(1:end,tidx);
        tmp = model_overall*tmp(:);
        res_close = [res_close; sum(tmp>=0)];
    end

    res=[res; sum(res_open)/length(res_open)...
        sum(res_close)/length(res_close)];
end
res
median(res)

a(1,1)=mean(res(:,1));
a(1,2)=1-a(1,1);
a(2,2) = mean(res(:,2));
a(2,1)=1-a(2,2);
figure;imagesc(a)
colormap bone
caxis([0 1])
xticklabels({'mime open','mime close'})
yticklabels({'mime open','mime close'})
set(gcf,'Color','w')
set(gca,'FontSize',14)

%% discriminating hand open/close with palm open/close

clc;clear
filepath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210212\Hand\';
cd(filepath)

% palm open close folders
folders = {'134740','140460','141200'};
files=[];
for i=1:length(folders)
    fullpath = [filepath folders{i} '\BCI_Fixed\'];
    files = [files findfiles('', fullpath)];
end

% get the trials

% load the data for each target
neural_data1={};
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    kinax = TrialData.TaskState;
    if kinax(1) == 0
        kinax=kinax(2:end);
    end
    idx = find(kinax==3);
    idx = idx(1)-3:idx(end)+3;
    kinax = idx;
    temp = cell2mat(features(kinax));
    temp = temp(129:end,:);
    neural_data1{i} = (temp);
end

% hand open/close
folders = {'141658','142936','143631'};
files=[];
for i=1:length(folders)
    fullpath = [filepath folders{i} '\BCI_Fixed\'];
    files = [files findfiles('', fullpath)];
end

% get the trials

% load the data for each target
neural_data={};
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    kinax = TrialData.TaskState;
    if kinax(1) == 0
        kinax=kinax(2:end);
    end
    idx = find(kinax==3);
    idx = idx(1)-3:idx(end)+3;
    kinax = idx;
    temp = cell2mat(features(kinax));
    temp = temp(129:end,:);
    neural_data{i} = (temp);
end


% using a linear classifier
data_open = neural_data;
data_close = neural_data1;

res=[];
for iter=1:10
    disp(iter)

    % take 2 random trials for testing
    test_idx_open = randperm(length(data_open),2);
    test_idx_close = randperm(length(data_close),2);

    test_open = data_open(test_idx_open);
    test_close = data_close(test_idx_close);

    idx = ones(length(data_open),1);
    idx(test_idx_open)=0;
    train_open = data_open(logical(idx));

    idx = ones(length(data_close),1);
    idx(test_idx_close)=0;
    train_close = data_close(logical(idx));

    % training the classifiers
    A1 = cell2mat(train_open);
    A = A1(1:end,:);

    B1 = cell2mat(train_close);
    B = B1(1:end,:);

    % train the model
    [res_acc,model_overall,pval] = svm_linear(A,B,1,1);
    model_overall = mean(model_overall,1);

    %figure;stem(model_overall)

    % test the model
    test_open =  cell2mat(test_open);
    test_open = test_open(1:end,:);
    test_res = model_overall*test_open;
    temp1 = sum(test_res<=0)/length(test_res);

    test_close =  cell2mat(test_close);
    test_close = test_close(1:end,:);
    test_res = model_overall*test_close;
    temp2=sum(test_res>=0)/length(test_res);

    res(iter)=mean([temp1 temp2]);
end

mean(res)

clc;clear
filepath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210212\Hand\';
cd(filepath)

% palm open close folders
folders = {'134740','140460','141200'};
files=[];
for i=1:length(folders)
    fullpath = [filepath folders{i} '\BCI_Fixed\'];
    files = [files findfiles('', fullpath)];
end

% get the trials

% load the data for each target
neural_data1={};
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    kinax = TrialData.TaskState;
    if kinax(1) == 0
        kinax=kinax(2:end);
    end
    idx = find(kinax==3);
    idx = idx(1)-3:idx(end)+3;
    kinax = idx;
    temp = cell2mat(features(kinax));
    temp = temp(129:end,:);
    neural_data1{i} = (temp);
end

% hand open/close
folders = {'141658','142936','143631'};
files=[];
for i=1:length(folders)
    fullpath = [filepath folders{i} '\BCI_Fixed\'];
    files = [files findfiles('', fullpath)];
end

% get the trials

% load the data for each target
neural_data={};
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    kinax = TrialData.TaskState;
    if kinax(1) == 0
        kinax=kinax(2:end);
    end
    idx = find(kinax==3);
    idx = idx(1)-3:idx(end)+3;
    kinax = idx;
    temp = cell2mat(features(kinax));
    temp = temp(129:end,:);
    neural_data{i} = (temp);
end


% using a linear classifier
data_open = neural_data;
data_close = neural_data1;

res=[];
for iter=1:3
    disp(iter)

    % take 2 random trials for testing
    test_idx_open = randperm(length(data_open),2);
    test_idx_close = randperm(length(data_close),2);

    test_open = data_open(test_idx_open);
    test_close = data_close(test_idx_close);

    idx = ones(length(data_open),1);
    idx(test_idx_open)=0;
    train_open = data_open(logical(idx));

    idx = ones(length(data_close),1);
    idx(test_idx_close)=0;
    train_close = data_close(logical(idx));

    % training the classifiers
    A1 = cell2mat(train_open);
    A = A1(1:end,:);

    B1 = cell2mat(train_close);
    B = B1(1:end,:);

    % train the model
    [res_acc,model_overall,pval] = svm_linear(A,B,1,1);
    model_overall = mean(model_overall,1);

    %figure;stem(model_overall)

    % test the model
    test_open =  cell2mat(test_open);
    test_open = test_open(1:end,:);
    test_res = model_overall*test_open;
    temp1 = sum(test_res<=0)/length(test_res);

    test_close =  cell2mat(test_close);
    test_close = test_close(1:end,:);
    test_res = model_overall*test_close;
    temp2=sum(test_res>=0)/length(test_res);

    res(iter)=mean([temp1 temp2]);
end

mean(res)



%% USING A CNN TO DISCRIMINATE DATA SAMPLE BY SAMPLE

clc;clear
close all
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
%load 6DOF_Online_Data_3Feat
load 6DOF_ALL_Online_Data_3Feat
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20201030\DiscreteArrow\114043\BCI_Fixed\Data0007.mat')
chmap=TrialData.Params.ChMap;

% split the data into training and testing data
% set the ecog activity as images of the grid, 3 channels
% format is width, height, channels and number of samples

condn_data_train={};
condn_data_test={};
for i=1:length(condn_data)
    temp = condn_data{i};
    % resize into images
    tmp_resized=[];
    for j=1:128:size(temp,2)
        chdata = temp(:,j:j+127);
        chdata = (chdata')';
        % reshape as a grid
        chtemp=[];
        for k=1:size(chdata,1)
            t = chdata(k,:);
            chtemp(:,:,k) = t(chmap);
        end
        tmp_resized = cat(4,tmp_resized,chtemp);
    end
    temp = permute(tmp_resized,[1 2 4 3]);
    %temp = temp(:,:,1,:);


    % splitting the data
    l = round(0.98*size(temp,4));
    idx = randperm(size(temp,4),l);

    % setting aside training trials
    condn_data_train{i} = temp(:,:,:,idx);
    % setting aside testing trials
    I = ones(size(temp,4),1);
    I(idx)=0;
    I=logical(I);
    condn_data_test{i} = temp(:,:,:,I);
end



% getting the data ready
XTrain = [];
YTrain = [];
for i=1:length(condn_data_train)
    tmp = condn_data_train{i};
    XTrain = cat(4,XTrain,tmp);
    YTrain = [YTrain;i*ones(size(tmp,4),1)];
end
YTrain = categorical(YTrain);

XTest = [];
YTest = [];
for i=1:length(condn_data_test)
    tmp = condn_data_test{i};
    XTest = cat(4,XTest,tmp);
    YTest = [YTest;i*ones(size(tmp,4),1)];
end
YTest = categorical(YTest);

%%%%%% CNN construction %%%%%
layers = [
    imageInputLayer([8 16 3])

    convolution2dLayer(2,4,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',1)

    convolution2dLayer(2,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',1)

    convolution2dLayer(2,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',1)
    %
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,'Stride',1)

    fullyConnectedLayer(128)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(.5)

    %     fullyConnectedLayer(64)
    %     batchNormalizationLayer
    %     reluLayer
    %     dropoutLayer(.5)

    fullyConnectedLayer(6)
    softmaxLayer
    classificationLayer];

%'ValidationData',{XTest,YTest},...
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',96,...
    'ValidationFrequency',30,...
    'L2Regularization',1e-4,...
    'ValidationData',{XTest,YTest},...
    'ExecutionEnvironment','auto');

%%%%%% CNN construction %%%%%


% build the classifier
net = trainNetwork(XTrain,YTrain,layers,options);
%analyzeNetwork(net)
%classify(net,zeros(28,28,1))

save CNN_classifier_B1_16thApr net

%
% XTrain=randn(16,8,3,500);
% XTrain(:,:,:,251:end) = XTrain(:,:,:,251:end)+1;
% YTrain = categorical(([ones(250,1);2*ones(250,1)]));

%%%%%% TESTING THE CNN ABOVE ON DATA FROM 19TH MAR 2021
% test this NN classifier on data from other time-periods when B1 was
% clicking

filepath='E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210409\Robot3DArrow';
foldernames = dir(filepath);
foldernames = foldernames(3:end);


files=[];
for i=8:length(foldernames)
    folderpath = fullfile(filepath, foldernames(i).name, 'BCI_Fixed');
    files = [files;findfiles('',folderpath)'];
end


% get the predictions
pred_acc = zeros(7);
pred_acc_cnn = zeros(7);
for i=1:length(files)
    disp(i)
    load(files{i})
    chmap=TrialData.Params.ChMap;
    tid = TrialData.TargetID;
    feat = TrialData.SmoothedNeuralFeatures;
    idx = find(TrialData.TaskState==3);
    feat = cell2mat(feat(idx));

    % augment the accuracy for the neural net during testing
    pred = TrialData.ClickerState;
    for j=1:length(pred)
        if pred(j)==0
            pred(j)=7;
        end
        pred_acc(tid,pred(j)) = pred_acc(tid,pred(j))+1;
    end

    % find predictions from the CNN
    feat_idx = [129:256 513:640 769:896];
    for j=1:size(feat,2)
        chtemp = [];
        tmp = feat(feat_idx,j);
        f1 = (tmp(1:128));
        f2 = (tmp(129:256));
        f3 = (tmp(257:384));
        chtemp(:,:,1) = f1(chmap);
        chtemp(:,:,2) = f2(chmap);
        chtemp(:,:,3) = f3(chmap);
        %out  = classify(net,chtemp);

        %act = squeeze(activations(net,chtemp,20));
        act = predict(net,chtemp);
        [aa out]=max(act);
        if aa< 0.2%TrialData.Params.NeuralNetSoftMaxThresh
            out=7;
        end
        pred_acc_cnn(tid,out) = pred_acc_cnn(tid,out)+1;
    end
end

pred_acc_cnn
pred=pred_acc

for i=1:6
    pred_acc_cnn(i,:) = pred_acc_cnn(i,:)./sum(pred_acc_cnn(i,:));
    pred(i,:) = pred(i,:)./sum(pred(i,:));
end
[diag(pred_acc_cnn(1:6,1:6)) diag(pred(1:6,1:6))]
figure; imagesc(pred_acc_cnn);caxis([0 .5])

%
% [diag(pred_acc_cnn) diag(pred_acc(1:6,1:6))]
%
%
% figure;imagesc(pred_acc_cnn)
% figure;imagesc(pred_acc(1:6,1:6))


%%%%%% CNN construction %%%%%
layers = [
    imageInputLayer([8 16 3])

    convolution2dLayer(2,4,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',1)

    convolution2dLayer(2,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',1)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,'Stride',1)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,'Stride',1)

    fullyConnectedLayer(6)
    softmaxLayer
    classificationLayer];


%% USING A CNN TO DISCRIMINATE DATA USING A WINDOW OF TIME

clc;clear
close all
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
%load 6DOF_Online_Data_3Feat
load 6DOF_Online_Data_hG_Temporal
load('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20201030\DiscreteArrow\114043\BCI_Fixed\Data0007.mat')
chmap=TrialData.Params.ChMap;

% split the data into training and testing data
% set the ecog activity as images of the grid, 3 channels
% format is width, height, channels and number of samples

condn_data_train={};
condn_data_test={};
for i=1:length(condn_data)
    temp = condn_data{i};

    % resize into images
    tmp_resized=[];
    chdata = temp;

    % reshape as a grid for each trial
    chtemp=[];
    for k=1:size(temp,3)
        tmp = squeeze(temp(:,:,k));
        chtemp1=[];
        for kk=1:size(tmp,2)
            tmp1 = tmp(:,kk);
            chtemp1 =  cat(3,chtemp1,tmp1(chmap));
        end
        chtemp = cat(4,chtemp,chtemp1);
    end
    temp  = chtemp;

    % splitting the data
    l = round(0.99*size(temp,4));
    idx = randperm(size(temp,4),l);

    % setting aside training trials
    condn_data_train{i} = temp(:,:,:,idx);
    % setting aside testing trials
    I = ones(size(temp,4),1);
    I(idx)=0;
    I=logical(I);
    condn_data_test{i} = temp(:,:,:,I);
end



% getting the data ready
XTrain = [];
YTrain = [];
for i=1:length(condn_data_train)
    tmp = condn_data_train{i};
    XTrain = cat(4,XTrain,tmp);
    YTrain = [YTrain;i*ones(size(tmp,4),1)];
end
YTrain = categorical(YTrain);

XTest = [];
YTest = [];
for i=1:length(condn_data_test)
    tmp = condn_data_test{i};
    XTest = cat(4,XTest,tmp);
    YTest = [YTest;i*ones(size(tmp,4),1)];
end
YTest = categorical(YTest);

%%%%%% CNN construction %%%%%
layers = [
    imageInputLayer([8 16 5])

    convolution2dLayer(2,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(.2)
    maxPooling2dLayer(2,'Stride',1)

    convolution2dLayer(2,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(.2)
    maxPooling2dLayer(2,'Stride',1)

    convolution2dLayer(2,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(.2)
    maxPooling2dLayer(2,'Stride',1)


    %convolution2dLayer(3,32,'Padding','same')
    %     fullyConnectedLayer(64)
    %     batchNormalizationLayer
    %     reluLayer
    %     dropoutLayer(.2)
    %
    fullyConnectedLayer(128)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(.4)

    fullyConnectedLayer(32)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(.4)

    fullyConnectedLayer(6)
    softmaxLayer
    classificationLayer];

%'ValidationData',{XTest,YTest},...
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',48,...
    'ValidationFrequency',30,...
    'L2Regularization',1e-4,...
    'ValidationData',{XTest,YTest},...
    'ExecutionEnvironment','auto');

%%%%%% CNN construction %%%%%


% build the classifier
net = trainNetwork(XTrain,YTrain,layers,options);
%analyzeNetwork(net)
%classify(net,zeros(28,28,1))

save CNN_classifier_B1_Temporal net

%
% XTrain=randn(16,8,3,500);
% XTrain(:,:,:,251:end) = XTrain(:,:,:,251:end)+1;
% YTrain = categorical(([ones(250,1);2*ones(250,1)]));

%%%%%% TESTING THE CNN ABOVE ON DATA FROM 19TH MAR 2021
% test this NN classifier on data from other time-periods when B1 was
% clicking

filepath='E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210409\Robot3DArrow';
foldernames = dir(filepath);
foldernames = foldernames(3:end);


files=[];
for i=3:length(foldernames)
    folderpath = fullfile(filepath, foldernames(i).name, 'BCI_Fixed');
    files = [files;findfiles('',folderpath)'];
end


% get the predictions
pred_acc = zeros(7);
pred_acc_cnn = zeros(7);
for i=1:length(files)
    disp(i)
    load(files{i})
    chmap=TrialData.Params.ChMap;
    tid = TrialData.TargetID;
    feat = TrialData.SmoothedNeuralFeatures;
    idx = [find(TrialData.TaskState==3)];
    feat = cell2mat(feat(idx));

    % augment the accuracy for the neural net during testing
    pred = TrialData.ClickerState;
    for j=1:length(pred)
        if pred(j)==0
            pred(j)=7;
        end
        pred_acc(tid,pred(j)) = pred_acc(tid,pred(j))+1;
    end

    % find predictions from the CNN
    feat_idx = [769:896];
    pred1=[];
    for j=5:size(feat,2)
        chtemp = [];
        tmp = feat(feat_idx,j-4:j);
        chtemp=[];
        for k=1:size(tmp,2)
            f = tmp(:,k);
            chtemp(:,:,k) = f(chmap);
        end
        out  = predict(net,chtemp);
        [aa out]=max(out);pred1=[pred1 out];
        if aa< 0.1%TrialData.Params.NeuralNetSoftMaxThresh
            out=7;
        end
        pred_acc_cnn(tid,out) = pred_acc_cnn(tid,out)+1;
    end
end

pred_acc_cnn
pred=pred_acc

for i=1:6
    pred_acc_cnn(i,:) = pred_acc_cnn(i,:)./sum(pred_acc_cnn(i,:));
    pred(i,:) = pred(i,:)./sum(pred(i,:));
end
figure; imagesc(pred_acc_cnn);caxis([0 .5])

[diag(pred(1:6,1:6)) diag(pred_acc_cnn(1:6,1:6))]

%
% [diag(pred_acc_cnn) diag(pred_acc(1:6,1:6))]
%
%
% figure;imagesc(pred_acc_cnn)
% figure;imagesc(pred_acc(1:6,1:6))


%%%%%% CNN construction %%%%%
layers = [
    imageInputLayer([8 16 3])

    convolution2dLayer(2,4,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',1)

    convolution2dLayer(2,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',1)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,'Stride',1)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,'Stride',1)

    fullyConnectedLayer(6)
    softmaxLayer
    classificationLayer];

%% CODE FOR SONOMA IN RETRAINING MODEL

% run from the BCI folder

clc;clear
root_path = 'E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames = {'20201218','20210108','20210115','20210128','20210201','20210212','20210219','20210226',...
    '20210305','20210312','20210319'};
cd(root_path)
addpath(fullfile('task_helpers'))

files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        files = [files;findfiles('',filepath)'];
    end
end


% load the data for each target
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    kinax = TrialData.TaskState;
    if kinax(1) == 0
        kinax=kinax(2:end);
    end
    idx = [find(kinax==2) find(kinax==3)];
    l = idx(end)-7:idx(end);
    idx=l(l>0);
    kinax =  (idx);

    temp = cell2mat(features(kinax));
    temp = temp(129:end,:);
    if TrialData.TargetID == 1
        D1 = [D1 temp];
    elseif TrialData.TargetID == 2
        D2 = [D2 temp];
    elseif TrialData.TargetID == 3
        D3 = [D3 temp];
    elseif TrialData.TargetID == 4
        D4 = [D4 temp];
    elseif TrialData.TargetID == 5
        D5 = [D5 temp];
    elseif TrialData.TargetID == 6
        D6 = [D6 temp];
    end
end

clear condn_data
%idx=641;
idx = [1:128 385:512 641:768];
condn_data{1}=[D1(idx,:) ]'; % right hand
condn_data{2}= [D2(idx,:)]'; % both feet
condn_data{3}=[D3(idx,:)]'; % left hand
condn_data{4}=[D4(idx,:)]'; % head
condn_data{5}=[D5(idx,:)]'; % mime up
condn_data{6}=[D6(idx,:)]'; % tongue in


chmap=TrialData.Params.ChMap;

% split the data into training and testing data
% set the ecog activity as images of the grid, 3 channels
% format is width, height, channels and number of samples

condn_data_train={};
condn_data_test={};
for i=1:length(condn_data)
    temp = condn_data{i};
    % resize into images
    tmp_resized=[];
    for j=1:128:size(temp,2)
        chdata = temp(:,j:j+127);
        chdata = (chdata')';
        % reshape as a grid
        chtemp=[];
        for k=1:size(chdata,1)
            t = chdata(k,:);
            chtemp(:,:,k) = t(chmap);
        end
        tmp_resized = cat(4,tmp_resized,chtemp);
    end
    temp = permute(tmp_resized,[1 2 4 3]);


    % splitting the data
    l = round(0.95*size(temp,4));
    idx = randperm(size(temp,4),l);

    % setting aside training trials
    condn_data_train{i} = temp(:,:,:,idx);
    % setting aside testing trials
    I = ones(size(temp,4),1);
    I(idx)=0;
    I=logical(I);
    condn_data_test{i} = temp(:,:,:,I);
end



% getting the data ready
XTrain = [];
YTrain = [];
for i=1:length(condn_data_train)
    tmp = condn_data_train{i};
    XTrain = cat(4,XTrain,tmp);
    YTrain = [YTrain;i*ones(size(tmp,4),1)];
end
YTrain = categorical(YTrain);

XTest = [];
YTest = [];
for i=1:length(condn_data_test)
    tmp = condn_data_test{i};
    XTest = cat(4,XTest,tmp);
    YTest = [YTest;i*ones(size(tmp,4),1)];
end
YTest = categorical(YTest);

%%%%%% CNN construction %%%%%
layers = [
    imageInputLayer([8 16 3])

    convolution2dLayer(2,4,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',1)

    convolution2dLayer(2,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',1)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',1)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',1)

    fullyConnectedLayer(6)
    softmaxLayer
    classificationLayer];

%'ValidationData',{XTest,YTest},...
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',64,...
    'ValidationFrequency',30,...
    'L2Regularization',1e-4,...
    'ValidationData',{XTest,YTest},...
    'ExecutionEnvironment','auto');

%%%%%% CNN construction %%%%%


% build the classifier
net = trainNetwork(XTrain,YTrain,layers,options);
% save this in the clicker folder
save CNN_classifier net


%% TESTING OUT THE PERFORMANCE OF THE TASK FOR B1 IN TERMS OF THE TARGET SELECTED
% using the number of consective bins, with a trained NN



clear;clc
filepath = 'E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210402\Robot3DArrow\';
cd(filepath)
d=dir(pwd);
folders = d(3:end);

files=[];
for i=1:length(folders)
    fullpath = [filepath folders(i).name '\BCI_Fixed\'];
    files = [files findfiles('', fullpath)];
end
load('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\CNN_classifier.mat')
acc=zeros(6);
time2target=zeros(1,6);
target_count=zeros(1,6);
time2target_trial=struct;
time2target_trial.T1=[];
time2target_trial.T2=[];
time2target_trial.T3=[];
time2target_trial.T4=[];
time2target_trial.T5=[];
time2target_trial.T6=[];

for iter=1:length(files)
    disp(iter)
    clear TrialData
    load(files{iter})
    count_thresh =  TrialData.Params.ClickCounter;
    decodes = TrialData.ClickerState;

    %%%% using the conv net
    decodes = [];
    chtemp=[];
    chmap=TrialData.Params.ChMap;
    idx = find(TrialData.TaskState==3);
    for i=1:length(idx)
        X = TrialData.SmoothedNeuralFeatures{idx(i)};
        X = X(:);
        feat_idx = [129:256 513:640 769:896];
        X = X(feat_idx);
        f1 = (X(1:128));
        f2 = (X(129:256));
        f3 = (X(257:384));
        chtemp(:,:,1) = f1(chmap);
        chtemp(:,:,2) = f2(chmap);
        chtemp(:,:,3) = f3(chmap);
        act = predict(net,chtemp);
        [aa bb]=max(act);
        decodes = [decodes bb];
    end
    %%%%

    counter=0;
    for i=2:length(decodes)
        if decodes(i) == decodes(i-1)
            counter = counter + 1;
        else
            counter=0;
        end
        if counter == count_thresh-1
            acc(TrialData.TargetID,decodes(i)) = acc(TrialData.TargetID,decodes(i))+1;
            if decodes(i) == TrialData.TargetID
                time2target= (i - ...
                    TrialData.Params.ClickCounter)*(1/TrialData.Params.UpdateRate);
                switch TrialData.TargetID
                    case 1
                        time2target_trial.T1=[time2target_trial.T1 time2target];
                    case 2
                        time2target_trial.T2=[time2target_trial.T2 time2target];
                    case 3
                        time2target_trial.T3=[time2target_trial.T3 time2target];
                    case 4
                        time2target_trial.T4=[time2target_trial.T4 time2target];
                    case 5
                        time2target_trial.T5=[time2target_trial.T5 time2target];
                    case 6
                        time2target_trial.T6=[time2target_trial.T6 time2target];
                end
            end
            break
        end
    end
end

% plotting accuracy
for i =1:length(acc)
    acc(i,:)  = acc(i,:)./sum(acc(i,:));
end
acc
figure;imagesc(acc)
colormap bone
caxis([0 .6])
xticklabels({'Right hand','Both feet','Left hand','Head','Mime up','Tongue in'})
yticklabels({'Right hand','Both feet','Left hand','Head','Mime up','Tongue in'})
set(gcf,'Color','w')
title('Trial level acc')

% plotting time to target
t1 = time2target_trial.T1;
t1b = sort(bootstrp(1000,@mean,t1));
t2 = time2target_trial.T2;
if length(t2)>1
    t2b = sort(bootstrp(1000,@mean,t2));
else
    t2b = t2*ones(1000,1);
end
t3 = time2target_trial.T3;
t3b = sort(bootstrp(1000,@mean,t3));
t4 = time2target_trial.T4;
t4b = sort(bootstrp(1000,@mean,t4));
t5 = time2target_trial.T5;
t5b = sort(bootstrp(1000,@mean,t5));
t6 = time2target_trial.T6;
t6b = sort(bootstrp(1000,@mean,t6));

y=[mean(t1) mean(t2) mean(t3) mean(t4) mean(t5) mean(t6)];
x=[1 2 3 4 5 6];
neg = [mean(t1)-t1b(25) mean(t2)-t2b(25) mean(t3)-t3b(25) mean(t4)-t4b(25)...
    mean(t5)-t5b(25) mean(t6)-t6b(25)];
pos = [t1b(975)-mean(t1) t2b(975)-mean(t2) t3b(975)-mean(t3) t4b(975)-mean(t4)...
    t5b(975)-mean(t5) t6b(975)-mean(t6)];
err(:,1)=neg;
err(:,2)=pos;
figure;
barwitherr(err,y')
xticklabels({'Right hand','Both feet','Left hand','Head','Mime up','Tongue in'})
ylabel('Time taken excluding hold (s)')
box off
set(gcf,'Color','w')
set(gca,'FontSize',14)
ylim([0 3])


%
%
%
% X = TrialData.NeuralFeatures;
% if TrialData.SelectedTargetID == TrialData.TargetID
%     acc(TrialData.TargetID) = acc(TrialData.TargetID)+1;
%     target_count(TrialData.TargetID) = target_count(TrialData.TargetID)+1;
% else
%     target_count(TrialData.TargetID) = target_count(TrialData.TargetID)+1;
% end
% time2target = (length(TrialData.ClickerState) - ...
%     TrialData.Params.ClickCounter)*(1/TrialData.Params.UpdateRate);
% id = TrialData.TargetID;
% if TrialData.SelectedTargetID == TrialData.TargetID
%     switch id
%         case 1
%             time2target_trial.T1=[time2target_trial.T1 time2target];
%         case 2
%             time2target_trial.T2=[time2target_trial.T2 time2target];
%         case 3
%             time2target_trial.T3=[time2target_trial.T3 time2target];
%         case 4
%             time2target_trial.T4=[time2target_trial.T4 time2target];
%     end
% end


%% TESTING OUT THE PERFORMANCE OF THE TASK FOR B1 IN TERMS OF THE TARGET SELECTED
% using the number of consective bins from data that was collected



clear;clc
filepath = 'E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210409\Robot3DArrow\';
cd(filepath)
d=dir(pwd);
folders = d(3:end);

files=[];
for i=[10 11 length(folders)-1]
    fullpath = [filepath folders(i).name '\BCI_Fixed\'];
    files = [files findfiles('', fullpath)];
end
load('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\CNN_classifier.mat')
acc=zeros(6);
time2target=zeros(1,6);
target_count=zeros(1,6);
time2target_trial=struct;
time2target_trial.T1=[];
time2target_trial.T2=[];
time2target_trial.T3=[];
time2target_trial.T4=[];
time2target_trial.T5=[];
time2target_trial.T6=[];

for iter=1:length(files)
    disp(iter)
    clear TrialData
    load(files{iter})
    count_thresh =  TrialData.Params.ClickCounter;
    decodes = TrialData.ClickerState;

    %     %%%% using the conv net
    %     decodes = [];
    %     chtemp=[];
    %     chmap=TrialData.Params.ChMap;
    %     idx = find(TrialData.TaskState==3);
    %     for i=1:length(idx)
    %         X = TrialData.SmoothedNeuralFeatures{idx(i)};
    %         X = X(:);
    %         feat_idx = [129:256 513:640 769:896];
    %         X = X(feat_idx);
    %         f1 = (X(1:128));
    %         f2 = (X(129:256));
    %         f3 = (X(257:384));
    %         chtemp(:,:,1) = f1(chmap);
    %         chtemp(:,:,2) = f2(chmap);
    %         chtemp(:,:,3) = f3(chmap);
    %         act = predict(net,chtemp);
    %         [aa bb]=max(act);
    %         decodes = [decodes bb];
    %     end
    %     %%%%

    counter=0;
    for i=2:length(decodes)
        if (decodes(i) == decodes(i-1)) && (decodes(i)>0)
            counter = counter + 1;
        else
            counter=0;
        end
        if counter == count_thresh-1
            acc(TrialData.TargetID,decodes(i)) = acc(TrialData.TargetID,decodes(i))+1;
            if decodes(i) == TrialData.TargetID
                time2target= (i - ...
                    TrialData.Params.ClickCounter)*(1/TrialData.Params.UpdateRate);
                switch TrialData.TargetID
                    case 1
                        time2target_trial.T1=[time2target_trial.T1 time2target];
                    case 2
                        time2target_trial.T2=[time2target_trial.T2 time2target];
                    case 3
                        time2target_trial.T3=[time2target_trial.T3 time2target];
                    case 4
                        time2target_trial.T4=[time2target_trial.T4 time2target];
                    case 5
                        time2target_trial.T5=[time2target_trial.T5 time2target];
                    case 6
                        time2target_trial.T6=[time2target_trial.T6 time2target];
                end
            end
            break
        end
    end
end

% plotting accuracy
for i =1:length(acc)
    acc(i,:)  = acc(i,:)./sum(acc(i,:));
end
acc
figure;imagesc(acc)
colormap bone
caxis([0 .6])
xticklabels({'Right hand','Both feet','Left hand','Head','Mime up','Tongue in'})
yticklabels({'Right hand','Both feet','Left hand','Head','Mime up','Tongue in'})
set(gcf,'Color','w')
title('Trial level acc')

% plotting time to target
t1 = time2target_trial.T1;
t1b = sort(bootstrp(1000,@mean,t1));
t2 = time2target_trial.T2;
if length(t2)>1
    t2b = sort(bootstrp(1000,@mean,t2));
else
    t2b = t2*ones(1000,1);
end
t3 = time2target_trial.T3;
t3b = sort(bootstrp(1000,@mean,t3));
t4 = time2target_trial.T4;
t4b = sort(bootstrp(1000,@mean,t4));
t5 = time2target_trial.T5;
t5b = sort(bootstrp(1000,@mean,t5));
t6 = time2target_trial.T6;
t6b = sort(bootstrp(1000,@mean,t6));

y=[mean(t1) mean(t2) mean(t3) mean(t4) mean(t5) mean(t6)];
x=[1 2 3 4 5 6];
neg = [mean(t1)-t1b(25) mean(t2)-t2b(25) mean(t3)-t3b(25) mean(t4)-t4b(25)...
    mean(t5)-t5b(25) mean(t6)-t6b(25)];
pos = [t1b(975)-mean(t1) t2b(975)-mean(t2) t3b(975)-mean(t3) t4b(975)-mean(t4)...
    t5b(975)-mean(t5) t6b(975)-mean(t6)];
err(:,1)=neg;
err(:,2)=pos;
figure;
barwitherr(err,y')
xticklabels({'Right hand','Both feet','Left hand','Head','Mime up','Tongue in'})
ylabel('Time taken excluding hold (s)')
box off
set(gcf,'Color','w')
set(gca,'FontSize',14)
ylim([0 3])


%% TRAINING A DECODER USING AN ADPATIVE BASELINE AND COMPARING IT DATA
% train a decoder with re-baseline to the state 1 time period and see how
% performance changes on  held out data


clc;clear
root_path = 'E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames = {'20210430'};
cd(root_path)

files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'DiscreteArrow');
    D=dir(folderpath);
    for j=6:7
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed')
        files = [files;findfiles('',filepath)'];
    end
end


% load the data for each target
D1=[];
D2=[];
D3=[];
D4=[];
m=[];
s=[];
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    kinax = TrialData.TaskState;
    if kinax(1) == 0
        kinax=kinax(2:end);
    end
    idx = [find(kinax==3)];
    idx_bl = find(kinax==1);
    l = idx(end)-10:idx(end);
    idx=l(l>0);
    kinax =  (idx);

    % getting the data
    temp = cell2mat(features(kinax));
    temp = temp(129:end,:);


    %baselining
    temp_bl = cell2mat(features(idx_bl));
    temp_bl = temp_bl(129:end,:);
    m = [m temp_bl];
    %s = [s std(temp_bl')'];
    %temp = (temp-m)./s;

    % baselining hG, delta and beta individually
    %[1:128 385:512 641:768];
    for j=1:size(temp,2)
        % delta
        m1  = mean(temp(1:128,j));
        s1  = std(temp(1:128,j));
        temp(1:128,j) = (temp(1:128,j)-m1)./s1;

        % beta
        idx=385:512;
        m1  = mean(temp(idx,j));
        s1  = std(temp(idx,j));
        temp(idx,j) = (temp(idx,j)-m1)./s1;

        % hG
        idx=641:768;
        m1  = mean(temp(idx,j));
        s1  = std(temp(idx,j));
        temp(idx,j) = (temp(idx,j)-m1)./s1;
    end

    if TrialData.TargetID == 1
        D1 = [D1 temp];
    elseif TrialData.TargetID == 2
        D2 = [D2 temp];
    elseif TrialData.TargetID == 3
        D3 = [D3 temp];
    elseif TrialData.TargetID == 4
        D4 = [D4 temp];
    end
end
%
% m1 = mean(m,2);
% s1 = std(m')';
% D1 = (D1-m1)./s1;
% D2 = (D2-m1)./s1;
% D3 = (D3-m1)./s1;
% D4 = (D4-m1)./s1;

clear condn_data
%idx=641;
idx = [1:128 385:512 641:768];
condn_data{1}=[D1(idx,:) ]';
condn_data{2}= [D2(idx,:)]';
condn_data{3}=[D3(idx,:)]';
condn_data{4}=[D4(idx,:)]';



clear N
A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};
N = [A' B' C' D' ];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    ];

T = zeros(size(T1,1),4);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;

net = patternnet([128 128 128 ]) ;
net.performParam.regularization=0.3;
net = train(net,N,T','useGPU','yes');
%cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
%genFunction(net,'ReBaseline_testing')


%%%% VALIDATING THE RE-BASELINE
% test the model on held out data
files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'DiscreteArrow');
    D=dir(folderpath);
    for j=8
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed')
        files = [files;findfiles('',filepath)'];
    end
end

% get the predictions
pred_acc = zeros(5);
pred_acc_cnn = zeros(5);
for i=1:length(files)
    disp(i)
    load(files{i})
    tid = TrialData.TargetID;
    feat = TrialData.SmoothedNeuralFeatures;
    idx = find(TrialData.TaskState==3);
    feat = cell2mat(feat(idx));
    feat = feat(129:end,:);

    % baseline statistics
    idx_bl = find(TrialData.TaskState==1);
    temp_bl = cell2mat(features(idx_bl));
    temp_bl = temp_bl(129:end,:);
    m2 = mean(temp_bl,2);
    s2 = std(temp_bl')';

    % baseline the data
    %feat = (feat-m1)./s1;

    % z-score the data across channels
    for j=1:size(feat,2)
        % delta
        m1  = mean(feat(1:128,j));
        s1  = std(feat(1:128,j));
        feat(1:128,j) = (feat(1:128,j)-m1)./s1;

        % beta
        idx=385:512;
        m1  = mean(feat(idx,j));
        s1  = std(feat(idx,j));
        feat(idx,j) = (feat(idx,j)-m1)./s1;

        % hG
        idx=641:768;
        m1  = mean(feat(idx,j));
        s1  = std(feat(idx,j));
        feat(idx,j) = (feat(idx,j)-m1)./s1;
    end


    % augment the accuracy for the neural net during testing
    pred = TrialData.ClickerState;
    for j=1:length(pred)
        if pred(j)==0
            pred(j)=5;
        end
        pred_acc(tid,pred(j)) = pred_acc(tid,pred(j))+1;
    end

    % find predictions from the CNN
    feat_idx = [1:128 385:512 641:768];
    for j=1:size(feat,2)
        tmp = feat(feat_idx,j);

        % predict
        act = net(tmp);
        [aa out]=max(act);
        if aa< TrialData.Params.NeuralNetSoftMaxThresh
            out=5;
        end
        pred_acc_cnn(tid,out) = pred_acc_cnn(tid,out)+1;
    end
end

pred_acc_cnn
pred=pred_acc

for i=1:5
    pred_acc_cnn(i,:) = pred_acc_cnn(i,:)./sum(pred_acc_cnn(i,:));
    pred(i,:) = pred(i,:)./sum(pred(i,:));
end
[diag(pred_acc_cnn(1:5,1:5)) diag(pred(1:5,1:5))]
%figure; imagesc(pred_acc_cnn);caxis([0 .5])

%% USING A TEMPORAL DECODER


clc;clear
root_path = 'E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames = {'20210519'};
cd(root_path)

files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=3:6
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed')
        if isempty(dir(filepath))
            filepath = fullfile(folderpath,D(j).name,'Imagined')
        end
        files = [files;findfiles('',filepath)'];
    end
end


% load the data for each target
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
m=[];
s=[];
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    kinax = TrialData.TaskState;
    if kinax(1) == 0
        kinax=kinax(2:end);
    end
    idx = [find(kinax==3)];
    idx_bl = find(kinax==1);
    %l = idx(end)-10:idx(end);
    %idx=l(l>0);
    kinax =  (idx);

    % getting the data
    temp = cell2mat(features(kinax));
    tidx = 1:5:41;
    tmp=[];temp1=[];
    for j=1:length(tidx)-1
        tmp = temp(129:end,tidx(j):tidx(j+1)-1);
        temp1 = cat(3,temp1,tmp);
    end
    temp = temp1;


    %baselining
    %     temp_bl = cell2mat(features(idx_bl));
    %     temp_bl = temp_bl(129:end,:);
    %     m = [m temp_bl];
    %s = [s std(temp_bl')'];
    %     m = mean(temp_bl,2);
    %     s = std(temp_bl')';
    %     temp = (temp-m)./s;

    % baselining hG, delta and beta individually
    %     %[1:128 385:512 641:768];
    %     for j=1:size(temp,2)
    %        % delta
    %        m1  = mean(temp(1:128,j));
    %        s1  = std(temp(1:128,j));
    %        temp(1:128,j) = (temp(1:128,j)-m1)./s1;
    %
    %        % beta
    %        idx=385:512;
    %        m1  = mean(temp(idx,j));
    %        s1  = std(temp(idx,j));
    %        temp(idx,j) = (temp(idx,j)-m1)./s1;
    %
    %        % hG
    %        idx=641:768;
    %        m1  = mean(temp(idx,j));
    %        s1  = std(temp(idx,j));
    %        temp(idx,j) = (temp(idx,j)-m1)./s1;
    %     end

    if TrialData.TargetID == 1
        D1 = cat(3,D1,temp);
    elseif TrialData.TargetID == 2
        D2 = cat(3,D2,temp);
    elseif TrialData.TargetID == 3
        D3 = cat(3,D3,temp);
    elseif TrialData.TargetID == 4
        D4 = cat(3,D4,temp);
    elseif TrialData.TargetID == 5
        D5 = cat(3,D5,temp);
    elseif TrialData.TargetID == 6
        D6 = cat(3,D6,temp);
    elseif TrialData.TargetID == 7
        D7 = cat(3,D7,temp);
    end

end
%
% m1 = mean(m,2);
% s1 = std(m')';
% D1 = (D1-m1)./s1;
% D2 = (D2-m1)./s1;
% D3 = (D3-m1)./s1;
% D4 = (D4-m1)./s1;
% D5 = (D5-m1)./s1;
% D6 = (D6-m1)./s1;
% D7 = (D7-m1)./s1;

clear condn_data
%idx=641;
idx = [1:128 385:512 641:768];
D1 = D1(idx,:,:);
D1 = permute(D1,[3 1 2]);
D1=D1(:,:);D1=D1';
condn_data{1}=[D1]';

D2 = D2(idx,:,:);
D2 = permute(D2,[3 1 2]);
D2=D2(:,:);D2=D2';
condn_data{2}= [D2]';

D3 = D3(idx,:,:);
D3 = permute(D3,[3 1 2]);
D3=D3(:,:);D3=D3';
condn_data{3}=[D3]';

D4 = D4(idx,:,:);
D4 = permute(D4,[3 1 2]);
D4=D4(:,:);D4=D4';
condn_data{4}=[D4]';

D5 = D5(idx,:,:);
D5 = permute(D5,[3 1 2]);
D5=D5(:,:);D5=D5';
condn_data{5}=[D5]';

D6 = D6(idx,:,:);
D6 = permute(D6,[3 1 2]);
D6=D6(:,:);D6=D6';
condn_data{6}=[D6]';

D7 = D7(idx,:,:);
D7 = permute(D7,[3 1 2]);
D7=D7(:,:);D7=D7';
condn_data{7}=[D7]';

clear N
A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};
E = condn_data{5};
F = condn_data{6};
G = condn_data{7};
N = [A' B' C' D' E' F' G' ];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1)];

T = zeros(size(T1,1),7);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;
[aa bb]=find(T1==5);[aa(1) aa(end)]
T(aa(1):aa(end),5)=1;
[aa bb]=find(T1==6);[aa(1) aa(end)]
T(aa(1):aa(end),6)=1;
[aa bb]=find(T1==7);[aa(1) aa(end)]
T(aa(1):aa(end),7)=1;


net = patternnet([512]) ;
net.performParam.regularization=0.2;
net = train(net,N,T','useGPU','yes');
[model,Conf_Matrix_Overall]=build_SVM_multiclass(condn_data(1:7),0.9,5);
Conf_Matrix = squeeze(mean(Conf_Matrix_Overall,1))
%cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
%genFunction(net,'ReBaseline_testing')


%%%% VALIDATING THE RE-BASELINE
% test the model on held out data
files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=7
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed')
        files = [files;findfiles('',filepath)'];
    end
end

% get the predictions
pred_acc = zeros(8);
pred_acc_cnn = zeros(8);
for i=1:length(files)
    disp(i)
    load(files{i})
    tid = TrialData.TargetID;
    feat = TrialData.SmoothedNeuralFeatures;
    idx = find(TrialData.TaskState==3);
    feat = cell2mat(feat(idx));
    feat = feat(129:end,:);

    %     % baseline statistics
    %     idx_bl = find(TrialData.TaskState==1);
    %     temp_bl = cell2mat(features(idx_bl));
    %     temp_bl = temp_bl(129:end,:);
    %     m2 = mean(temp_bl,2);
    %     s2 = std(temp_bl')';
    %
    %     % baseline the data
    %     feat = (feat-m1)./s1;

    %     % z-score the data across channels
    %     for j=1:size(feat,2)
    %         delta
    %         m1  = mean(feat(1:128,j));
    %         s1  = std(feat(1:128,j));
    %         feat(1:128,j) = (feat(1:128,j)-m1)./s1;
    %
    %         beta
    %         idx=385:512;
    %         m1  = mean(feat(idx,j));
    %         s1  = std(feat(idx,j));
    %         feat(idx,j) = (feat(idx,j)-m1)./s1;
    %
    %         hG
    %         idx=641:768;
    %         m1  = mean(feat(idx,j));
    %         s1  = std(feat(idx,j));
    %         feat(idx,j) = (feat(idx,j)-m1)./s1;
    %     end


    % augment the accuracy for the neural net during testing
    pred = TrialData.ClickerState;
    for j=1:length(pred)
        if pred(j)==0
            pred(j)=8;
        end
        pred_acc(tid,pred(j)) = pred_acc(tid,pred(j))+1;
    end

    % get the prediction from the SVM
    % have a rolling buffer of the last 5-6 samples of the smoothed neural
    % features and then run a svm classifier on that.
    temp = cell2mat(features(kinax));
    tidx = 1:5:41;
    tmp=[];temp1=[];
    for j=1:length(tidx)-1
        tmp = temp(129:end,tidx(j):tidx(j+1)-1);
        temp1 = cat(3,temp1,tmp);
    end
    temp = temp1;


    %     % find predictions from the CNN
    %     feat_idx = [1:128 385:512 641:768];
    %     for j=1:size(feat,2)
    %         tmp = feat(feat_idx,j);
    %
    %          % predict
    %         act = net(tmp);
    %         [aa out]=max(act);
    %         if aa< TrialData.Params.NeuralNetSoftMaxThresh
    %             out=5;
    %         end
    %         pred_acc_cnn(tid,out) = pred_acc_cnn(tid,out)+1;
    %     end
end

pred_acc_cnn
pred=pred_acc

for i=1:8
    pred_acc_cnn(i,:) = pred_acc_cnn(i,:)./sum(pred_acc_cnn(i,:));
    pred(i,:) = pred(i,:)./sum(pred(i,:));
end
[diag(pred_acc_cnn(1:end,1:end)) diag(pred(1:end,1:end))]



%%%% CHECKING THIS EFFECT OF ADAPTIVE BASELING FOR 5/19/2020 DATA
% here, train model on data from am session using the adaptive baseline
% method and then compare to the output from afternoon session


clc;clear
root_path = 'E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames = {'20210519'};
cd(root_path)

files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=3:13
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed')
        if isempty(dir(filepath))
            filepath = fullfile(folderpath,D(j).name,'Imagined')
        end
        files = [files;findfiles('',filepath)'];
    end
end


% load the data for each target
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
m=[];
s=[];
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    kinax = TrialData.TaskState;
    if kinax(1) == 0
        kinax=kinax(2:end);
    end
    idx = [find(kinax==3)];
    idx_bl = find(kinax==1);
    %l = idx(end)-10:idx(end);
    %idx=l(l>0);
    kinax =  (idx);

    % getting the data
    temp = cell2mat(features(kinax));
    temp = temp(129:end,:);


    %baselining
    temp_bl = cell2mat(features(idx_bl));
    temp_bl = temp_bl(129:end,:);
    m = [m temp_bl];
    %s = [s std(temp_bl')'];
    %     m = mean(temp_bl,2);
    %     s = std(temp_bl')';
    %     temp = (temp-m)./s;

    % baselining hG, delta and beta individually
    %     %[1:128 385:512 641:768];
    %     for j=1:size(temp,2)
    %        % delta
    %        m1  = mean(temp(1:128,j));
    %        s1  = std(temp(1:128,j));
    %        temp(1:128,j) = (temp(1:128,j)-m1)./s1;
    %
    %        % beta
    %        idx=385:512;
    %        m1  = mean(temp(idx,j));
    %        s1  = std(temp(idx,j));
    %        temp(idx,j) = (temp(idx,j)-m1)./s1;
    %
    %        % hG
    %        idx=641:768;
    %        m1  = mean(temp(idx,j));
    %        s1  = std(temp(idx,j));
    %        temp(idx,j) = (temp(idx,j)-m1)./s1;
    %     end

    if TrialData.TargetID == 1
        D1 = [D1 temp];
    elseif TrialData.TargetID == 2
        D2 = [D2 temp];
    elseif TrialData.TargetID == 3
        D3 = [D3 temp];
    elseif TrialData.TargetID == 4
        D4 = [D4 temp];
    elseif TrialData.TargetID == 5
        D5 = [D5 temp];
    elseif TrialData.TargetID == 6
        D6 = [D6 temp];
    elseif TrialData.TargetID == 7
        D7 = [D7 temp];
    end

end

m1 = mean(m,2);
s1 = std(m')';
D1 = (D1-m1)./s1;
D2 = (D2-m1)./s1;
D3 = (D3-m1)./s1;
D4 = (D4-m1)./s1;
D5 = (D5-m1)./s1;
D6 = (D6-m1)./s1;
D7 = (D7-m1)./s1;

clear condn_data
%idx=641;
idx = [1:128 385:512 641:768];
condn_data{1}=[D1(idx,:) ]';
condn_data{2}= [D2(idx,:)]';
condn_data{3}=[D3(idx,:)]';
condn_data{4}=[D4(idx,:)]';
condn_data{5}=[D5(idx,:)]';
condn_data{6}=[D6(idx,:)]';
condn_data{7}=[D7(idx,:)]';



clear N
A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};
E = condn_data{5};
F = condn_data{6};
G = condn_data{7};
N = [A' B' C' D' E' F' G' ];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1)];

T = zeros(size(T1,1),7);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;
[aa bb]=find(T1==5);[aa(1) aa(end)]
T(aa(1):aa(end),5)=1;
[aa bb]=find(T1==6);[aa(1) aa(end)]
T(aa(1):aa(end),6)=1;
[aa bb]=find(T1==7);[aa(1) aa(end)]
T(aa(1):aa(end),7)=1;


net = patternnet([128 128 128 ]) ;
net.performParam.regularization=0.3;
net = train(net,N,T','useGPU','yes');
%cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
%genFunction(net,'ReBaseline_testing')


%%%% VALIDATING THE RE-BASELINE
% test the model on held out data
files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=14:16
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed')
        files = [files;findfiles('',filepath)'];
    end
end

% get the predictions
pred_acc = zeros(8);
pred_acc_cnn = zeros(8);
for i=1:length(files)
    disp(i)
    load(files{i})
    tid = TrialData.TargetID;
    feat = TrialData.SmoothedNeuralFeatures;
    idx = find(TrialData.TaskState==3);
    feat = cell2mat(feat(idx));
    feat = feat(129:end,:);

    % baseline statistics
    idx_bl = find(TrialData.TaskState==1);
    temp_bl = cell2mat(features(idx_bl));
    temp_bl = temp_bl(129:end,:);
    m2 = mean(temp_bl,2);
    s2 = std(temp_bl')';

    % baseline the data
    feat = (feat-m1)./s1;

    %     % z-score the data across channels
    %     for j=1:size(feat,2)
    %         delta
    %         m1  = mean(feat(1:128,j));
    %         s1  = std(feat(1:128,j));
    %         feat(1:128,j) = (feat(1:128,j)-m1)./s1;
    %
    %         beta
    %         idx=385:512;
    %         m1  = mean(feat(idx,j));
    %         s1  = std(feat(idx,j));
    %         feat(idx,j) = (feat(idx,j)-m1)./s1;
    %
    %         hG
    %         idx=641:768;
    %         m1  = mean(feat(idx,j));
    %         s1  = std(feat(idx,j));
    %         feat(idx,j) = (feat(idx,j)-m1)./s1;
    %     end


    % augment the accuracy for the neural net during testing
    pred = TrialData.ClickerState;
    for j=1:length(pred)
        if pred(j)==0
            pred(j)=8;
        end
        pred_acc(tid,pred(j)) = pred_acc(tid,pred(j))+1;
    end

    % find predictions from the CNN
    feat_idx = [1:128 385:512 641:768];
    for j=1:size(feat,2)
        tmp = feat(feat_idx,j);

        % predict
        act = net(tmp);
        [aa out]=max(act);
        if aa< TrialData.Params.NeuralNetSoftMaxThresh
            out=5;
        end
        pred_acc_cnn(tid,out) = pred_acc_cnn(tid,out)+1;
    end
end

pred_acc_cnn
pred=pred_acc

for i=1:8
    pred_acc_cnn(i,:) = pred_acc_cnn(i,:)./sum(pred_acc_cnn(i,:));
    pred(i,:) = pred(i,:)./sum(pred(i,:));
end
[diag(pred_acc_cnn(1:end,1:end)) diag(pred(1:end,1:end))]


%% MAKING MOVIES OF THE GRID TASK

clc;clear;
close all
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20201218\')
filepath = fullfile(pwd, 'DiscreteArrow');
D = dir(filepath);

files=[];
for i=3:length(D)
    folder_name = fullfile(filepath,D(i).name, 'BCI_fixed\')
    files=[files; findfiles('', folder_name)'];
end

%
%%%%% TEMP TESTING %%%%%%
load(files{1})
Params = TrialData.Params;
DEBUG = true;

if DEBUG
    [Params.WPTR, Params.ScreenRectangle] = Screen('OpenWindow', 0, 0, [50 100 1750 1000]);
else
    [Params.WPTR, Params.ScreenRectangle] = Screen('OpenWindow', max(Screen('Screens')), 0);
end

Params.Center = [mean(Params.ScreenRectangle([1,3])),mean(Params.ScreenRectangle([2,4]))];

% Font
Screen('TextFont',Params.WPTR, 'Arial');
Screen('TextSize',Params.WPTR, 28);

TargetsCol = repmat(Params.TargetsColor,Params.NumReachTargets,1);
TargetsCol(TrialData.TargetID,:) = Params.CuedTargetColor; % cue
%



% draw target triangles
for i=1:Params.NumReachTargets,
    % center vertices to define triangle for each target
    TargetVerts = Params.ReachTargetVerts{i}*0.4;
    TargetVerts(:,1) = TargetVerts(:,1) + Params.Center(1);
    TargetVerts(:,2) = TargetVerts(:,2) + Params.Center(2);

    Screen('FillPoly', Params.WPTR, ...
        TargetsCol(i,:)', TargetVerts, 1);
    Screen('FramePoly', Params.WPTR, ... % black frame around triangles
        0, TargetVerts, Params.TargetSpacing);

    % draw target circles
    CircRect = Params.InnerCircleRect*0.40;
    CircRect([1,3]) = CircRect([1,3]) + Params.Center(1); % add x-pos
    CircRect([2,4]) = CircRect([2,4]) + Params.Center(2); % add y-pos
    Screen('FillOval', Params.WPTR, ...
        Params.InnerCircleColor, CircRect')

end
% Screen('DrawingFinished', Params.WPTR);
% Screen('Flip', Params.WPTR);


% draw rectangles
% left border, top border , right border, bottom border
%     ht = [.2 .4 .5 0.4]*100;
% rect=[100 250-ht(1) 150 250;...
%     150 250-ht(2) 200 250;...
%     200 250-ht(3) 250 250;
%     250 250-ht(4) 300 250]';
% col = [100 100 250;...
%     100 150 250;...
%     100 175 250;
%     100 200 250]';
%
% Screen('FillRect', Params.WPTR, col, rect)
%
% Screen('DrawText',Params.WPTR,'R',[100+10],[260],[225 225 225]);
% Screen('DrawText',Params.WPTR,'D',[150+10],[260],[225 225 225]);
% Screen('DrawText',Params.WPTR,'L',[200+10],[260],[225 225 225]);
% Screen('DrawText',Params.WPTR,'U',[250+10],[260],[225 225 225]);
%
% thresh = TrialData.Params.NeuralNetSoftMaxThresh*200;
% Screen('DrawLine',Params.WPTR,[200 20 20],100,250-thresh,300,250-thresh,3)
% tim_display = '0.35';
% Screen('DrawText',Params.WPTR,tim_display,[1200],[750],[225 225 225]);


ArrowStart = Params.Center;
Click_Decision=1;
if Click_Decision == 1
    temp_dir = .30*Params.ReachTargetPositions(1,:);
elseif Click_Decision == 2
    temp_dir = 0.3*Params.ReachTargetPositions(2,:);
elseif Click_Decision == 3
    temp_dir = 0.3*Params.ReachTargetPositions(3,:);
elseif Click_Decision == 4
    temp_dir = 0.3*Params.ReachTargetPositions(4,:);
elseif Click_Decision == 0 % null class
    temp_dir = 0;
end
ArrowEnd = Params.Center + temp_dir;

% draw the arrow
Screen('DrawLine', Params.WPTR, [255 50 50],ArrowStart(1),ArrowStart(2),...
    ArrowEnd(1),ArrowEnd(2),2);
Screen('FillOval',Params.WPTR,[255 50 50],[ArrowEnd(1)-20,ArrowEnd(2)-20,...
    ArrowEnd(1)+20,ArrowEnd(2)+20],2);
Screen('FrameOval',Params.WPTR,[255 050 50],[ArrowEnd(1)-20,ArrowEnd(2)-20,...
    ArrowEnd(1)+20,ArrowEnd(2)+20],2);

Screen('DrawingFinished', Params.WPTR);
Screen('Flip', Params.WPTR);
%pause(1)
%Screen('Flip', Params.WPTR);

% %%%% TEMP TESTING %%%%%%
%
% %%% CHECKING IF NN OUTPUT IS SAME AS ONLINE EXPERIMENT %%%%
% for iter=1:length(files)
%     load(files{iter})
%     % get the classifier
%     net = TrialData.Params.NeuralNetFunction;
%     % get the features
%     feat = TrialData.SmoothedNeuralFeatures;
%
%     clicker_state = TrialData.ClickerState
%
%
% end
% %%%%%



%add clicker path
addpath('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2\20210324\Models\')



% open the screen
[Params.WPTR, Params.ScreenRectangle] = Screen('OpenWindow', 0, 0, [50 100 1750 1000]);
Params.Center = [mean(Params.ScreenRectangle([1,3])),mean(Params.ScreenRectangle([2,4]))];
% Font
Screen('TextFont',Params.WPTR, 'Arial');
Screen('TextSize',Params.WPTR, 28);

% movie
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\Movies\')
moviePtr = Screen('CreateMovie', Params.WPTR, 'B1_Imag_Actions.avi', 1920,1080,6)%;

% now make the videos
for iter=1:length(files)



    clear TrialData
    load(files{iter})
    tim = (1/TrialData.Params.UpdateRate)*length(TrialData.ClickerState);

    % check if correct and fast
    if TrialData.SelectedTargetID ==  TrialData.TargetID && tim<2

        % get the classifier
        %net = TrialData.Params.NeuralNetFunction;
        % net='MLP_4Dir_Actions_Imagined_20210324';
        % get the features
        %feat = TrialData.SmoothedNeuralFeatures;
        % get the Params
        Params1 = TrialData.Params;
        Params1.WPTR = Params.WPTR;
        Params1.ScreenRectangle = Params.ScreenRectangle;
        Params1.Center = Params.Center;
        Params = Params1;

        Click_Decision = TrialData.ClickerState;

        % set counter
        Counter=0;
        tim=0;
        % loop through the task state
        for loop=1:length(TrialData.TaskState)

            %             % get the classifier output
            %             dec = feval(net,feat{loop}(129:end));
            %             [aa bb] = max(dec)
            %             if aa>=TrialData.Params.NeuralNetSoftMaxThresh
            %                 Click_Decision = bb;
            %             else
            %                 Click_Decision = 0;
            %             end
            %
            %
            %             % plot the classifier o/p on the top left
            %             % ht is scaled to 200 pixels
            %             ht = dec*200;
            %             thresh = TrialData.Params.NeuralNetSoftMaxThresh*200;
            %             plot_dec_bars(Params,ht,thresh)
            %             %plot_class_op(Params.WPTR,ht);



            task_state = TrialData.TaskState(loop);

            % plot time
            tim_bin = tim*(1/Params.UpdateRate);
            tim = tim+1;
            tim_display = [num2str(tim_bin) 's'];
            Screen('DrawText',Params.WPTR,tim_display,[1250],[800],[225 225 225]);

            switch task_state
                % if task state 1
                case 1
                    % just display the grid layout
                    plot_target_layout(Params,0.45)

                    %flip to screen
                    Screen('DrawingFinished', Params.WPTR);
                    Screen('Flip', Params.WPTR);
                    % capture frame
                    Screen('AddFrameToMovie', Params.WPTR)
                    pause(.1)

                    % if task state 2
                case 2
                    % plot the grid layout, with cue lighted up
                    plot_target_cue_layout(Params,TrialData,0.45)

                    % draw a light colored arrow
                    %plot_cue_arrow(Params,Click_Decision,0.3)



                    %flip to screen
                    Screen('DrawingFinished', Params.WPTR);
                    Screen('Flip', Params.WPTR);
                    % capture frame
                    Screen('AddFrameToMovie', Params.WPTR)
                    pause(.1)


                    % if task state 3
                case 3
                    % plot the grid layout, with cue lighted up
                    plot_target_cue_layout(Params,TrialData,0.45)

                    % draw fully colored arrow
                    plot_target_arrow(Params,Click_Decision,0.3)

                    % track decision
                    if Click_Decision == TrialData.TargetID
                        Counter = Counter+1;
                    else
                        Counter = 0;
                    end

                    % draw the red circle if successful trial
                    if Counter == Params.ClickCounter
                        CursorCol = Params.InTargetColor';
                        CursorRect = Params.CursorRect*0.9;
                        reach_loc = 0.35*Params.ReachTargetPositions(Click_Decision,:);
                        CursorRect([1,3]) = CursorRect([1,3]) + Params.Center(1) + reach_loc(1); % add x-pos
                        CursorRect([2,4]) = CursorRect([2,4]) + Params.Center(2)+ reach_loc(2); % add y-pos
                        Screen('FillOval', Params.WPTR, ...
                            CursorCol', CursorRect')
                    end


                    %flip to screen
                    Screen('DrawingFinished', Params.WPTR);
                    Screen('Flip', Params.WPTR);

                    % capture frame
                    Screen('AddFrameToMovie', Params.WPTR)

                    pause(.1)

                    % if task state 4
                case 4
                    % blank screen
                    Screen('Flip', Params.WPTR);
                    % capture frame
                    Screen('AddFrameToMovie', Params.WPTR)
                    pause(.1)
            end

        end
    end

end
%
Screen('FinalizeMovie', moviePtr);
%
% % adding a frame to a movie
% %Screen('AddFrameToMovie', ...
% %    windowPtr, [rect] ,[bufferName] ,[moviePtr = 0], [frameduration=1])
%
% movieFile = 'E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2\Movies\20210324_143514.avi';
% moviePtr = Screen('CreateMovie', Params.WPTR, movieFile);
%

%% CREATING POOLING FOR DECODER

% load a channel map

% take the average over a rolling window over the map, stride 2.
[xx yy]= size(TrialData.Params.ChMap)
% move a 2X2 square along the grid and average within the square

tmp=randn(xx,yy);
B=[1 1;1 1];


tic
temp = smooth2D(tmp);
toc


% creating a decoder see how it does
clc;clear
close all
root_path='E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';

foldernames = {'20210609'};
cd(root_path)

files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=9:11
        filepath=fullfile(folderpath,D(j).name,'Imagined')
        files = [files;findfiles('',filepath)'];
    end
end


% load the data for each target
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
for ii=1:length(files)
    disp(ii)
    load(files{ii});
    features  = TrialData.SmoothedNeuralFeatures;
    kinax = TrialData.TaskState;
    kinax = [find(kinax==3)];
    temp = cell2mat(features(kinax));

    % get the pooled data
    new_temp=[];
    [xx yy] = size(TrialData.Params.ChMap);
    for k=1:size(temp,2)
        tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
        tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
        tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
        pooled_data=[];
        for i=1:2:xx
            for j=1:2:yy
                delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                pooled_data = [pooled_data; delta; beta ;hg];
            end
        end
        new_temp= [new_temp pooled_data];
    end
    temp=new_temp;
    if TrialData.TargetID == 1
        D1 = [D1 temp];
    elseif TrialData.TargetID == 2
        D2 = [D2 temp];
    elseif TrialData.TargetID == 3
        D3 = [D3 temp];
    elseif TrialData.TargetID == 4
        D4 = [D4 temp];
    elseif TrialData.TargetID == 5
        D5 = [D5 temp];
    elseif TrialData.TargetID == 6
        D6 = [D6 temp];
    end
end


clear condn_data
% combing both onlien plus offline
%idx=641;
idx = [1:96];
condn_data{1}=[D1(idx,:) ]'; % right hand
condn_data{2}= [D2(idx,:)]'; % both feet
condn_data{3}=[D3(idx,:)]'; % left hand
condn_data{4}=[D4(idx,:)]'; % head
condn_data{5}=[D5(idx,:)]'; % mime up
condn_data{6}=[D6(idx,:)]'; % tongue in

A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};
E = condn_data{5};
F = condn_data{6};


clear N
N = [A' B' C' D' E' F'];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1)];

T = zeros(size(T1,1),4);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;
[aa bb]=find(T1==5);[aa(1) aa(end)]
T(aa(1):aa(end),5)=1;
[aa bb]=find(T1==6);[aa(1) aa(end)]
T(aa(1):aa(end),6)=1;

% code to train a neural network
clear net
net = patternnet([96 96 96]) ;
net.performParam.regularization=0.2;
net = train(net,N,T','UseGPU','yes');
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
genFunction(net,'MLP_6DoF_Trained4mOnlineData_3Features')
save net net



%% LOOKING AT TEMPORAL PROPERTIES OF ECOG FOR B1 CONTROL

clc;clear
close all
root_path='E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';

foldernames = {'20210604'};
cd(root_path)

files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=11:14
        filepath=fullfile(folderpath,D(j).name,'Imagined')
        files = [files;findfiles('',filepath)'];
    end
end


% load the data for each target
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1000);
for i=2:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.BroadbandData;
    kinax = TrialData.TaskState;
    if kinax(1) == 0
        kinax=kinax(2:end);
    end
    kinax = [find(kinax==3)];
    temp = cell2mat(features(kinax)');
    if size(temp,1)<5000
        temp(end+1:5000,:) = temp(end,:);
    end
    temp = temp(1:5000,:);
    temp = abs(hilbert(filtfilt(bpFilt,temp)));
    if TrialData.TargetID == 1
        D1 = cat(3,D1,temp);
        %D1 = [D1 temp];
    elseif TrialData.TargetID == 2
        D2 = cat(3,D2,temp);
    elseif TrialData.TargetID == 3
        D3 = cat(3,D3,temp);
    elseif TrialData.TargetID == 4
        D4 = cat(3,D4,temp);
    elseif TrialData.TargetID == 5
        D5 = cat(3,D5,temp);
    elseif TrialData.TargetID == 6
        D6 = cat(3,D6,temp);
    end
end

% building AR models for each trial and seeing how they are
coeff=[];
for i=1:size(D1,3)
    tmp = squeeze(D1(1:2500,3,i));
    ar_mdl = ar(tmp,4);
    coeff  = [coeff getpvec(ar_mdl)];
end

coeff1=[];
for i=1:size(D3,3)
    tmp = squeeze(D3(1:2500,3,i));
    ar_mdl = ar(tmp,4);
    coeff1  = [coeff1 getpvec(ar_mdl)];
end

figure;plot3(coeff(1,:),coeff(2,:),coeff(4,:),'.','MarkerSize',20)
hold on
plot3(coeff1(1,:),coeff1(2,:),coeff1(4,:),'.','MarkerSize',20)

size(D1)
tmp1 = squeeze(mean(D1,3))';
tmp2 = squeeze(mean(D2,3))';
tmp3 = squeeze(mean(D3,3))';
tmp4 = squeeze(mean(D4,3))';
tmp5 = squeeze(mean(D5(:,:,[1:2 4:end]),3))';
tmp6 = squeeze(mean(D6,3))';

clear tmp
tmp(:,1,:) = tmp1;
tmp(:,2,:) = tmp2;
tmp(:,3,:) = tmp3;
tmp(:,4,:) = tmp4;
tmp(:,5,:) = tmp5;
tmp(:,6,:) = tmp6;

firingRatesAverage_Subj{1} = tmp



for i=1:16
    figure;
    plot(squeeze(D5(:,:,i)));
end

%% LOOKING AT PERFROMANCE OF BOTH HANDS
% IS THE MAX WINNER IN A TRIAL ACROSS TIME GIVEN DECODES?

clc;clear
close all
root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';

foldernames = {'20220713'};
cd(root_path)

files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'DiscreteArrow');
    D=dir(folderpath);
    for j=8:9
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed')
        files = [files;findfiles('',filepath)'];
    end
end


% look at the decodes per direction to get a max vote
T=zeros(4);
for i=1:length(files)
    disp(i)
    load(files{i});
    kinax = TrialData.TaskState;
    clicker_state = TrialData.ClickerState;
    idx = TrialData.TargetID;
    t(1) = sum(clicker_state ==1);
    t(2) = sum(clicker_state ==2);
    t(3) = sum(clicker_state ==3);
    t(4) = sum(clicker_state ==4);
    [aa bb]=max(t);
    T(idx,bb) = T(idx,bb)+1;
end
T
for i=1:size(T)
    T(i,:) = T(i,:)./sum(T(i,:));
end
figure;imagesc(T)
colormap bone
caxis([0 .9])
xticks([1:4])
yticks([1:4])
xticklabels({'Rt thumb','Both hands','Lt. thumb','rt index'})
yticklabels({'Rt thumb','Both hands','Lt. thumb','Rt index'})
set(gcf,'Color','w')
set(gca,'FontSize',12)
title('Classif. using temporal history')

%%%% DOING THE SAME NOW BUT IN THE ROBOT 3D ARROW ENVIRONMENT
clc;clear
close all
root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';

foldernames = {'20220713'};
cd(root_path)

files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed')
        files = [files;findfiles('',filepath)'];
    end
end


% look at the decodes per direction to get a max vote
T=zeros(7);
for i=1:length(files)
    disp(i)
    load(files{i});
    kinax = TrialData.TaskState;
    clicker_state = TrialData.ClickerState;
    idx = TrialData.TargetID;
    t(1) = sum(clicker_state ==1);
    t(2) = sum(clicker_state ==2);
    t(3) = sum(clicker_state ==3);
    t(4) = sum(clicker_state ==4);
    t(5) = sum(clicker_state ==5);
    t(6) = sum(clicker_state ==6);
    t(7) = sum(clicker_state ==7);
    [aa bb]=max(t);
    T(idx,bb) = T(idx,bb)+1;
end
T
for i=1:size(T)
    T(i,:) = T(i,:)./sum(T(i,:));
end
figure;imagesc(T)
colormap bone
caxis([0 .9])
xticks([1:4])
yticks([1:4])
xticklabels({'Rt thumb','Both hands','Lt. thumb','rt index'})
yticklabels({'Rt thumb','Both hands','Lt. thumb','Rt index'})
set(gcf,'Color','w')
set(gca,'FontSize',12)
title('Classif. using temporal history')

%% PERFORMANCE MEASURE USING MAX VOTE STRATEGY FOR DECODES (MAIN)
% on a block by block basis and also across blocks


%%%% DOING THE SAME NOW BUT IN THE ROBOT 3D ARROW ENVIRONMENT
clc;clear

root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'
addpath 'C:\Users\nikic\Documents\MATLAB'

foldernames = {'20220713'};
cd(root_path)

files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=1:length(D)
        filepath=fullfile(folderpath,D((j)).name,'BCI_Fixed');
        if exist(filepath)
            disp(filepath)
            files = [files;findfiles('',filepath)'];
        end
    end
end


% look at the decodes per direction to get a max vote
T=zeros(7);
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
        clicker_state = TrialData.FilteredClickerState;
        idx = TrialData.TargetID;
        t(1) = sum(clicker_state ==1);
        t(2) = sum(clicker_state ==2);
        t(3) = sum(clicker_state ==3);
        t(4) = sum(clicker_state ==4);
        t(5) = sum(clicker_state ==5);
        t(6) = sum(clicker_state ==6);
        t(7) = sum(clicker_state ==7);
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
T
for i=1:size(T)
    T(i,:) = T(i,:)./sum(T(i,:));
end
figure;imagesc(T)
colormap bone
caxis([0 1])
xticks([1:7])
yticks([1:7])
xticklabels({'Rt thumb','Both Feet','Lt. thumb','Head', 'Lips','Tong','Both middle'})
yticklabels({'Rt thumb','Both Feet','Lt. thumb','Head', 'Lips','Tong','Both middle'})
set(gcf,'Color','w')
set(gca,'FontSize',12)
%title('Classif. using temporal history original action space')
colorbar

% bit rate calculations
tim_to_target = tim_to_target.*(1/TrialData.Params.UpdateRate);
B = log2(7-1) * (sum(num_suc)-sum(num_fail)) / sum(tim_to_target)
title(['Accuracy of ' num2str(100*mean(diag(T))) '%' ' and bitrate of ' num2str(B)])



%% BLOCK BY BLOCK BIT RATE CALCULATIONS ACROSS DAYS
%%%%(MAIN)


clc;clear
close all


root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'
addpath 'C:\Users\nikic\Documents\MATLAB'


%foldernames = {'20220601'};
foldernames = {'20210813','20210818','20210825','20210827','20210901','20210903',...
    '20210910','20210915','20210917','20210922','20210924'};


cd(root_path)
folders={};
br_across_days={};
time2target_days=[];
acc_days=[];
conf_matrix_overall=[];
for i=1:10%length(foldernames)

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
            filepath
            files = [files;findfiles('mat',filepath)'];
            folders=[folders;filepath];
        end
        if length(files)>0
            [b,a,t,T] = compute_bitrate(files,7);
            %[b,a,t,T] = compute_bitrate_constTime(files,7);
            conf_matrix_overall = cat(3,conf_matrix_overall,T);
            br = [br b];
            acc = [acc mean(a)];
            time2target = [time2target; mean(t)];
            %[br, acc ,t] = [br compute_bitrate(files)];
        end
    end
    close all
    br_across_days{i}=br;
    time2target_days{i} = time2target(:);
    acc_days{i} = acc(:);
    %time2target_days = [time2target_days ;time2target(:)];
    %acc_days = [acc_days ;acc(:)];
end

%plot results as a scatter plot
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95');
figure;hold on
br=[];
brh=[];
%cmap = brewermap(11,'blues');
%cmap=flipud(cmap);
%cmap=cmap(1:length(br_across_days),:);
%cmap=flipud(cmap);
cmap = turbo(7);%turbo(length(br_across_days));
for i=1:7%length(br_across_days)
    tmp = br_across_days{i};
    brh = [brh tmp];
    idx= i*ones(size(tmp))+0.1*randn(size(tmp));
    plot(idx,tmp,'.','Color',cmap(i,:),'MarkerSize',15);
    br(i) = median(tmp);
end
plot(br(1:end),'k','LineWidth',2)
days={'1','5','12','14','19','21','28','32','35','40','42'};
xticks(1:length(br))
set(gca,'XTickLabel',days)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('BitRate')
set(gca,'LineWidth',1)
%set(gca,'Color',[.85 .85 .85])
xlim([0 7.5])
ylim([0 3.5])
%

figure
boxplot(brh,'whisker',1.75)
set(gcf,'Color','w')
xticks(1)
xticklabels('PnP Experiment 1')
ylabel('Effective bit rate')
set(gca,'FontSize',12)
box off
xlim([.75 1.25])
ylim([0 3.5])
yticks([0:.5:3.5])
set(gca,'LineWidth',1,'TickLength',[0.025 0.025]);

figure;hold on
acc=[];
acch=[];
for i=1:7%length(acc_days)
    tmp  = acc_days{i};
    idx= i*ones(size(tmp))+0.1*randn(size(tmp));
    plot(idx,tmp,'.','Color',cmap(i,:),'MarkerSize',15);
    acc(i) = median(tmp);
    acch = [acch ;tmp];
end
plot(acc,'k','LineWidth',2)
ylim([0 1])
xticks(1:length(acc))
set(gca,'XTickLabel',days)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('Decoder Accuracy')
set(gca,'LineWidth',1)
xlim([0.5 7.5])
h=hline(1/7);
set(h,'LineWidth',2)
yticks([0:.2:1])

figure;hold on
t2t=[];
t2th=[];
for i=1:7%length(time2target_days)
    tmp  = time2target_days{i};
    idx= i*ones(size(tmp))+0.1*randn(size(tmp));
    plot(idx,tmp,'.','Color',cmap(i,:),'MarkerSize',15);
    t2t(i) = median(tmp);
    t2th = [t2th;tmp];
end
plot(t2t,'k','LineWidth',2)
ylim([0 3])
xticks(1:length(acc))
set(gca,'XTickLabel',days)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('Mean time to Target (s)')
set(gca,'LineWidth',1)
xlim([0.5 7.5])
yticks([0:.5:3])

figure;hist(acch,10)
xlim([0 1])
vline(1/7,'r')
box off
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Decoder Accuracy')
ylabel('Count')
set(gca,'LineWidth',1)
%vline(median(acch),'k')

figure;hist(t2th,10)
xlim([0 2.25])
box off
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Time to Target')
ylabel('Count')
set(gca,'LineWidth',1)




%save bit_rate_discrete_PnP -v7.3


%
% figure;hist(time2target_days);
% figure;hist(acc_days);


%
% figure;boxplot(acc_days,'notch','off')
% figure;
% idx=ones(size(acc_days)) + 0.1*randn(size(acc_days));
% scatter(idx,acc_days,'k')


%% BLOCK BY BLOCK BIT RATE CALCULATIONS ACROSS DAYS - PNP 2
%%%%(MAIN)
% second plug and play experiment 


clc;clear

root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath('C:\Users\nikic\Documents\MATLAB')

%foldernames = {'20210903'};
foldernames = {'20211013','20211015','20211020','20211022','20211027','20211029',...
    '20211103','20211105','20211117','20211119'};

% days -> 1 3 8 10 15 17 22 24 36 38

cd(root_path)
folders={};
br_across_days={};
time2target_days=[];
acc_days=[];
for i=1:length(foldernames)
    
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);

    if i==1
        idx=[1 2 3 4 7 8];
        D = D(idx);
    end

    if i==2
        idx=[1 2 4:length(D)];
        D = D(idx);
    end

    if i==6
        idx=[1 2 3 4 7];
        D = D(idx);
    end

    if i==7
        idx=[1 2 5 8 ];
        D = D(idx);
    end

    br=[];acc=[];time2target=[];
    for j=3:length(D)
        files=[];
        filepath=fullfile(folderpath,D((j)).name,'BCI_Fixed');
        if exist(filepath)
            filepath
            files = [files;findfiles('mat',filepath)'];
            folders=[folders;filepath];
        end
        if length(files)>0
            [b,a,t,T] = compute_bitrate(files,7);
            %[b,a,t] = compute_bitrate_constTime(files,7);
            br = [br b];
            acc = [acc mean(a)];
            time2target = [time2target; mean(t)];
            %[br, acc ,t] = [br compute_bitrate(files)];
        end
    end
    close all
    br_across_days{i}=br;
    time2target_days{i} = time2target(:);
    acc_days{i} = acc(:);
    %time2target_days = [time2target_days ;time2target(:)];
    %acc_days = [acc_days ;acc(:)];
end

% plotting as scatter plot
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95');
figure;hold on
br=[];
brh=[];
cmap = turbo(10);%turbo(length(br_across_days));
for i=1:10%length(br_across_days)
    tmp = br_across_days{i};
    brh =[brh tmp];
    %tmp=tmp(tmp>0.5);
    idx= i*ones(size(tmp))+0.1*randn(size(tmp));
    plot(idx,tmp,'.','Color',cmap(i,:),'MarkerSize',15);
    br(i) = median(tmp);
end
plot(br(1:end),'k','LineWidth',2)
days={'1', '3', '8' ,'10' ,'15', '17', '22', '24', '36', '38'};
%days={'1','5','12','14','19','21','28','32','35','40','42'};
xticks(1:length(br))
set(gca,'XTickLabel',days)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('BitRate')
set(gca,'LineWidth',1)
%set(gca,'Color',[.85 .85 .85])
xlim([0 10.5])
ylim([0 3.5])

figure
boxplot(brh,'whisker',1.75)
set(gcf,'Color','w')
xticks(1)
xticklabels('PnP Experiment 2')
ylabel('Effective bit rate')
set(gca,'FontSize',12)
box off
xlim([.75 1.25])
ylim([0 3.5])
yticks([0:.5:3.5])
set(gca,'LineWidth',1,'TickLength',[0.025 0.025]);

% accuracy
figure;hold on
acc=[];
acch=[];
idxx=[];
for i=1:10%length(acc_days)
    tmp  = acc_days{i};
    %tmp=tmp(tmp>0.6);
    idx= i*ones(size(tmp))+0.1*randn(size(tmp));
    plot(idx,tmp,'.','Color',cmap(i,:),'MarkerSize',15);
    acc(i) = median(tmp);
    acch = [acch ;tmp];
    idxx=[idxx;idx];
end
plot(acc,'k','LineWidth',2)
ylim([0 1])
xticks(1:length(acc))
set(gca,'XTickLabel',days)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('Decoder Accuracy')
set(gca,'LineWidth',1)
xlim([0.5 10.5])
h=hline(1/7);
set(h,'LineWidth',2)
yticks([0:.2:1])

figure;hold on
t2t=[];
t2th=[];
idxx=[];
for i=1:10%length(time2target_days)
    tmp  = time2target_days{i};
    %tmp=tmp(tmp<1.2);
    idx= i*ones(size(tmp))+0.1*randn(size(tmp));
    plot(idx,tmp,'.','Color',cmap(i,:),'MarkerSize',15);
    t2t(i) = median(tmp);
    t2th = [t2th;tmp];
    idxx=[idxx;idx];
end
plot(t2t,'k','LineWidth',2)
ylim([0 3])
xticks(1:length(acc))
set(gca,'XTickLabel',days)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Days - PnP')
ylabel('Time to Target (s)')
set(gca,'LineWidth',1)
xlim([0.5 10.5])

figure;hist(acch,10)
xlim([0 1])
vline(1/7,'r')
box off
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Decoder Accuracy')
ylabel('Count')
set(gca,'LineWidth',1)
%vline(median(acch),'k')

figure;hist(t2th,10)
xlim([0 2.25])
box off
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Time to Target')
ylabel('Count')
set(gca,'LineWidth',1)



%save bit_rate_discrete_PnP2 -v7.3


%% (MAIN) plotting combined time to target and accuracy across both PnP experiments

clear;clc
close all

a = load('bit_rate_discrete_PnP.mat');
b = load('bit_rate_discrete_PnP2.mat');

acch =[a.acch;b.acch];
figure;hist(acch,10)
xlim([0 1])

t2th =[a.t2th;b.t2th];
figure;hist(t2th,10)
xlim([0 2.5])

br =[a.br';b.br'];
figure;hist(br,10)

tmp = [(a.br) (b.br)];


br_overall=[];
br_across_days = a.br_across_days;
for i=1:7
    br_overall = [br_overall br_across_days{i}];        
end
br_across_days = b.br_across_days;
for i=1:10
    br_overall = [br_overall br_across_days{i}];        
end

m = median(br_overall);
mb = sort(bootstrp(1000,@median,br_overall));
[mb(25) mb(975)]


%% %% BLOCK BY BLOCK BIT RATE CALCULATIONS ACROSS DAYS 9DOF
%%%%(MAIN)


clc;clear

root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';

%foldernames = {'20210903'};
foldernames = {'20220204'};


cd(root_path)
folders={};
br_across_days={};
time2target_days=[];
acc_days=[];
for i=1:length(foldernames)
    
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    if i==1
        idx = [1 2 12:15];
        D = D(idx);   
    end
    br=[];acc=[];time2target=[];
    for j=3:length(D)
        files=[];
        filepath=fullfile(folderpath,D((j)).name,'BCI_Fixed');
        if exist(filepath)
            filepath
            files = [files;findfiles('mat',filepath)'];
            folders=[folders;filepath];
        end
        if length(files)>0
            [b,a,t] = compute_bitrate(files,9);
            br = [br b];
            acc = [acc mean(a)];
            time2target = [time2target; mean(t)];
            %[br, acc ,t] = [br compute_bitrate(files)];
        end
    end
    close all
    br_across_days{i}=br;
    time2target_days{i} = time2target(:);
    acc_days{i} = acc(:);
    %time2target_days = [time2target_days ;time2target(:)];
    %acc_days = [acc_days ;acc(:)];
end

%% RUNNING ICA ON THE DATA
clc;clear
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
load 6DOF_Online_Data_3Feat


% arrange the data accordingly
data=[];
idx=[];
for i=1:length(condn_data)
    tmp = condn_data{i};
    tmp = tmp(:,257:end);
    idx = [idx; i*ones(size(tmp,1),1)];
    data = [data; tmp];
end

data = data-mean(data);
[weights,sphere,~,~,~,~,activations] = ...
    runica(data','maxsteps',1000,'momentum',0.2,'sphering','off');

figure;hold on
cmap = parula(length(unique(idx)));
for i=1:size(activations,2)
    c = cmap(idx(i),:);
    plot3(activations(1,i),activations(2,i),activations(3,i),'.','Color',c);
end


%% (MAIN) Classifier with historical data 
% and also GETTING DATA FOR JENSEN FOR TESTING META TRANSFER 
% split into erp and online on a day to day basis
% get the pooled data and feed it to jason


clc;clear
close all
root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';

% for only 6 DoF original:
%foldernames = {'20210526','20210528','20210602','20210609_pm','20210611'};

foldernames = {'20210615','20210616','20210623','20210625','20210630','20210702',...
    '20210707','20210716','20210728','20210804','20210806','20210813','20210818',...
    '20210825','20210827','20210901','20210903','20210910','20210917','20210929',...
    '20211001','20211006','20211008','20211013','20211020','20211022','20211027','20211029',...
    '20211103','20211105','20211117','20211119','20220126','20220128','20220202',...
    '20220204','20220209','20220211','20220216','20220218','20220223','20220225',...
    '20220302','20220304','20220309','20220311','20220316','20220323','20220325',...
    '20220422','20220427','20220429'};
cd(root_path)

imag_files={};
online_files={};
k=1;jj=1;
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    imag_files_temp=[];
    online_files_temp=[];
    
    if i==17
        D = D([1:2 3:6]);
    elseif i==19
        D = D([1:2 5:10]);
    elseif i==29
        D = D([1:2 8:9]);        
    end
    
    if strcmp(foldernames{i},'20220427')
        D=D([1:2 5:7]);        
    end
    
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'Imagined');
        if exist(filepath)
            imag_files_temp = [imag_files_temp;findfiles('',filepath)'];
        end
        filepath1=fullfile(folderpath,D(j).name,'BCI_Fixed');
        if exist(filepath1)
            online_files_temp = [online_files_temp;findfiles('',filepath1)'];
        end
    end
    if ~isempty(imag_files_temp)
        imag_files{k} = imag_files_temp;k=k+1;
    end
    if ~isempty(online_files_temp)
        online_files{jj} = online_files_temp;jj=jj+1;
    end
    %     imag_files{i} = imag_files_temp;
    %     online_files{i} = online_files_temp;
end

% load the imagined data files
Data=[];
files_not_loaded=[];
for iter=1:length(imag_files)
    tmp_files = imag_files{iter};
    D1={};
    D2={};
    D3={};
    D4={};
    D5={};
    D6={};
    D7={};
    for ii=1:length(tmp_files)
        disp(ii)
        
        indicator=1;
        try
            load(tmp_files{ii});
        catch ME
            warning('Not able to load file, skipping to next')
            indicator = 0;
            files_not_loaded=[files_not_loaded;tmp_files{ii}];
        end
        
        if indicator
            features  = TrialData.SmoothedNeuralFeatures;
            kinax = TrialData.TaskState;
            kinax = [find(kinax==3)];
            temp = cell2mat(features(kinax));
            
            %get the pooled data
            new_temp=[];
            [xx yy] = size(TrialData.Params.ChMap);
            for k=1:size(temp,2)
                tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
                tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
                tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
                pooled_data=[];
                for i=1:2:xx
                    for j=1:2:yy
                        delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                        beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                        hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                        pooled_data = [pooled_data; delta; beta ;hg];
                    end
                end
                new_temp= [new_temp pooled_data];
            end
            temp_data=new_temp;
            
            
%             % get the pooled data for state 1
%             kinax = TrialData.TaskState;
%             kinax = [find(kinax==1)];
%             temp = cell2mat(features(kinax));
%             
%             %get the pooled data
%             new_temp=[];
%             [xx yy] = size(TrialData.Params.ChMap);
%             for k=1:size(temp,2)
%                 tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
%                 tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
%                 tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
%                 pooled_data=[];
%                 for i=1:2:xx
%                     for j=1:2:yy
%                         delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
%                         beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
%                         hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
%                         pooled_data = [pooled_data; delta; beta ;hg];
%                     end
%                 end
%                 new_temp= [new_temp pooled_data];
%             end
%             temp_bl = new_temp;
%             
            % baseline the data
            %temp = temp_data - mean(temp_bl,2);
            
            
            % no baseline
            temp = temp_data;
            
            
            %temp = temp(769:896,:);
            if TrialData.TargetID == 1
                D1=cat(2,D1,temp);
            elseif TrialData.TargetID == 2
                D2=cat(2,D2,temp);
            elseif TrialData.TargetID == 3
                D3=cat(2,D3,temp);
            elseif TrialData.TargetID == 4
                D4=cat(2,D4,temp);
            elseif TrialData.TargetID == 5
                D5=cat(2,D5,temp);
            elseif TrialData.TargetID == 6
                D6=cat(2,D6,temp);
            elseif TrialData.TargetID == 7
                D7=cat(2,D7,temp);
            end
        end
    end
    
    clear condn_data
    %idx = [1:128];
    condn_data{1}=[D1 ]'; % right hand
    condn_data{2}= [D2]'; % both feet
    condn_data{3}=[D3]'; % left hand
    condn_data{4}=[D4]'; % head
    condn_data{5}=[D5]'; % mime up
    condn_data{6}=[D6]'; % tongue in
    condn_data{7}=[D7]'; % squeeze both hands
    
    Data.Day(iter).imagined_data = condn_data;
end

% load the online data files
files_not_loaded_fixed=[];
for iter=1:length(online_files)
    tmp_files = online_files{iter};
    D1={};
    D2={};
    D3={};
    D4={};
    D5={};
    D6={};
    D7={};
    for ii=1:length(tmp_files)
        disp(ii)
        
        indicator=1;
        try
            load(tmp_files{ii});
        catch ME
            warning('Not able to load file, skipping to next')
            indicator = 0;
            files_not_loaded_fixed=[files_not_loaded_fixed;tmp_files{ii}];
        end
        
        if indicator
            
            features  = TrialData.SmoothedNeuralFeatures;
            kinax = TrialData.TaskState;
            kinax = [find(kinax==3)];
            temp = cell2mat(features(kinax));
            
            %get the pooled data
            new_temp=[];
            [xx yy] = size(TrialData.Params.ChMap);
            for k=1:size(temp,2)
                tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
                tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
                tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
                pooled_data=[];
                for i=1:2:xx
                    for j=1:2:yy
                        delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                        beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                        hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                        pooled_data = [pooled_data; delta; beta ;hg];
                    end
                end
                new_temp= [new_temp pooled_data];
            end
            temp_data=new_temp;
            
            
%             % get the pooled data for state 1
%             kinax = TrialData.TaskState;
%             kinax = [find(kinax==1)];
%             temp = cell2mat(features(kinax));
%             
%             %get the pooled data
%             new_temp=[];
%             [xx yy] = size(TrialData.Params.ChMap);
%             for k=1:size(temp,2)
%                 tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
%                 tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
%                 tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
%                 pooled_data=[];
%                 for i=1:2:xx
%                     for j=1:2:yy
%                         delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
%                         beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
%                         hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
%                         pooled_data = [pooled_data; delta; beta ;hg];
%                     end
%                 end
%                 new_temp= [new_temp pooled_data];
%             end
%             temp_bl = new_temp;
            
            % baseline the data
            %temp = temp_data - mean(temp_bl,2);
            
            % no baseline
            temp = temp_data;
            
            %temp = temp(769:896,:);
            if TrialData.TargetID == 1
                D1=cat(2,D1,temp);
            elseif TrialData.TargetID == 2
                D2=cat(2,D2,temp);
            elseif TrialData.TargetID == 3
                D3=cat(2,D3,temp);
            elseif TrialData.TargetID == 4
                D4=cat(2,D4,temp);
            elseif TrialData.TargetID == 5
                D5=cat(2,D5,temp);
            elseif TrialData.TargetID == 6
                D6=cat(2,D6,temp);
            elseif TrialData.TargetID == 7
                D7=cat(2,D7,temp);
            end
        end
    end
    
    clear condn_data
    %idx = [1:128];
    condn_data{1}=[D1 ]'; % right hand
    condn_data{2}= [D2]'; % both feet
    condn_data{3}=[D3]'; % left hand
    condn_data{4}=[D4]'; % head
    condn_data{5}=[D5]'; % mime up
    condn_data{6}=[D6]'; % tongue in
    condn_data{7}=[D7]'; % squeeze both hands
    
    Data.Day(iter).online_data = condn_data;
end

cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
save 7DoF_Data_SpatioTemp_AcrossDays_20220429 Data -v7.3

condn_data={};
for i=1:7
    data=[];
    for j=8
        tmp = Data.Day(j).online_data;
        tmp = tmp{i};
        data = [data;tmp];
    end
    condn_data{i} = data;
end



% train an autoencoder and see how it goes
X=[];
idx=[];
for i=1:length(condn_data)
    tmp = condn_data{i};
    X = [X;tmp];
    idx = [idx ;i*ones(size(tmp,1),1)];
end
% artfiact correct
m = mean(X);
for i=1:size(X,2)
    [aa bb] = find(abs(X(:,i))>=4);
    X(aa,i) = m(i);
end
X=X';
size(X)


% X7 is day 7
% X6 is day 6
% need to project data from day 7 onto space spanned by day 6 as decoder is
% built on day 6

jj = randperm(size(X7,1),size(X6,1));
X7a = X7(jj,:);
A = X6\X7a;
X = X7*A;
X=X';

%
% autoenc = trainAutoencoder(X,5,...
%      'EncoderTransferFunction','satlin',...
%         'DecoderTransferFunction','satlin',...
%         'L2WeightRegularization',0.1,...
%         'SparsityRegularization',4,...
%         'SparsityProportion',0.15,...
%         'ScaleData',0,...
%         'MaxEpochs',1000,...
%         'UseGPU',1);
%
% XReconstructed = predict(autoenc,X);
% mseError = mse(X-XReconstructed)
%
% Z = encode(autoenc,X);
% [coeff,score,latent]=pca(Z');
% Z = score';
% cmap = parula(7);
% figure;hold on
% for i=1:7
%     idxx = find(idx==i);
%     plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:));
% end

latent_Z=[];
for loop=1:10
    
    
    % using custom layers
    layers = [ ...
        featureInputLayer(96)
        fullyConnectedLayer(64)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(32)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(8)
        batchNormalizationLayer
        reluLayer('Name','autoencoder')
        fullyConnectedLayer(32)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(64)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(96)
        regressionLayer];
    
    
    %'ValidationData',{XTest,YTest},...
    options = trainingOptions('sgdm', ...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',500, ...
        'Shuffle','every-epoch', ...
        'Verbose',true, ...
        'Plots','training-progress',...
        'MiniBatchSize',4096,...
        'ValidationFrequency',30,...
        'L2Regularization',1e-4,...
        'ExecutionEnvironment','gpu');
    
    % build the autoencoder
    net = trainNetwork(X',X',layers,options);
    
    % MSE
    tmp = predict(net,X');
    mseError = mse(X-tmp')
    %
    % % now get activations in deepest layer
    Z = activations(net,X','autoencoder');
    %
    %     % plotting
    %     [coeff,score,latent]=pca(Z');
    %     Z = score';
    %     cmap = parula(7);
    %     figure;hold on
    %     for i=1:7
    %         idxx = find(idx==i);
    %         plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:));
    %     end
    %     title(num2str(loop))
    latent_Z(loop,:,:) = Z;
end

Z = squeeze(mean(latent_Z,1));
[coeff,score,latent]=pca(Z');
Z = score';
cmap = parula(7);
figure;hold on
for i=1:7
    idxx = find(idx==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:));
end


%
%Z = activations(net,X','autoencoder');
Z = squeeze(mean(latent_Z,1));
Y = tsne(Z','Algorithm','exact','Standardize',false,'Perplexity',30,'NumDimensions',3,...
    'Exaggeration',10);
Y=Y';
cmap = parula(7);
figure;hold on
for i=1:7
    idxx = find(idx==i);
    plot3(Y(1,idxx),Y(2,idxx),Y(3,idxx),'.','color',cmap(i,:));
    %plot(Y(1,idxx),Y(2,idxx),'.','color',cmap(i,:));
end


layers1=layers;
layers1=layers1(1:10);
layers2 = [layers1
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer]



%'ValidationData',{XTest,YTest},...
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',512,...
    'ValidationFrequency',30,...
    'L2Regularization',1e-4,...
    'ExecutionEnvironment','auto');

Y=categorical(idx);
% build the classifier
net = trainNetwork(X',Y,layers2,options);


% % now get activations in deepest layer
Z = activations(net,X','autoencoder');

% plotting
[coeff,score,latent]=pca(Z');
Z = score';
cmap = parula(7);
figure;hold on
for i=1:7
    idxx = find(idx==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:));
end


cmap = parula(7);
figure;hold on
for i=1:7
    idxx = find(idx==i);
    plot3(Z(7,idxx),Z(8,idxx),Z(9,idxx),'color',cmap(i,:));
end


%% (MAIN) building a neural decoder with all the historical data

clc;clear

cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
load 7DoF_Data_SpatioTemp_AcrossDays_20220429
%load 7DoF_Data_SpatioTemp_AcrossDays_20220224
%load Data_SpatioTemp_AcrossDays
%load Data_SpatioTemp_AcrossDays_9282021
%load 7DoF_Data_Training

% get the training data
D1i=[];
D2i=[];
D3i=[];
D4i=[];
D5i=[];
D6i=[];
D7i=[];
for i=1:15%length(Data.Day)
    disp(i)
    tmp=Data.Day(i).imagined_data;
    D1i = [D1i cell2mat(tmp{1}')];
    D2i = [D2i cell2mat(tmp{2}')];
    D3i = [D3i cell2mat(tmp{3}')];
    D4i = [D4i cell2mat(tmp{4}')];
    D5i = [D5i cell2mat(tmp{5}')];
    D6i = [D6i cell2mat(tmp{6}')];
    D7i = [D7i cell2mat(tmp{7}')];
end

D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
for i=1:length(Data.Day)
    disp(i)
    tmp=Data.Day(i).online_data;
    D1 = [D1 cell2mat(tmp{1}')];
    D2 = [D2 cell2mat(tmp{2}')];
    D3 = [D3 cell2mat(tmp{3}')];
    D4 = [D4 cell2mat(tmp{4}')];
    D5 = [D5 cell2mat(tmp{5}')];
    D6 = [D6 cell2mat(tmp{6}')];
    D7 = [D7 cell2mat(tmp{7}')];
end


% build the decoder; compare ML using patternet and using layers
clear condn_data
% combing both onlien plus offline
%idx=641;
idx = [1:96];
condn_data{1}=[D1(idx,:) D1i]'; % right hand
condn_data{2}= [D2(idx,:) D2i]'; % both feet
condn_data{3}=[D3(idx,:) D3i]'; % left hand
condn_data{4}=[D4(idx,:) D4i]'; % head
condn_data{5}=[D5(idx,:) D5i]'; % mime up
condn_data{6}=[D6(idx,:) D6i]'; % tongue in
condn_data{7}=[D7(idx,:) D7i]'; % both hands

% 2norm
for i=1:length(condn_data)
   tmp = condn_data{i}; 
   for j=1:size(tmp,1)
       tmp(j,:) = tmp(j,:)./norm(tmp(j,:));
   end
   condn_data{i}=tmp;
end

A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};
E = condn_data{5};
F = condn_data{6};
G = condn_data{7};


clear N
N = [A' B' C' D' E' F' G'];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1)];

T = zeros(size(T1,1),7);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;
[aa bb]=find(T1==5);[aa(1) aa(end)]
T(aa(1):aa(end),5)=1;
[aa bb]=find(T1==6);[aa(1) aa(end)]
T(aa(1):aa(end),6)=1;
[aa bb]=find(T1==7);[aa(1) aa(end)]
T(aa(1):aa(end),7)=1;

% code to train an ensemble of neural networks
clear net_7DoF_PnP4
for i=1:10
    disp(i)
    clear net
    net = patternnet([64 64 64]) ;
    net.performParam.regularization=0.2;
    net.trainParam.epochs=1500;
    net = train(net,N,T','useParallel','yes');
    net_7DoF_PnP4{i}=net;
end
%pretrain_net.layers{1}.transferFcn = 'poslin';
%pretrain_net.layers{2}.transferFcn = 'poslin';
%pretrain_net.layers{3}.transferFcn = 'poslin';
%net1.divideParam.trainRatio
net_7DoF_PnP_2022Feb_2norm = net_7DoF_PnP3;
genFunction(net_7DoF_PnP_2022Feb_2norm,'MLP_7DoF_PnP_2022Feb_2norm')
save net_7DoF_PnP_2022Feb_2norm net_7DoF_PnP_2022Feb_2norm
% now train a NN using layers
% organize the data

net_7DoF_PnP4_ensemble = net_7DoF_PnP4
save net_7DoF_PnP4_ensemble net_7DoF_PnP4_ensemble 


% using custom layers
layers = [ ...
    featureInputLayer(96)
    fullyConnectedLayer(96)
    layerNormalizationLayer
    leakyReluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(96)
    layerNormalizationLayer
    eluReluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(64)
    layerNormalizationLayer
    sigmoidLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer
    ];



X = N;
Y=categorical(T1);
idx = randperm(length(Y),round(0.8*length(Y)));
Xtrain = X(:,idx);
Ytrain = Y(idx);
I = ones(length(Y),1);
I(idx)=0;
idx1 = find(I~=0);
Xtest = X(:,idx1);
Ytest = Y(idx1);



%'ValidationData',{XTest,YTest},...
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',15, ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',256,...
    'ValidationFrequency',30,...
    'ValidationPatience',5,...
    'ExecutionEnvironment','gpu',...
    'ValidationData',{Xtest',Ytest});

% build the classifier
net = trainNetwork(Xtrain',Ytrain,layers,options);
net_mlp_7DoF_Feb2022 = net;
save net_mlp_7DoF_Feb2022 net_mlp_7DoF_Feb2022
save net net
genFunction(net,'MLP_PreTrained_7DoF_PnP4_New')

% get the data and test on held out day
D1i=[];
D2i=[];
D3i=[];
D4i=[];
D5i=[];
D6i=[];
D7i=[];
for i=length(Data.Day)
    disp(i)
    tmp=Data.Day(i).imagined_data;
    D1i = [D1i cell2mat(tmp{1}')];
    D2i = [D2i cell2mat(tmp{2}')];
    D3i = [D3i cell2mat(tmp{3}')];
    D4i = [D4i cell2mat(tmp{4}')];
    D5i = [D5i cell2mat(tmp{5}')];
    D6i = [D6i cell2mat(tmp{6}')];
    D7i = [D7i cell2mat(tmp{7}')];
end


D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
for i=length(Data.Day)
    disp(i)
    tmp=Data.Day(i).online_data;
    D1 = [D1 cell2mat(tmp{1}')];
    D2 = [D2 cell2mat(tmp{2}')];
    D3 = [D3 cell2mat(tmp{3}')];
    D4 = [D4 cell2mat(tmp{4}')];
    D5 = [D5 cell2mat(tmp{5}')];
    D6 = [D6 cell2mat(tmp{6}')];
    D7 = [D7 cell2mat(tmp{7}')];
end


% test the classifier output on imagined data
condn_data{1}=[D1(:,201:end)]'; % right hand
condn_data{2}= [ D2(:,201:end)]'; % both feet
condn_data{3}=[ D3(:,201:end)]'; % left hand
condn_data{4}=[ D4(:,201:end)]'; % head
condn_data{5}=[D5(:,201:end)]'; % mime up
condn_data{6}=[ D6(:,201:end)]'; % tongue in
condn_data{7}=[ D7(:,201:end)]'; % both hands
acc_layers=zeros(7);
acc_net=zeros(7);
for i=1:length(condn_data)
    disp(i)
    X = condn_data{i};
    for j=1:size(X,1)
        res=predict(net,X(j,:));
        [aa bb]=max(res);
        acc_layers(i,bb) = acc_layers(i,bb)+1;
        
        res = net1(X(j,:)');
        [aa bb]=max(res);
        acc_net(i,bb) = acc_net(i,bb)+1;
    end
end
for i=1:7
    acc_net(i,:) = acc_net(i,:)./sum(acc_net(i,:));
    acc_layers(i,:) = acc_layers(i,:)./sum(acc_layers(i,:));
end
[diag(acc_layers)  diag(acc_net)]
mean(ans)

% adapt the network to imagined data and test on online data
net1=pretrain_net;
A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};
E = condn_data{5};
F = condn_data{6};
G = condn_data{7};
clear N
N = [A' B' C' D' E' F' G'];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1)];
T = zeros(size(T1,1),7);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;
[aa bb]=find(T1==5);[aa(1) aa(end)]
T(aa(1):aa(end),5)=1;
[aa bb]=find(T1==6);[aa(1) aa(end)]
T(aa(1):aa(end),6)=1;
[aa bb]=find(T1==7);[aa(1) aa(end)]
T(aa(1):aa(end),7)=1;
% code to train a neural network
net1 = train(net1,N,T','UseGPU','yes');
% net1.divideParam.trainRatio = 0.8;
% net1.divideParam.valRatio = 0.1;
% net1.divideParam.testRatio = 0.1;

% update the deep layernetowkr now
X = N;
Y=categorical(T1);
idx = randperm(length(Y),round(0.8*length(Y)));
Xtrain = X(:,idx);
Ytrain = Y(idx);
I = ones(length(Y),1);
I(idx)=0;
idx1 = find(I~=0);
Xtest = X(:,idx);
Ytest = Y(idx);

% retrain the deep network
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',20, ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',64,...
    'ValidationFrequency',50,...
    'ExecutionEnvironment','gpu',...
    'ValidationData',{Xtest',Ytest});
net = trainNetwork(Xtrain',Ytrain,net.Layers,options);



% test the data on a held out day
folderpath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220304\RealRobotBatch';
files=findfiles('mat',folderpath,1)';

acc=zeros(7,8);
mlp_acc=zeros(7,8);
for i=1:length(files)
    disp(i)
    if regexp(files{i},'BCI_Fixed')
        try
            load(files{i})
            files_loaded=1;
        catch
            files_loaded=0;
        end
        
        if files_loaded
            idx = find(TrialData.TaskState==3);
            feat = (TrialData.SmoothedNeuralFeatures);
            temp = cell2mat(feat(idx));
            
            % pooling
            new_temp=[];
            [xx yy] = size(TrialData.Params.ChMap);
            for k=1:size(temp,2)
                tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
                tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
                tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
                pooled_data=[];
                for i=1:2:xx
                    for j=1:2:yy
                        delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                        beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                        hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                        pooled_data = [pooled_data; delta; beta ;hg];
                    end
                end
                new_temp= [new_temp pooled_data];
            end
            temp=new_temp;
            
            decodes=[];
            for j=1:size(temp,2)
                temp(:,j) = temp(:,j)./norm(temp(:,j));
                %act = predict(net_7DoF_PnP_2022Feb_norm2,temp(:,j)');
                act = net_7DoF_PnP_2022Feb_norm2(temp(:,j))';
                [aa bb]=max(act);
                decodes = [decodes; act];
                if aa>=0.5
                    acc(TrialData.TargetID,bb)=acc(TrialData.TargetID,bb)+1;
                else
                    acc(TrialData.TargetID,8)=acc(TrialData.TargetID,8)+1;
                end
            end
            
            decodes = TrialData.ClickerState;
            for j=1:length(decodes)
                if decodes(j)==0
                    mlp_acc(TrialData.TargetID,8) = mlp_acc(TrialData.TargetID,8)+1;
                else
                    mlp_acc(TrialData.TargetID,decodes(j)) = ...
                        mlp_acc(TrialData.TargetID,decodes(j))+1;
                end
                
            end
        end
    end
end

for i=1:size(mlp_acc,1)
    mlp_acc(i,:) = mlp_acc(i,:)/sum(mlp_acc(i,:));
    acc(i,:) = acc(i,:)/sum(acc(i,:));
end

[(diag(acc)) (diag(mlp_acc))]



% test the data on a held out day with mode filtering


%% (MAIN) UPDATING HISTORICAL DECODER FEB 2022 WITH NEW TRAINING DATA

clc;clear
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
root_path=pwd;
%load net_7DoF_PnP_2022Feb_2norm

% get the training data from March
foldernames = {'20220304','20220309','20220311','20220316','20220318','20220323',...
    '20220325','20220330','20220420','20220422','20220427','20220429'};


arrow_files={};
robot_files={};
k=1;jj=1;kk=1;
for i=1:length(foldernames)
%     %arrow files    
%     folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
%     D=dir(folderpath);
%     online_files_temp=[];
%     for j=3:length(D)        
%         filepath1=fullfile(folderpath,D(j).name,'BCI_Fixed');
%         if exist(filepath1)
%             online_files_temp = [online_files_temp;findfiles('',filepath1)'];
%         end
%     end
%     
%     if i==6
%         online_files_temp = online_files_temp([1:21 23:end]);
%     end
%     
%     if ~isempty(online_files_temp)
%         arrow_files{jj} = online_files_temp;jj=jj+1;
%     end    
    
    %robot files
    folderpath = fullfile(root_path, foldernames{i},'RealRobotBatch');
    D=dir(folderpath);
    online_files_temp=[];
    for j=3:length(D)        
        filepath1=fullfile(folderpath,D(j).name,'BCI_Fixed');
        if exist(filepath1)
            online_files_temp = [online_files_temp;findfiles('',filepath1)'];
        end
    end
    
    if i==6
        online_files_temp = online_files_temp([1:21 23:end]);
    end
    
    if ~isempty(online_files_temp)
        robot_files{kk} = online_files_temp;kk=kk+1;
    end    
end

% load the online data files
online_files = [arrow_files robot_files];
Data=[];
files_not_loaded_fixed=[];
for iter=1:length(online_files)
    tmp_files = online_files{iter};
    D1={};
    D2={};
    D3={};
    D4={};
    D5={};
    D6={};
    D7={};
    for ii=1:length(tmp_files)
        disp(ii)
        
        indicator=1;
        try
            load(tmp_files{ii});
        catch ME
            warning('Not able to load file, skipping to next')
            indicator = 0;
            files_not_loaded_fixed=[files_not_loaded_fixed;tmp_files{ii}];
        end
        
        if indicator
            
            features  = TrialData.SmoothedNeuralFeatures;
            kinax = TrialData.TaskState;
            kinax = [find(kinax==3)];
            temp = cell2mat(features(kinax));
            temp = temp(:,5:end);
            
            %get the pooled data
            new_temp=[];
            [xx yy] = size(TrialData.Params.ChMap);
            for k=1:size(temp,2)
                tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
                tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
                tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
                pooled_data=[];
                for i=1:2:xx
                    for j=1:2:yy
                        delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                        beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                        hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                        pooled_data = [pooled_data; delta; beta ;hg];
                    end
                end
                new_temp= [new_temp pooled_data];
            end
            temp_data=new_temp;
            
            
            % no baseline
            temp = temp_data;
            
            %temp = temp(769:896,:);
            if TrialData.TargetID == 1
                D1=cat(2,D1,temp);
            elseif TrialData.TargetID == 2
                D2=cat(2,D2,temp);
            elseif TrialData.TargetID == 3
                D3=cat(2,D3,temp);
            elseif TrialData.TargetID == 4
                D4=cat(2,D4,temp);
            elseif TrialData.TargetID == 5
                D5=cat(2,D5,temp);
            elseif TrialData.TargetID == 6
                D6=cat(2,D6,temp);
            elseif TrialData.TargetID == 7
                D7=cat(2,D7,temp);
            end
        end
    end
    
    clear condn_data
    %idx = [1:128];
    condn_data{1}=[D1 ]'; % right hand
    condn_data{2}= [D2]'; % both feet
    condn_data{3}=[D3]'; % left hand
    condn_data{4}=[D4]'; % head
    condn_data{5}=[D5]'; % mime up
    condn_data{6}=[D6]'; % tongue in
    condn_data{7}=[D7]'; % squeeze both hands
    
    Data.Day(iter).online_data = condn_data;
end


D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
for i=1:length(Data.Day)
    disp(i)
    tmp=Data.Day(i).online_data;
    D1 = [D1 cell2mat(tmp{1}')];
    D2 = [D2 cell2mat(tmp{2}')];
    D3 = [D3 cell2mat(tmp{3}')];
    D4 = [D4 cell2mat(tmp{4}')];
    D5 = [D5 cell2mat(tmp{5}')];
    D6 = [D6 cell2mat(tmp{6}')];
    D7 = [D7 cell2mat(tmp{7}')];
end


% build the decoder; compare ML using patternet and using layers
clear condn_data
% combing both onlien plus offline
%idx=641;
idx = [1:96];
condn_data{1}=[D1(idx,:) ]'; % right hand
condn_data{2}= [D2(idx,:) ]'; % both feet
condn_data{3}=[D3(idx,:) ]'; % left hand
condn_data{4}=[D4(idx,:) ]'; % head
condn_data{5}=[D5(idx,:) ]'; % mime up
condn_data{6}=[D6(idx,:) ]'; % tongue in
condn_data{7}=[D7(idx,:) ]'; % both hands

% 2norm
for i=1:length(condn_data)
   tmp = condn_data{i}; 
   for j=1:size(tmp,1)
       tmp(j,:) = tmp(j,:)./norm(tmp(j,:));
   end
   condn_data{i}=tmp;
end

A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};
E = condn_data{5};
F = condn_data{6};
G = condn_data{7};


clear N
N = [A' B' C' D' E' F' G'];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1)];

T = zeros(size(T1,1),7);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;
[aa bb]=find(T1==5);[aa(1) aa(end)]
T(aa(1):aa(end),5)=1;
[aa bb]=find(T1==6);[aa(1) aa(end)]
T(aa(1):aa(end),6)=1;
[aa bb]=find(T1==7);[aa(1) aa(end)]
T(aa(1):aa(end),7)=1;

% updating the pretrained decoder
% net_7DoF_PnP_2022Feb_2norm = train(net_7DoF_PnP_2022Feb_2norm,...
%     N,T','useParallel','yes');

% updating the ensemble decoder
load net_7DoF_PnP4_ensemble
clear net_7DoF_PnP4_ensemble_batch
for i=1:length(net_7DoF_PnP4_ensemble)
    disp(i)
    net = net_7DoF_PnP4_ensemble{i};
    net = train(net,N,T','useParallel','yes');
    net_7DoF_PnP4_ensemble_batch{i} = net;
end

save net_7DoF_PnP4_ensemble_batch net_7DoF_PnP4_ensemble_batch

% 
% % save the data 
% net_7DoF_PnP_2022Mar_2norm = net_7DoF_PnP_2022Feb_2norm;
% genFunction(net_7DoF_PnP_2022Mar_2norm,'MLP_7DoF_PnP_2022Mar_2norm')
% save net_7DoF_PnP_2022Mar_2norm net_7DoF_PnP_2022Mar_2norm



%% (MAIN) checking to see if the data from problematic sessions are out of range


clc;clear

%%%%% first get the distribution of the data from the good days
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
load 7DoF_Data_SpatioTemp_AcrossDays_20220224
%load Data_SpatioTemp_AcrossDays
%load Data_SpatioTemp_AcrossDays_9282021
%load 7DoF_Data_Training

% get the training data
D1i=[];
D2i=[];
D3i=[];
D4i=[];
D5i=[];
D6i=[];
D7i=[];
for i=1:15%length(Data.Day)
    disp(i)
    tmp=Data.Day(i).imagined_data;
    D1i = [D1i cell2mat(tmp{1}')];
    D2i = [D2i cell2mat(tmp{2}')];
    D3i = [D3i cell2mat(tmp{3}')];
    D4i = [D4i cell2mat(tmp{4}')];
    D5i = [D5i cell2mat(tmp{5}')];
    D6i = [D6i cell2mat(tmp{6}')];
    D7i = [D7i cell2mat(tmp{7}')];
end

D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
for i=1:length(Data.Day)
    disp(i)
    tmp=Data.Day(i).online_data;
    D1 = [D1 cell2mat(tmp{1}')];
    D2 = [D2 cell2mat(tmp{2}')];
    D3 = [D3 cell2mat(tmp{3}')];
    D4 = [D4 cell2mat(tmp{4}')];
    D5 = [D5 cell2mat(tmp{5}')];
    D6 = [D6 cell2mat(tmp{6}')];
    D7 = [D7 cell2mat(tmp{7}')];
end


% build the decoder; compare ML using patternet and using layers
clear condn_data
% combing both onlien plus offline
%idx=641;
idx = [1:96];
condn_data{1}=[D1(idx,:) D1i]'; % right hand
condn_data{2}= [D2(idx,:) D2i]'; % both feet
condn_data{3}=[D3(idx,:) D3i]'; % left hand
condn_data{4}=[D4(idx,:) D4i]'; % head
condn_data{5}=[D5(idx,:) D5i]'; % mime up
condn_data{6}=[D6(idx,:) D6i]'; % tongue in
condn_data{7}=[D7(idx,:) D7i]'; % both hands

% 2norm
for i=1:length(condn_data)
   tmp = condn_data{i}; 
   for j=1:size(tmp,1)
       tmp(j,:) = tmp(j,:)./norm(tmp(j,:));
   end
   condn_data{i}=tmp;
end

distribution_features=[];
for i=1:length(condn_data)
   tmp=condn_data{i}; 
   m = mean(tmp,1);
   s = std(tmp,1);
   temp = [m+s; m-s]; 
   distribution_features(i,:,:)=temp; 
end

save distribution_features distribution_features -v7.3


% now compare it to a held out day 
clear
load distribution_features

foldernames = {'20220427'};
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=7:length(D)
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        files = [files;findfiles('',filepath)'];
    end
end


% load the online data files
files_not_loaded_fixed=[];
D1=[];D2=[];D3=[];D4=[];D5=[];D6=[];D7=[];
for iter=1:length(files)
    disp(iter)
    
    indicator=1;
    try
        load(files{iter});
    catch ME
        warning('Not able to load file, skipping to next')
        indicator = 0;
        files_not_loaded_fixed=[files_not_loaded_fixed;files{iter}];
    end
    
    if indicator
        
        features  = TrialData.SmoothedNeuralFeatures;
        kinax = TrialData.TaskState;
        kinax = [find(kinax==3)];
        temp = cell2mat(features(kinax));
        
        %get the pooled data
        new_temp=[];
        [xx yy] = size(TrialData.Params.ChMap);
        for k=1:size(temp,2)
            tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
            tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
            tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
            pooled_data=[];
            for i=1:2:xx
                for j=1:2:yy
                    delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                    beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                    hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                    pooled_data = [pooled_data; delta; beta ;hg];
                end
            end
            new_temp= [new_temp pooled_data];
        end
        temp_data=new_temp;
        
        temp = temp_data;
        
        %temp = temp(769:896,:);
        if TrialData.TargetID == 1
            D1=cat(2,D1,temp);
        elseif TrialData.TargetID == 2
            D2=cat(2,D2,temp);
        elseif TrialData.TargetID == 3
            D3=cat(2,D3,temp);
        elseif TrialData.TargetID == 4
            D4=cat(2,D4,temp);
        elseif TrialData.TargetID == 5
            D5=cat(2,D5,temp);
        elseif TrialData.TargetID == 6
            D6=cat(2,D6,temp);
        elseif TrialData.TargetID == 7
            D7=cat(2,D7,temp);
        end
    end    
end

condn_data{1}=[ D1]'; % right hand
condn_data{2}= [ D2]'; % both feet
condn_data{3}=[ D3]'; % left hand
condn_data{4}=[ D4]'; % head
condn_data{5}=[ D5]'; % mime up
condn_data{6}=[ D6]'; % tongue in
condn_data{7}=[ D7]'; % both hands



% 2norm
for i=1:length(condn_data)
   tmp = condn_data{i}; 
   for j=1:size(tmp,1)
       tmp(j,:) = tmp(j,:)./norm(tmp(j,:));
   end
   condn_data{i}=tmp;
end

figure;
for i=1:length(condn_data)
    subplot(4,2,i)
    tmp=condn_data{i};
    m = mean(tmp,1);
    s= std(tmp,1);
    %figure;
    hold on
    plot(m+s,'-k')
    plot(squeeze(distribution_features(i,1,:)),'-r')
    plot(m-s,'-k')
    plot(squeeze(distribution_features(i,2,:)),'-r')
    title(['Target ' num2str(i)])
    set(gcf,'Color','w')
    xlabel('Feature')
    ylabel('Norm value')
    if i== 7
        legend({'20220427','Historical Data'})
    end
    ylim([-0.4 .4])
end

%% GETTING 7DOF DATA FOR JENSEN FOR TESTING META TRANSFER
% split into erp and online on a day to day basis
% get the pooled data and feed it to jason
% THIS IS A POSSIBLE REPEAT OF ABOVE CELLS


clc;clear
close all
root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';

% for only 6 DoF original:
%foldernames = {'20210526','20210528','20210602','20210609_pm','20210611'};

foldernames = {'20210615','20210616','20210623','20210625','20210630','20210702',...
    '20210707','20210716','20210728','20210804','20210806','20210813','20210818',...
    '20210825','20210827','20210901','20210903','20210910','20210917','20210929',...
    '20211001','20211006','20211008','20211013','20211020','20211022','20211027','20211029',...
    '20211103','20211105','20211117','20211119','20220126','20220128','20220202',...
    '20220204','20220209','20220211','20220216','20220218','20220223','20220225','20220309',...
    '20220311','20220316','20220325','20220330'};
cd(root_path)

imag_files={};
online_files={};
k=1;jj=1;
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    imag_files_temp=[];
    online_files_temp=[];

    if i==17
        D = D([1:2 3:6]);
    elseif i==19
        D = D([1:2 5:10]);
    elseif i==29
        D = D([1:2 8:9]);
    end

    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'Imagined');
        if exist(filepath)
            imag_files_temp = [imag_files_temp;findfiles('',filepath)'];
        end
        filepath1=fullfile(folderpath,D(j).name,'BCI_Fixed');
        if exist(filepath1)
            online_files_temp = [online_files_temp;findfiles('',filepath1)'];
        end
    end
    if ~isempty(imag_files_temp)
        imag_files{k} = imag_files_temp;k=k+1;
    end
    if ~isempty(online_files_temp)
        online_files{jj} = online_files_temp;jj=jj+1;
    end
    %     imag_files{i} = imag_files_temp;
    %     online_files{i} = online_files_temp;
end

% load the imagined data files
Data=[];
files_not_loaded=[];
for iter=1:length(imag_files)
    disp(iter/length(imag_files))
    tmp_files = imag_files{iter};
    D1={};
    D2={};
    D3={};
    D4={};
    D5={};
    D6={};
    D7={};
    for ii=1:length(tmp_files)
        %disp(ii)

        indicator=1;
        try
            load(tmp_files{ii});
        catch ME
            warning('Not able to load file, skipping to next')
            indicator = 0;
            files_not_loaded=[files_not_loaded;tmp_files{ii}];
        end

        if indicator
            features  = TrialData.SmoothedNeuralFeatures;
            kinax = TrialData.TaskState;
            kinax = [find(kinax==3)];
            temp = cell2mat(features(kinax));

            %get the pooled data
            new_temp=[];
            [xx yy] = size(TrialData.Params.ChMap);
            for k=1:size(temp,2)
                tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
                tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
                tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
                tmp4 = temp(641:768,k);tmp4 = tmp4(TrialData.Params.ChMap);%low gamma
                pooled_data=[];
                for i=1:2:xx
                    for j=1:2:yy
                        delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                        beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                        hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                        lg = (tmp4(i:i+1,j:j+1));lg=mean(lg(:));
                        pooled_data = [pooled_data; delta; beta ;lg;hg];
                    end
                end
                new_temp= [new_temp pooled_data];
            end
            temp_data=new_temp;


            %             % get the pooled data for state 1
            %             kinax = TrialData.TaskState;
            %             kinax = [find(kinax==1)];
            %             temp = cell2mat(features(kinax));
            %
            %             %get the pooled data
            %             new_temp=[];
            %             [xx yy] = size(TrialData.Params.ChMap);
            %             for k=1:size(temp,2)
            %                 tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
            %                 tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
            %                 tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
            %                 pooled_data=[];
            %                 for i=1:2:xx
            %                     for j=1:2:yy
            %                         delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
            %                         beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
            %                         hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
            %                         pooled_data = [pooled_data; delta; beta ;hg];
            %                     end
            %                 end
            %                 new_temp= [new_temp pooled_data];
            %             end
            %             temp_bl = new_temp;
            %
            % baseline the data
            %temp = temp_data - mean(temp_bl,2);


            % no baseline
            temp = temp_data;


            %temp = temp(769:896,:);
            if TrialData.TargetID == 1
                D1=cat(2,D1,temp);
            elseif TrialData.TargetID == 2
                D2=cat(2,D2,temp);
            elseif TrialData.TargetID == 3
                D3=cat(2,D3,temp);
            elseif TrialData.TargetID == 4
                D4=cat(2,D4,temp);
            elseif TrialData.TargetID == 5
                D5=cat(2,D5,temp);
            elseif TrialData.TargetID == 6
                D6=cat(2,D6,temp);
            elseif TrialData.TargetID == 7
                D7=cat(2,D7,temp);
            end
        end
    end

    clear condn_data
    %idx = [1:128];
    condn_data{1}=[D1 ]'; % right hand
    condn_data{2}= [D2]'; % both feet
    condn_data{3}=[D3]'; % left hand
    condn_data{4}=[D4]'; % head
    condn_data{5}=[D5]'; % mime up
    condn_data{6}=[D6]'; % tongue in
    condn_data{7}=[D7]'; % squeeze both hands

    Data.Day(iter).imagined_data = condn_data;
end

% load the online data files
files_not_loaded_fixed=[];
for iter=1:length(online_files)
    disp(iter/length(online_files))
    tmp_files = online_files{iter};
    D1={};
    D2={};
    D3={};
    D4={};
    D5={};
    D6={};
    D7={};
    for ii=1:length(tmp_files)
        %disp(ii)

        indicator=1;
        try
            load(tmp_files{ii});
        catch ME
            warning('Not able to load file, skipping to next')
            indicator = 0;
            files_not_loaded_fixed=[files_not_loaded_fixed;tmp_files{ii}];
        end

        if indicator

            features  = TrialData.SmoothedNeuralFeatures;
            kinax = TrialData.TaskState;
            kinax = [find(kinax==3)];
            temp = cell2mat(features(kinax));

            %get the pooled data
            new_temp=[];
            [xx yy] = size(TrialData.Params.ChMap);
            for k=1:size(temp,2)
                tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
                tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
                tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
                tmp4 = temp(641:768,k);tmp4 = tmp4(TrialData.Params.ChMap);%low gamma
                pooled_data=[];
                for i=1:2:xx
                    for j=1:2:yy
                        delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                        beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                        hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                        lg = (tmp4(i:i+1,j:j+1));lg=mean(lg(:));
                        pooled_data = [pooled_data; delta; beta ;lg;hg];
                    end
                end
                new_temp= [new_temp pooled_data];
            end
            temp_data=new_temp;


            %             % get the pooled data for state 1
            %             kinax = TrialData.TaskState;
            %             kinax = [find(kinax==1)];
            %             temp = cell2mat(features(kinax));
            %
            %             %get the pooled data
            %             new_temp=[];
            %             [xx yy] = size(TrialData.Params.ChMap);
            %             for k=1:size(temp,2)
            %                 tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
            %                 tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
            %                 tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
            %                 pooled_data=[];
            %                 for i=1:2:xx
            %                     for j=1:2:yy
            %                         delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
            %                         beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
            %                         hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
            %                         pooled_data = [pooled_data; delta; beta ;hg];
            %                     end
            %                 end
            %                 new_temp= [new_temp pooled_data];
            %             end
            %             temp_bl = new_temp;

            % baseline the data
            %temp = temp_data - mean(temp_bl,2);

            % no baseline
            temp = temp_data;

            %temp = temp(769:896,:);
            if TrialData.TargetID == 1
                D1=cat(2,D1,temp);
            elseif TrialData.TargetID == 2
                D2=cat(2,D2,temp);
            elseif TrialData.TargetID == 3
                D3=cat(2,D3,temp);
            elseif TrialData.TargetID == 4
                D4=cat(2,D4,temp);
            elseif TrialData.TargetID == 5
                D5=cat(2,D5,temp);
            elseif TrialData.TargetID == 6
                D6=cat(2,D6,temp);
            elseif TrialData.TargetID == 7
                D7=cat(2,D7,temp);
            end
        end
    end

    clear condn_data
    %idx = [1:128];
    condn_data{1}=[D1 ]'; % right hand
    condn_data{2}= [D2]'; % both feet
    condn_data{3}=[D3]'; % left hand
    condn_data{4}=[D4]'; % head
    condn_data{5}=[D5]'; % mime up
    condn_data{6}=[D6]'; % tongue in
    condn_data{7}=[D7]'; % squeeze both hands

    Data.Day(iter).online_data = condn_data;
end

cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
save 7DoF_Data_SpatioTemp_AcrossDays_lg_20220712 Data -v7.3

condn_data={};
for i=1:7
    data=[];
    for j=8
        tmp = Data.Day(j).online_data;
        tmp = tmp{i};
        data = [data;tmp];
    end
    condn_data{i} = data;
end



% train an autoencoder and see how it goes
X=[];
idx=[];
for i=1:length(condn_data)
    tmp = condn_data{i};
    X = [X;tmp];
    idx = [idx ;i*ones(size(tmp,1),1)];
end
% artfiact correct
m = mean(X);
for i=1:size(X,2)
    [aa bb] = find(abs(X(:,i))>=4);
    X(aa,i) = m(i);
end
X=X';
size(X)


% X7 is day 7
% X6 is day 6
% need to project data from day 7 onto space spanned by day 6 as decoder is
% built on day 6

jj = randperm(size(X7,1),size(X6,1));
X7a = X7(jj,:);
A = X6\X7a;
X = X7*A;
X=X';

%
% autoenc = trainAutoencoder(X,5,...
%      'EncoderTransferFunction','satlin',...
%         'DecoderTransferFunction','satlin',...
%         'L2WeightRegularization',0.1,...
%         'SparsityRegularization',4,...
%         'SparsityProportion',0.15,...
%         'ScaleData',0,...
%         'MaxEpochs',1000,...
%         'UseGPU',1);
%
% XReconstructed = predict(autoenc,X);
% mseError = mse(X-XReconstructed)
%
% Z = encode(autoenc,X);
% [coeff,score,latent]=pca(Z');
% Z = score';
% cmap = parula(7);
% figure;hold on
% for i=1:7
%     idxx = find(idx==i);
%     plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:));
% end

latent_Z=[];
for loop=1:10


    % using custom layers
    layers = [ ...
        featureInputLayer(96)
        fullyConnectedLayer(64)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(32)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(8)
        batchNormalizationLayer
        reluLayer('Name','autoencoder')
        fullyConnectedLayer(32)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(64)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(96)
        regressionLayer];


    %'ValidationData',{XTest,YTest},...
    options = trainingOptions('sgdm', ...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',500, ...
        'Shuffle','every-epoch', ...
        'Verbose',true, ...
        'Plots','training-progress',...
        'MiniBatchSize',4096,...
        'ValidationFrequency',30,...
        'L2Regularization',1e-4,...
        'ExecutionEnvironment','gpu');

    % build the autoencoder
    net = trainNetwork(X',X',layers,options);

    % MSE
    tmp = predict(net,X');
    mseError = mse(X-tmp')
    %
    % % now get activations in deepest layer
    Z = activations(net,X','autoencoder');
    %
    %     % plotting
    %     [coeff,score,latent]=pca(Z');
    %     Z = score';
    %     cmap = parula(7);
    %     figure;hold on
    %     for i=1:7
    %         idxx = find(idx==i);
    %         plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:));
    %     end
    %     title(num2str(loop))
    latent_Z(loop,:,:) = Z;
end

Z = squeeze(mean(latent_Z,1));
[coeff,score,latent]=pca(Z');
Z = score';
cmap = parula(7);
figure;hold on
for i=1:7
    idxx = find(idx==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:));
end


%
%Z = activations(net,X','autoencoder');
Z = squeeze(mean(latent_Z,1));
Y = tsne(Z','Algorithm','exact','Standardize',false,'Perplexity',30,'NumDimensions',3,...
    'Exaggeration',10);
Y=Y';
cmap = parula(7);
figure;hold on
for i=1:7
    idxx = find(idx==i);
    plot3(Y(1,idxx),Y(2,idxx),Y(3,idxx),'.','color',cmap(i,:));
    %plot(Y(1,idxx),Y(2,idxx),'.','color',cmap(i,:));
end


layers1=layers;
layers1=layers1(1:10);
layers2 = [layers1
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer]



%'ValidationData',{XTest,YTest},...
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',512,...
    'ValidationFrequency',30,...
    'L2Regularization',1e-4,...
    'ExecutionEnvironment','auto');

Y=categorical(idx);
% build the classifier
net = trainNetwork(X',Y,layers2,options);


% % now get activations in deepest layer
Z = activations(net,X','autoencoder');

% plotting
[coeff,score,latent]=pca(Z');
Z = score';
cmap = parula(7);
figure;hold on
for i=1:7
    idxx = find(idx==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:));
end


cmap = parula(7);
figure;hold on
for i=1:7
    idxx = find(idx==i);
    plot3(Z(7,idxx),Z(8,idxx),Z(9,idxx),'color',cmap(i,:));
end


%% building a neural decoder with all the online data
%POSSIBLE REPEAT OF ABOVE CELLS
clc;clear

cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
load 7DoF_Data_SpatioTemp_AcrossDays_20220224
%load Data_SpatioTemp_AcrossDays
%load Data_SpatioTemp_AcrossDays_9282021
%load 7DoF_Data_Training

% get the training data
D1i=[];
D2i=[];
D3i=[];
D4i=[];
D5i=[];
D6i=[];
D7i=[];
for i=1:15%length(Data.Day)
    disp(i)
    tmp=Data.Day(i).imagined_data;
    D1i = [D1i cell2mat(tmp{1}')];
    D2i = [D2i cell2mat(tmp{2}')];
    D3i = [D3i cell2mat(tmp{3}')];
    D4i = [D4i cell2mat(tmp{4}')];
    D5i = [D5i cell2mat(tmp{5}')];
    D6i = [D6i cell2mat(tmp{6}')];
    D7i = [D7i cell2mat(tmp{7}')];
end

D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
for i=1:length(Data.Day)
    disp(i)
    tmp=Data.Day(i).online_data;
    D1 = [D1 cell2mat(tmp{1}')];
    D2 = [D2 cell2mat(tmp{2}')];
    D3 = [D3 cell2mat(tmp{3}')];
    D4 = [D4 cell2mat(tmp{4}')];
    D5 = [D5 cell2mat(tmp{5}')];
    D6 = [D6 cell2mat(tmp{6}')];
    D7 = [D7 cell2mat(tmp{7}')];
end


% build the decoder; compare ML using patternet and using layers
clear condn_data
% combing both onlien plus offline
%idx=641;
idx = [1:128];
condn_data{1}=[D1(idx,:) D1i]'; % right hand
condn_data{2}= [D2(idx,:) D2i]'; % both feet
condn_data{3}=[D3(idx,:) D3i]'; % left hand
condn_data{4}=[D4(idx,:) D4i]'; % head
condn_data{5}=[D5(idx,:) D5i]'; % mime up
condn_data{6}=[D6(idx,:) D6i]'; % tongue in
condn_data{7}=[D7(idx,:) D7i]'; % both hands

for i=1:length(condn_data)
    tmp=condn_data{i};
    for j=1:size(tmp,1)
        tmp(j,:) = tmp(j,:)./norm(tmp(j,:));
    end
    condn_data{i}=tmp;
end

A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};
E = condn_data{5};
F = condn_data{6};
G = condn_data{7};


clear N
N = [A' B' C' D' E' F' G'];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1)];

T = zeros(size(T1,1),7);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;
[aa bb]=find(T1==5);[aa(1) aa(end)]
T(aa(1):aa(end),5)=1;
[aa bb]=find(T1==6);[aa(1) aa(end)]
T(aa(1):aa(end),6)=1;
[aa bb]=find(T1==7);[aa(1) aa(end)]
T(aa(1):aa(end),7)=1;

% code to train a neural network
clear net_7DoF_PnP_lg
net_7DoF_PnP_lg = patternnet([64 64 64]) ;
net_7DoF_PnP_lg.performParam.regularization=0.2;
net_7DoF_PnP_lg.trainParam.epochs=1500;
net_7DoF_PnP_lg = train(net_7DoF_PnP_lg,N,T','useGPU','yes');
%pretrain_net.layers{1}.transferFcn = 'poslin';
%pretrain_net.layers{2}.transferFcn = 'poslin';
%pretrain_net.layers{3}.transferFcn = 'poslin';
%net1.divideParam.trainRatio
net_7DoF_PnP_lg = net_7DoF_PnP_lg;
genFunction(net_7DoF_PnP_lg,'MLP_7DoF_PnP_2022July_lg')
save net_7DoF_PnP_lg net_7DoF_PnP_lg
% now train a NN using layers
% organize the data

% using custom layers
layers = [ ...
    featureInputLayer(128)
    fullyConnectedLayer(128)
    layerNormalizationLayer
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(96)
    layerNormalizationLayer
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(96)
    layerNormalizationLayer
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer
    ];



X = N;
Y=categorical(T1);
idx = randperm(length(Y),round(0.8*length(Y)));
Xtrain = X(:,idx);
Ytrain = Y(idx);
I = ones(length(Y),1);
I(idx)=0;
idx1 = find(I~=0);
Xtest = X(:,idx1);
Ytest = Y(idx1);



%'ValidationData',{XTest,YTest},...
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',15, ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',256,...
    'ValidationFrequency',128,...
    'ValidationPatience',5,...
    'ExecutionEnvironment','gpu',...
    'ValidationData',{Xtest',Ytest});

% build the classifier
net = trainNetwork(Xtrain',Ytrain,layers,options);
net_mlp_7DoF_Feb2022 = net;
save net_mlp_7DoF_Feb2022 net_mlp_7DoF_Feb2022
save net net
genFunction(net,'MLP_PreTrained_7DoF_PnP4_New')

% get the data and test on held out day
D1i=[];
D2i=[];
D3i=[];
D4i=[];
D5i=[];
D6i=[];
D7i=[];
for i=length(Data.Day)
    disp(i)
    tmp=Data.Day(i).imagined_data;
    D1i = [D1i cell2mat(tmp{1}')];
    D2i = [D2i cell2mat(tmp{2}')];
    D3i = [D3i cell2mat(tmp{3}')];
    D4i = [D4i cell2mat(tmp{4}')];
    D5i = [D5i cell2mat(tmp{5}')];
    D6i = [D6i cell2mat(tmp{6}')];
    D7i = [D7i cell2mat(tmp{7}')];
end


D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
for i=length(Data.Day)
    disp(i)
    tmp=Data.Day(i).online_data;
    D1 = [D1 cell2mat(tmp{1}')];
    D2 = [D2 cell2mat(tmp{2}')];
    D3 = [D3 cell2mat(tmp{3}')];
    D4 = [D4 cell2mat(tmp{4}')];
    D5 = [D5 cell2mat(tmp{5}')];
    D6 = [D6 cell2mat(tmp{6}')];
    D7 = [D7 cell2mat(tmp{7}')];
end


% test the classifier output on imagined data
condn_data{1}=[D1(:,201:end)]'; % right hand
condn_data{2}= [ D2(:,201:end)]'; % both feet
condn_data{3}=[ D3(:,201:end)]'; % left hand
condn_data{4}=[ D4(:,201:end)]'; % head
condn_data{5}=[D5(:,201:end)]'; % mime up
condn_data{6}=[ D6(:,201:end)]'; % tongue in
condn_data{7}=[ D7(:,201:end)]'; % both hands
acc_layers=zeros(7);
acc_net=zeros(7);
for i=1:length(condn_data)
    disp(i)
    X = condn_data{i};
    for j=1:size(X,1)
        res=predict(net,X(j,:));
        [aa bb]=max(res);
        acc_layers(i,bb) = acc_layers(i,bb)+1;

        res = net1(X(j,:)');
        [aa bb]=max(res);
        acc_net(i,bb) = acc_net(i,bb)+1;
    end
end
for i=1:7
    acc_net(i,:) = acc_net(i,:)./sum(acc_net(i,:));
    acc_layers(i,:) = acc_layers(i,:)./sum(acc_layers(i,:));
end
[diag(acc_layers)  diag(acc_net)]
mean(ans)

% adapt the network to imagined data and test on online data
net1=pretrain_net;
A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};
E = condn_data{5};
F = condn_data{6};
G = condn_data{7};
clear N
N = [A' B' C' D' E' F' G'];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1)];
T = zeros(size(T1,1),7);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;
[aa bb]=find(T1==5);[aa(1) aa(end)]
T(aa(1):aa(end),5)=1;
[aa bb]=find(T1==6);[aa(1) aa(end)]
T(aa(1):aa(end),6)=1;
[aa bb]=find(T1==7);[aa(1) aa(end)]
T(aa(1):aa(end),7)=1;
% code to train a neural network
net1 = train(net1,N,T','UseGPU','yes');
% net1.divideParam.trainRatio = 0.8;
% net1.divideParam.valRatio = 0.1;
% net1.divideParam.testRatio = 0.1;

% update the deep layernetowkr now
X = N;
Y=categorical(T1);
idx = randperm(length(Y),round(0.8*length(Y)));
Xtrain = X(:,idx);
Ytrain = Y(idx);
I = ones(length(Y),1);
I(idx)=0;
idx1 = find(I~=0);
Xtest = X(:,idx);
Ytest = Y(idx);

% retrain the deep network
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',20, ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',64,...
    'ValidationFrequency',50,...
    'ExecutionEnvironment','gpu',...
    'ValidationData',{Xtest',Ytest});
net = trainNetwork(Xtrain',Ytrain,net.Layers,options);



% test the data on a held out day
folderpath='E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20211008\Robot3DArrow';
files=findfiles('mat',folderpath,1)';

acc=zeros(7,8);
mlp_acc=zeros(7,8);
for i=1:length(files)
    disp(i)
    if regexp(files{i},'BCI_Fixed')
        load(files{i})
        idx = find(TrialData.TaskState==3);
        feat = (TrialData.SmoothedNeuralFeatures);
        temp = cell2mat(feat(idx));

        % pooling
        new_temp=[];
        [xx yy] = size(TrialData.Params.ChMap);
        for k=1:size(temp,2)
            tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
            tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
            tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
            pooled_data=[];
            for i=1:2:xx
                for j=1:2:yy
                    delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                    beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                    hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                    pooled_data = [pooled_data; delta; beta ;hg];
                end
            end
            new_temp= [new_temp pooled_data];
        end
        temp=new_temp;

        decodes=[];
        for j=1:size(temp,2)
            act = predict(net,temp(:,j)');
            [aa bb]=max(act);
            decodes = [decodes; act];
            %             if aa>=0.4
            %                 acc(TrialData.TargetID,bb)=acc(TrialData.TargetID,bb)+1;
            %             else
            %                 acc(TrialData.TargetID,8)=acc(TrialData.TargetID,8)+1;
            %             end
        end

        decodes = TrialData.ClickerState;
        for j=1:length(decodes)
            if decodes(j)==0
                mlp_acc(TrialData.TargetID,8) = mlp_acc(TrialData.TargetID,8)+1;
            else
                mlp_acc(TrialData.TargetID,decodes(j)) = ...
                    mlp_acc(TrialData.TargetID,decodes(j))+1;
            end

        end
    end
end

for i=1:size(mlp_acc,1)
    mlp_acc(i,:) = mlp_acc(i,:)/sum(mlp_acc(i,:));
    acc(i,:) = acc(i,:)/sum(acc(i,:));
end

[(diag(acc)) (diag(mlp_acc))]



%% BUILDING AN AUTOENCODER WITH ALL THE META LEARNING DATA
%POSSIBLE REPEAT OF ABOVE CELLS

clc;clear

cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
load Data_SpatioTemp_AcrossDays
%load Data_SpatioTemp_AcrossDays_RelativeToState1
% get the training data
D1i=[];
D2i=[];
D3i=[];
D4i=[];
D5i=[];
D6i=[];
D7i=[];
for i=1:length(Data.Day)
    disp(i)
    tmp=Data.Day(i).imagined_data;
    D1i = [D1i cell2mat(tmp{1}')];
    D2i = [D2i cell2mat(tmp{2}')];
    D3i = [D3i cell2mat(tmp{3}')];
    D4i = [D4i cell2mat(tmp{4}')];
    D5i = [D5i cell2mat(tmp{5}')];
    D6i = [D6i cell2mat(tmp{6}')];
    D7i = [D7i cell2mat(tmp{7}')];
end

D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
for i=1:length(Data.Day)
    disp(i)
    tmp=Data.Day(i).online_data;
    D1 = [D1 cell2mat(tmp{1}')];
    D2 = [D2 cell2mat(tmp{2}')];
    D3 = [D3 cell2mat(tmp{3}')];
    D4 = [D4 cell2mat(tmp{4}')];
    D5 = [D5 cell2mat(tmp{5}')];
    D6 = [D6 cell2mat(tmp{6}')];
    D7 = [D7 cell2mat(tmp{7}')];
end


% build the decoder; compare ML using patternet and using layers
clear condn_data
% combing both onlien plus offline
%idx=641;
idx = [65:96];
condn_data{1}=[ D1i(idx,:)]'; % right hand
condn_data{2}= [ D2i(idx,:)]'; % both feet
condn_data{3}=[ D3i(idx,:)]'; % left hand
condn_data{4}=[ D4i(idx,:)]'; % head
condn_data{5}=[ D5i(idx,:)]'; % mime up
condn_data{6}=[ D6i(idx,:)]'; % tongue in
condn_data{7}=[ D7i(idx,:)]'; % both hands

A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};
E = condn_data{5};
F = condn_data{6};
G = condn_data{7};

clear N
N = [A' B' C' D' E' F' G'];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1)];
T = zeros(size(T1,1),7);
[aa bb]=find(T1==1);[aa(1) aa(end)]
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)]
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)]
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)]
T(aa(1):aa(end),4)=1;
[aa bb]=find(T1==5);[aa(1) aa(end)]
T(aa(1):aa(end),5)=1;
[aa bb]=find(T1==6);[aa(1) aa(end)]
T(aa(1):aa(end),6)=1;
[aa bb]=find(T1==7);[aa(1) aa(end)]
T(aa(1):aa(end),7)=1;


% using custom layers
layers = [ ...
    featureInputLayer(32)
    fullyConnectedLayer(16)
    reluLayer
    batchNormalizationLayer
    fullyConnectedLayer(8)
    reluLayer
    batchNormalizationLayer
    fullyConnectedLayer(3)
    reluLayer('Name','autoencoder')
    fullyConnectedLayer(8)
    reluLayer
    batchNormalizationLayer
    fullyConnectedLayer(16)
    reluLayer
    batchNormalizationLayer
    fullyConnectedLayer(32)
    regressionLayer
    ];

X = N;
Y=categorical(T1);
idx = randperm(length(Y),round(0.8*length(Y)));
Xtrain = X(:,idx);
Ytrain = Y(idx);
I = ones(length(Y),1);
I(idx)=0;
idx1 = find(I~=0);
Xtest = X(:,idx1);
Ytest = Y(idx1);


%'ValidationData',{XTest,YTest},...
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',20, ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',256,...
    'ValidationFrequency',50,...
    'ExecutionEnvironment','gpu',...
    'ValidationData',{Xtest',Xtest'});


% build the classifier
net = trainNetwork(Xtrain',Xtrain',layers,options);


% % now get activations in deepest layer
Z = activations(net,Xtrain','autoencoder');
%Z=Xtrain;
% plotting
idx = double(Ytrain);
[coeff,score,latent]=pca(Z');
Z = score';
cmap = parula(length(unique(idx)));
figure;hold on
for i=1:size(cmap,1)
    idxx = find(idx==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:));
end





%% TESTING SOME STUFF ON PROCRUSES ANALYSES TO ALIGN NEURAL BASELINE DATA
clc;clear


mse_boot=[];
for iter=1:1000
    x = rand(100,10);
    y = randn(100,10) + 5*rand(1,10);

    xtrain = x(1:80,:);
    ytrain = y(1:80,:);
    xtest = x(81:end,:);
    ytest = y(81:end,:);

    % project y onto x using least squares
    %ytrain*B = xtrain
    B = pinv(ytrain)*xtrain;

    ytest_proj = ytest*B;
    zz=mse(ytest,xtest);
    yy=mse(ytest_proj,xtest);


    % now using procrustus
    [d,Z,transform] = procrustes(xtrain,ytrain);
    c = transform.c;
    T = transform.T;
    b = transform.b;
    ytest_proc = b*ytest*T + mean(c);
    xx=mse(ytest_proc,xtest);

    mse_boot = [mse_boot;zz yy xx];
end

figure;boxplot(mse_boot(:,[2:3]))


%% USING AUTOENCODER ON MNIST

clear
addpath('C:\Users\Nikhlesh\Documents\MATLAB\MNIST')
trainImagesFile = 'train-images-idx3-ubyte.gz';
trainLabelsFile = 'train-labels-idx1-ubyte.gz';
testImagesFile = 't10k-images-idx3-ubyte.gz';
testLabelsFile = 't10k-labels-idx1-ubyte.gz';

XTrain = processImagesMNIST(trainImagesFile);
labels = processLabelsMNIST(trainLabelsFile);
XTest = processImagesMNIST(testImagesFile);
labels1 = processLabelsMNIST(testLabelsFile);

% build an autoencoder
XTrain = squeeze(XTrain);
X = reshape(XTrain,[28*28 60000]);
X = extractdata(X);

XTest = squeeze(XTest);
X1 = reshape(XTest,[28*28 size(XTest,3)]);
X1 = extractdata(X1);



% using custom layers
layers = [ ...
    featureInputLayer(784)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(128)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(8)
    batchNormalizationLayer
    reluLayer('Name','autoencoder')
    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(128)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(784)
    regressionLayer];


%'ValidationData',{XTest,YTest},...
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',4096,...
    'ValidationFrequency',30,...
    'L2Regularization',1e-4,...
    'ExecutionEnvironment','gpu');

% build the autoencoder
net = trainNetwork(X',X',layers,options);

% MSE
tmp = predict(net,X');
mseError = mse(X-tmp')
%
% % now get activations in deepest layer
Z = activations(net,X','autoencoder');

% plotting
idx = double(labels);
[coeff,score,latent]=pca(Z');
Z = score';
cmap = parula(length(unique(idx)));
figure;hold on
for i=1:size(cmap,1)
    idxx = find(idx==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:));
end

% splitting it further using a classifier layer
layers1=layers;
layers1=layers1(1:13);
layers2 = [layers1
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer]

options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',8000,...
    'ValidationFrequency',30,...
    'L2Regularization',1e-4,...
    'ExecutionEnvironment','auto');

Y=categorical(idx);
% build the classifier
net1 = trainNetwork(X',Y,layers2,options);


% % now get activations in deepest layer
Z = activations(net1,X','autoencoder');

% plotting
[coeff,score,latent]=pca(Z');
Z = score';
cmap = parula(length(unique(idx)));
figure;hold on
for i=1:size(cmap,1)
    idxx = find(idx==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:));
end

% doing it on validation data
Z = activations(net1,X1','autoencoder');

% plotting
idx1 = double(labels1);
[coeff,score,latent]=pca(Z');
Z = score';
cmap = parula(length(unique(idx1)));
figure;hold on
for i=1:size(cmap,1)
    idxx = find(idx1==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:));
end

% just plain PCA

% % now get activations in deepest layer
Z = X;
% plotting
[coeff,score,latent]=pca(Z');
Z = score';
cmap = parula(length(unique(idx)));
figure;hold on
for i=1:size(cmap,1)
    idxx = find(idx==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:));
end


%% AUTOENCODER WITH PROCRUSTUS ALIGNMENT


clc;clear
cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
load Data_SpatioTemp_AcrossDays Data

% take random 250 samples from imagined data
Data1=[];
for i=2:length(Data.Day)
    tmp = Data.Day(i).online_data;
    for j=1:length(tmp)
        tmp1 = tmp{j};
        tmp1 = tmp1(1:80,:);
        tmp{j} = tmp1;
    end
    Data1.Day(i).online_data = tmp;
    % now do procustus alignment
    if i>1
        disp(i)
        base_data =  Data1.Day(1).online_data;
        curr_data =  Data1.Day(i).online_data;
        for k=1:length(base_data)
            tmpb = base_data{k};
            tmpc = curr_data{k};
            [D, Z] = procrustes(tmpb, tmpc);
            curr_data{k} = Z;
        end
        Data1.Day(i).online_data = curr_data;
    end
end

condn_data={};
for i=1:7
    data=[];
    for j=2:9
        tmp = Data1.Day(j).online_data;
        tmp = tmp{i};
        data = [data;tmp];
    end
    condn_data{i} = data;
end

% train an autoencoder and see how it goes
X=[];
idx=[];
for i=1:length(condn_data)
    tmp = condn_data{i};
    X = [X;tmp];
    idx = [idx ;i*ones(size(tmp,1),1)];
end
% artfiact correct
m = mean(X);
for i=1:size(X,2)
    [aa bb] = find(abs(X(:,i))>=4);
    X(aa,i) = m(i);
end
X=X';
size(X)


% using custom layers
layers = [ ...
    featureInputLayer(96)
    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(32)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(16)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(8)
    batchNormalizationLayer
    reluLayer('Name','autoencoder')
    fullyConnectedLayer(16)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(32)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(96)
    regressionLayer];


%'ValidationData',{XTest,YTest},...
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',500, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',4096,...
    'ValidationFrequency',30,...
    'L2Regularization',1e-4,...
    'ExecutionEnvironment','gpu',...
    'Plots','none');

% build the autoencoder
net = trainNetwork(X',X',layers,options);

% MSE
tmp = predict(net,X');
mseError = mse(X-tmp')
%
% % now get activations in deepest layer
Z = activations(net,X','autoencoder');
% plotting
[coeff,score,latent]=pca(Z');
Z = score';
cmap = parula(length(unique(idx)));
figure;hold on
for i=1:size(cmap,1)
    idxx = find(idx==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(4,idxx),'.','color',cmap(i,:));
end

% just PCA
Z = X;
[coeff,score,latent]=pca(Z');
Z = score';
cmap = parula(length(unique(idx)));
figure;hold on
for i=1:size(cmap,1)
    idxx = find(idx==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(4,idxx),'.','color',cmap(i,:));
end

% ICA version
cd('C:\Users\Nikhlesh\Documents\MATLAB\eeglab2021.0');
eeglab;
close

[weights,sphere,compvars,bias,signs,lrates,activations] ...
    = runica(X);
Z = activations;
[coeff,score,latent]=pca(Z');
Z = score';
cmap = parula(length(unique(idx)));
figure;hold on
for i=1:size(cmap,1)
    idxx = find(idx==i);
    plot3(Z(10,idxx),Z(11,idxx),Z(12,idxx),'.','color',cmap(i,:));
end



%% autoencoder predictive model


clc;clear
close all
root_path='E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';

foldernames = {'20210615','20210616','20210623','20210625','20210630','20210702'...
    '20210707','20210716','20210728'};
cd(root_path)

imag_files={};
online_files={};
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    imag_files_temp=[];
    online_files_temp=[];
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'Imagined');
        if exist(filepath)
            imag_files_temp = [imag_files_temp;findfiles('',filepath)'];
        end
        filepath1=fullfile(folderpath,D(j).name,'BCI_Fixed');
        if exist(filepath1)
            online_files_temp = [online_files_temp;findfiles('',filepath1)'];
        end
    end
    imag_files{i} = imag_files_temp;
    online_files{i} = online_files_temp;
end

% load the imagined data files
Data=[];
for iter=1:length(imag_files)
    tmp_files = imag_files{iter};
    D1=[];
    D2=[];
    D3=[];
    D4=[];
    D5=[];
    D6=[];
    D7=[];
    for ii=1:length(tmp_files)
        disp(ii)

        indicator=1;
        try
            load(tmp_files{ii});
        catch ME
            warning('Not able to load file, skipping to next')
            indicator = 0;
        end

        if indicator
            features  = TrialData.BroadbandData';
            kinax = TrialData.TaskState;
            kinax = [find(kinax==3)];
            temp = cell2mat(features(kinax));
            temp = temp(1:4990,:);

            %get the pooled data
            new_temp=[];
            [xx yy] = size(TrialData.Params.ChMap);
            for k=1:size(temp,2)
                tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
                tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
                tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
                pooled_data=[];
                for i=1:2:xx
                    for j=1:2:yy
                        delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                        beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                        hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                        pooled_data = [pooled_data; delta; beta ;hg];
                    end
                end
                new_temp= [new_temp pooled_data];
            end
            temp=new_temp;
            %temp = temp(767:896,:);
            if TrialData.TargetID == 1
                D1 = cat(3,D1,temp);
            elseif TrialData.TargetID == 2
                D2 = cat(3,D2,temp);
            elseif TrialData.TargetID == 3
                D3 = cat(3,D3,temp);
            elseif TrialData.TargetID == 4
                D4 = cat(3,D4,temp);
            elseif TrialData.TargetID == 5
                D5 = cat(3,D5,temp);
            elseif TrialData.TargetID == 6
                D6 = cat(3,D6,temp);
            elseif TrialData.TargetID == 7
                D7 = cat(3,D7,temp);
            end
        end
    end

    clear condn_data
    %idx = [1:128];
    condn_data{1}=[D1 ]; % right hand
    condn_data{2}= [D2]; % both feet
    condn_data{3}=[D3]; % left hand
    condn_data{4}=[D4]; % head
    condn_data{5}=[D5]; % mime up
    condn_data{6}=[D6]; % tongue in
    condn_data{7}=[D7]; % squeeze both hands

    Data.Day(iter).imagined_data = condn_data;
end

% load the online data files
for iter=1:length(online_files)
    tmp_files = online_files{iter};
    D1=[];
    D2=[];
    D3=[];
    D4=[];
    D5=[];
    D6=[];
    D7=[];
    for ii=1:length(tmp_files)
        disp(ii)

        indicator=1;
        try
            load(tmp_files{ii});
        catch ME
            warning('Not able to load file, skipping to next')
            indicator = 0;
        end

        if indicator
            features  = TrialData.SmoothedNeuralFeatures;
            kinax = TrialData.TaskState;
            kinax = [find(kinax==3)];
            temp = cell2mat(features(kinax));

            % get the pooled data
            new_temp=[];
            [xx yy] = size(TrialData.Params.ChMap);
            for k=1:size(temp,2)
                tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
                tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
                tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
                pooled_data=[];
                for i=1:2:xx
                    for j=1:2:yy
                        delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                        beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                        hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                        pooled_data = [pooled_data; delta; beta ;hg];
                    end
                end
                new_temp= [new_temp pooled_data];
            end
            temp=new_temp;
            %temp = temp(769:896,:);
            if TrialData.TargetID == 1
                D1 = [D1 temp];
            elseif TrialData.TargetID == 2
                D2 = [D2 temp];
            elseif TrialData.TargetID == 3
                D3 = [D3 temp];
            elseif TrialData.TargetID == 4
                D4 = [D4 temp];
            elseif TrialData.TargetID == 5
                D5 = [D5 temp];
            elseif TrialData.TargetID == 6
                D6 = [D6 temp];
            elseif TrialData.TargetID == 7
                D7 = [D7 temp];
            end
        end
    end


    clear condn_data
    idx = [1:96];
    condn_data{1}=[D1(idx,:) ]'; % right hand
    condn_data{2}= [D2(idx,:)]'; % both feet
    condn_data{3}=[D3(idx,:)]'; % left hand
    condn_data{4}=[D4(idx,:)]'; % head
    condn_data{5}=[D5(idx,:)]'; % mime up
    condn_data{6}=[D6(idx,:)]'; % tongue in
    condn_data{7}=[D7(idx,:)]'; % squeeze both hands

    Data.Day(iter).online_data = condn_data;
end

cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
save Data_SpatioTemp_AcrossDays_Samples Data -v7.3

condn_data={};
for i=1:7
    data=[];
    for j=1:9
        tmp = Data.Day(j).online_data;
        tmp = tmp{i};
        data = [data;tmp];
    end
    condn_data{i} = data;
end



% train an autoencoder and see how it goes
X=[];
idx=[];
for i=1:length(condn_data)
    tmp = condn_data{i};
    X = [X;tmp];
    idx = [idx ;i*ones(size(tmp,1),1)];
end
% artfiact correct
m = mean(X);
for i=1:size(X,2)
    [aa bb] = find(abs(X(:,i))>=4);
    X(aa,i) = m(i);
end
X=X';
size(X)


% X7 is day 7
% X6 is day 6
% need to project data from day 7 onto space spanned by day 6 as decoder is
% built on day 6

jj = randperm(size(X7,1),size(X6,1));
X7a = X7(jj,:);
A = X6\X7a;
X = X7*A;
X=X';

%
% autoenc = trainAutoencoder(X,5,...
%      'EncoderTransferFunction','satlin',...
%         'DecoderTransferFunction','satlin',...
%         'L2WeightRegularization',0.1,...
%         'SparsityRegularization',4,...
%         'SparsityProportion',0.15,...
%         'ScaleData',0,...
%         'MaxEpochs',1000,...
%         'UseGPU',1);
%
% XReconstructed = predict(autoenc,X);
% mseError = mse(X-XReconstructed)
%
% Z = encode(autoenc,X);
% [coeff,score,latent]=pca(Z');
% Z = score';
% cmap = parula(7);
% figure;hold on
% for i=1:7
%     idxx = find(idx==i);
%     plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:));
% end


% building a 1-step predictive autoencoder
% using custom layers
layers = [ ...
    featureInputLayer(96)
    fullyConnectedLayer(64)
    layerNormalizationLayer
    reluLayer
    fullyConnectedLayer(16)
    layerNormalizationLayer
    reluLayer
    fullyConnectedLayer(6)
    layerNormalizationLayer
    reluLayer('Name','autoencoder')
    fullyConnectedLayer(16)
    layerNormalizationLayer
    reluLayer
    fullyConnectedLayer(64)
    layerNormalizationLayer
    reluLayer
    fullyConnectedLayer(96)
    layerNormalizationLayer
    regressionLayer];

X1=[];X2=[];
% getting the data ready
for i=1:length(condn_data)
    tmp = condn_data{i};
    for j=1:11
        tmp1 = squeeze(tmp(:,:,j));
        X1 = [X1 tmp1(1:end-1,:)'];
        X2 = [X2 tmp1(2:end,:)'];
    end
end

Xtrain = X1(:,1:380000);
Ytrain = X2(:,1:380000);
Xtest = X1(:,380001:end);
Ytest = X2(:,380001:end);



% network training parameters
options = trainingOptions('adam', ...
    'InitialLearnRate',1e-1, ...
    'MaxEpochs',30, ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',256,...
    'ValidationFrequency',10,...
    'L2Regularization',1e-3,...
    'ExecutionEnvironment','gpu');
%'Shuffle','every-epoch', ...
%'ValidationData',{Xtest',Ytest'}

net = trainNetwork(X',X',layers,options);

%size(Xtrain)

% % now get activations in deepest layer
Z = activations(net,X','autoencoder');

% plotting
%Z=X;
[coeff,score,latent]=pca(Z');
Z = score';
cmap = parula(7);
figure;hold on
for i=1:7
    idxx = find(idx==i);
    plot3(Z(3,idxx),Z(5,idxx),Z(6,idxx),'.','color',cmap(i,:));
end


latent_Z=[];
for loop=1:10


    % using custom layers
    layers = [ ...
        featureInputLayer(96)
        fullyConnectedLayer(64)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(32)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(8)
        batchNormalizationLayer
        reluLayer('Name','autoencoder')
        fullyConnectedLayer(32)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(64)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(96)
        regressionLayer];


    %'ValidationData',{XTest,YTest},...
    options = trainingOptions('sgdm', ...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',500, ...
        'Shuffle','every-epoch', ...
        'Verbose',true, ...
        'Plots','training-progress',...
        'MiniBatchSize',256,...
        'ValidationFrequency',30,...
        'L2Regularization',1e-4,...
        'ExecutionEnvironment','gpu',...
        'Plots','none');

    % build the autoencoder
    net = trainNetwork(X',X',layers,options);

    % MSE
    tmp = predict(net,X');
    mseError = mse(X-tmp')
    %
    % % now get activations in deepest layer
    Z = activations(net,X','autoencoder');
    %
    %     % plotting
    %     [coeff,score,latent]=pca(Z');
    %     Z = score';
    %     cmap = parula(7);
    %     figure;hold on
    %     for i=1:7
    %         idxx = find(idx==i);
    %         plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:));
    %     end
    %     title(num2str(loop))
    latent_Z(loop,:,:) = Z;
end

Z = squeeze(mean(latent_Z,1));
[coeff,score,latent]=pca(Z');
Z = score';
cmap = parula(7);
figure;hold on
for i=1:7
    idxx = find(idx==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:));
end


%
%Z = activations(net,X','autoencoder');
Z = squeeze(mean(latent_Z,1));
Y = tsne(Z','Algorithm','exact','Standardize',false,'Perplexity',30,'NumDimensions',3,...
    'Exaggeration',10);
Y=Y';
cmap = parula(7);
figure;hold on
for i=1:7
    idxx = find(idx==i);
    plot3(Y(1,idxx),Y(2,idxx),Y(3,idxx),'.','color',cmap(i,:));
    %plot(Y(1,idxx),Y(2,idxx),'.','color',cmap(i,:));
end


layers1=layers;
layers1=layers1(1:10);
layers2 = [layers1
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer]



%'ValidationData',{XTest,YTest},...
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',512,...
    'ValidationFrequency',30,...
    'L2Regularization',1e-4,...
    'ExecutionEnvironment','auto');

Y=categorical(idx);
% build the classifier
net = trainNetwork(X',Y,layers2,options);


% % now get activations in deepest layer
Z = activations(net,X','autoencoder');

% plotting
[coeff,score,latent]=pca(Z');
Z = score';
cmap = parula(7);
figure;hold on
for i=1:7
    idxx = find(idx==i);
    plot3(Z(2,idxx),Z(1,idxx),Z(3,idxx),'.','color',cmap(i,:));
end


cmap = parula(7);
figure;hold on
for i=1:7
    idxx = find(idx==i);
    plot3(Z(7,idxx),Z(8,idxx),Z(9,idxx),'color',cmap(i,:));
end


%% B1 LOOKING AT DISTRIBUTION OF FEATURE VALUES FROM FIRST BLOCK TO LAST BLOCK

clc;clear
filepath=('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210806');
first_folder = 'Robot3DArrow\110512';
last_folder = 'Robot3DArrow\141223';

% first folder
dir_name = fullfile(filepath,first_folder,'BCI_Fixed')
cd(dir_name)
files = findfiles('',pwd)';

first_dist=[];
for i=1:length(files)
    disp(i)
    load(files{i})
    features = TrialData.SmoothedNeuralFeatures;
    kinax = find(TrialData.TaskState==3);
    features = cell2mat(features(kinax));
    if TrialData.TargetID==3
        first_dist = [first_dist features(769:896,:)];
    end
end

% last folder
dir_name = fullfile(filepath,last_folder,'BCI_Fixed')
cd(dir_name)
files = findfiles('',pwd)';

last_dist=[];
for i=1:length(files)
    disp(i)
    load(files{i})
    features = TrialData.SmoothedNeuralFeatures;
    kinax = find(TrialData.TaskState==3);
    features = cell2mat(features(kinax));
    if TrialData.TargetID==3
        last_dist = [last_dist features(769:896,:)];
    end
end


last_dist(:,end+1:size(first_dist,2)) = NaN;
%first_dist(:,end+1:size(last_dist,2)) = NaN;

figure;boxplot(first_dist','whisker',3)
figure;boxplot(last_dist','whisker',3)


% plotting on the grid
ecog_grid = TrialData.Params.ChMap;
figure;
ha=tight_subplot(8,16);
set(gcf,'Color','w')
j=1;
for i=1:128

    [x y] = find(ecog_grid==j);
    if x == 1
        axes(ha(y));
        %subplot(8, 16, y)
    else
        s = 16*(x-1) + y;
        axes(ha(s));
        %subplot(8, 16, s)
    end


    tmp = [first_dist(i,:)' last_dist(i,:)'];
    boxplot(tmp,'whisker',3)
    axis tight
    box off

    h=hline(0);
    set(h,'LineWidth',1.5)
    if i~=102
        yticklabels ''
        xticklabels ''
    else
        %xticks([tim])
        %xticklabels({'S1','S2','S3','S4'})
    end

    j=j+1;

end

net=patternet(96,96,96)

%% ROBOT ENGAGEMENT ANALYSES

clc;clear
close all

foldernames = {'20210827','20210901'};
root_path = 'E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'RobotEngagement');
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'Imagined')
        files = [files;findfiles('',filepath)'];
    end
end

% load each trial's data
D1=[]; % this is during robot control
D2=[]; % this is during stopping
for i=1:length(files)
    disp(i)
    load(files{i});
    clicker_state = TrialData.TaskState;
    go_state = TrialData.GoState;
    l = find(clicker_state==3);
    kinax = [l];
    on_idx = find(go_state==1);
    off_idx = find(go_state==0);
    features = TrialData.SmoothedNeuralFeatures;
    temp = cell2mat(features(kinax));

    % get the pooled data
    new_temp=[];
    [xx yy] = size(TrialData.Params.ChMap);
    for k=1:size(temp,2)
        tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
        tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
        tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
        pooled_data=[];
        for i=1:2:xx
            for j=1:2:yy
                delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                pooled_data = [pooled_data; delta; beta ;hg];
            end
        end
        new_temp= [new_temp pooled_data];
    end
    temp=new_temp(65:end,:);

    % just the old features
    %temp = temp(769:896,:);
    temp_on = temp(:,on_idx);
    temp_off = temp(:,off_idx);
    if size(temp_off,2) >40
        D2 = cat(3,temp_off,D2);
        D1 = cat(3,temp_on,D1);
    end
end

acc_overall=[];
for iter=1:10
    idx = randperm(53,47);
    ii = ones(53,1);
    ii(idx)=0;
    idx1 = find(ii==1);
    D1train = D1(:,:,idx);D1train = D1train(:,:);
    D2train = D2(:,:,idx);D2train = D2train(:,:);
    %D1test = D1(:,:,idx1);D1test= D1test(:,:);
    %D2test = D2(:,:,idx1);D2test= D2test(:,:);

    clear condn_data
    condn_data{1} = D1train(:,1:size(D2train,2))';
    condn_data{2} = D2train';

    A = condn_data{1};
    B = condn_data{2};


    clear N
    N = [A' B' ];
    T1 = [ones(size(A,1),1);2*ones(size(B,1),1)];

    T = zeros(size(T1,1),2);
    [aa bb]=find(T1==1);[aa(1) aa(end)]
    T(aa(1):aa(end),1)=1;
    [aa bb]=find(T1==2);[aa(1) aa(end)]
    T(aa(1):aa(end),2)=1;

    % code to train a neural network
    clear net
    net = patternnet([16 16 16]) ;
    net.performParam.regularization=0.2;
    net = train(net,N,T','UseGPU','yes');

    % test it on individual trials
    acc=[];
    D2test = D2(:,:,idx1);
    for i=1:size(D2test,3)
        tmp = squeeze(D2test(:,1:end,i));
        out = net(tmp);
        out = (out>=0.5);
        out = sum(out');
        [aa bb]=max(out);
        if bb==2
            acc=[acc 1];
        else
            acc =[acc 0];
        end
    end
    acc;
    acc_overall(iter,:) = mean(acc);
end
mean(acc_overall)
tmp = bootstrp(1000,@median,acc_overall);
figure;hist(tmp,6)
xlim([0 1])
set(gcf,'Color','w')
set(gca,'FontSize',14)
title('Acc in detecting Stop state')
vline(median(acc_overall))

%% GETTING DATA FOR JENSEN ROBOT CONTINOUS

% get the 3D robot task and the 3D path task

% for the path task, paths for target 2 and 4 are flipped, can change with
% the flip command.

% get the data for the 3D robot task with clicker option
clc;clear
parent_folder = 'E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';


% more recent folders: 20210901 20210903 20210910 20210922

dirnames = {'20210901', '20210903', '20210910','20210924','20211001',...
    '20211006','20211008'};
target1_data={};idx1=1;
target2_data={};idx2=1;
target3_data={};idx3=1;
target4_data={};idx4=1;
target5_data={};idx5=1;
target6_data={};idx6=1;
target7_data={};idx7=1;
target8_data={};idx8=1;
target9_data={};idx9=1;
target10_data={};idx10=1;
target11_data={};idx11=1;
target12_data={};idx12=1;
target13_data={};idx13=1;
target14_data={};idx14=1;
for j=1:length(dirnames)
    foldername = fullfile(parent_folder,dirnames{j},'Robot');
    if exist(foldername)
        disp(foldername)
        files=findfiles('Data',foldername,1)';
        for i=1:length(files)
            disp(i)
            load(files{i})
            features = TrialData.SmoothedNeuralFeatures;
            task_state = TrialData.TaskState;
            kin = TrialData.CursorState;
            filtered_decodes = TrialData.FilteredClickerState;
            decodes = TrialData.ClickerState;
            tid = TrialData.TargetID;
            kinax = find(task_state==3);
            kin = kin(1:3,kinax);
            temp = cell2mat(features(kinax));
            target=TrialData.TargetPosition;
            % get the pooled data
            new_temp=[];
            [xx yy] = size(TrialData.Params.ChMap);
            for k=1:size(temp,2)
                tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
                tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
                tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
                pooled_data=[];
                for i=1:2:xx
                    for j=1:2:yy
                        delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                        beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                        hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                        pooled_data = [pooled_data; delta; beta ;hg];
                    end
                end
                new_temp= [new_temp pooled_data];
            end
            temp=new_temp;

            % store the data
            if tid==1
                target1_data(idx1).neural=temp;
                target1_data(idx1).kin=kin;
                target1_data(idx1).TargetPosition=target;
                target1_data(idx1).decodes = decodes;
                target1_data(idx1).filtered_decodes = decodes;
                idx1=idx1+1;
            elseif tid==2
                target2_data(idx2).neural=temp;
                target2_data(idx2).kin=kin;
                target2_data(idx2).TargetPosition=target;
                target2_data(idx2).decodes = decodes;
                target2_data(idx2).filtered_decodes = decodes;
                idx2=idx2+1;
            elseif tid==3
                target3_data(idx3).neural=temp;
                target3_data(idx3).kin=kin;
                target3_data(idx3).TargetPosition=target;
                target3_data(idx3).decodes = decodes;
                target3_data(idx3).filtered_decodes = decodes;
                idx3=idx3+1;
            elseif tid==4
                target4_data(idx4).neural=temp;
                target4_data(idx4).kin=kin;
                target4_data(idx4).TargetPosition=target;
                target4_data(idx4).decodes = decodes;
                target4_data(idx4).filtered_decodes = decodes;
                idx4=idx4+1;
            elseif tid==5
                target5_data(idx5).neural=temp;
                target5_data(idx5).kin=kin;
                target5_data(idx5).TargetPosition=target;
                target5_data(idx5).decodes = decodes;
                target5_data(idx5).filtered_decodes = decodes;
                idx5=idx5+1;
            elseif tid==6
                target6_data(idx6).neural=temp;
                target6_data(idx6).kin=kin;
                target6_data(idx6).TargetPosition=target;
                target6_data(idx6).decodes = decodes;
                target6_data(idx6).filtered_decodes = decodes;
                idx6=idx6+1;
            elseif tid==7
                target7_data(idx7).neural=temp;
                target7_data(idx7).kin=kin;
                target7_data(idx7).TargetPosition=target;
                target7_data(idx7).decodes = decodes;
                target7_data(idx7).filtered_decodes = decodes;
                idx7=idx7+1;
            elseif tid==8
                target8_data(idx8).neural=temp;
                target8_data(idx8).kin=kin;
                target8_data(idx8).TargetPosition=target;
                target8_data(idx8).decodes = decodes;
                target8_data(idx8).filtered_decodes = decodes;
                idx8=idx8+1;
            elseif tid==9
                target9_data(idx9).neural=temp;
                target9_data(idx9).kin=kin;
                target9_data(idx9).TargetPosition=target;
                target9_data(idx9).decodes = decodes;
                target9_data(idx9).filtered_decodes = decodes;
                idx9=idx9+1;
            elseif tid==10
                target10_data(idx10).neural=temp;
                target10_data(idx10).kin=kin;
                target10_data(idx10).TargetPosition=target;
                target10_data(idx10).decodes = decodes;
                target10_data(idx10).filtered_decodes = decodes;
                idx10=idx10+1;
            elseif tid==11
                target11_data(idx11).neural=temp;
                target11_data(idx11).kin=kin;
                target11_data(idx11).TargetPosition=target;
                target11_data(idx11).decodes = decodes;
                target11_data(idx11).filtered_decodes = decodes;
                idx11=idx11+1;
            elseif tid==12
                target12_data(idx12).neural=temp;
                target12_data(idx12).kin=kin;
                target12_data(idx12).TargetPosition=target;
                target12_data(idx12).decodes = decodes;
                target12_data(idx12).filtered_decodes = decodes;
                idx12=idx12+1;
            elseif tid==13
                target13_data(idx13).neural=temp;
                target13_data(idx13).kin=kin;
                target13_data(idx13).TargetPosition=target;
                target13_data(idx13).decodes = decodes;
                target13_data(idx13).filtered_decodes = decodes;
                idx13=idx13+1;
            elseif tid==14
                target14_data(idx14).neural=temp;
                target14_data(idx14).kin=kin;
                target14_data(idx14).TargetPosition=target;
                target14_data(idx14).decodes = decodes;
                target14_data(idx14).filtered_decodes = decodes;
                idx14=idx14+1;
            end

        end
    end
end

cd('E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
save CenterOutRobotData target1_data target2_data target3_data target4_data ....
    target5_data  target6_data target7_data target8_data target9_data target10_data ...
    target13_data target14_data target12_data target11_data



for j=3:65
    foldername = fullfile(parent_folder,folders(j).name);
    D = dir(foldername);

end


figure;

plot3(TrialData.TargetPosition(1),TrialData.TargetPosition(2),...
    TrialData.TargetPosition(3),'.r','MarkerSize',20)

kin = TrialData.CursorState;
hold on
plot3(kin(1,:),kin(2,:),kin(3,:),'b')




path=TrialData.Params.Paths(TrialData.TargetID);
path=path{1};

for i=1:size(path,1)
    plot3(path(i,1),path(i,2),path(i,3),'.r','MarkerSize',20)
end



%% GETTING ROBOT CENTER OUT BATCH DATA FOR JENSEN AND JIAANG

clc;clear
close all

root_folder='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
folder_days = {'20210716','20210728','20210804','20210806','20210813','20220202','20220211',...
    '20220225','20220304','20220309','20220311','20220316','20220323','20220325','20220330',...
    '20220420','20220422','20220427','20220429','20220504','20220506','20220518','20220513',...
    '20220520','20220527','20220601','20220622','20220629','20220701','20220715',...
    '20220720','20220722','20220727'};

% download 05042022 and 20220506 data

files=[];
python_files=[];
for i=1:length(folder_days)
    folder_path = fullfile(root_folder,folder_days{i},'RealRobotBatch');
    files = [files;findfiles('',folder_path,1)'];

%     python_folder_path = dir(folder_path);
%     python_folders={};
%     for j=3:length(python_folder_path)
%         python_folders=cat(2,python_folders,python_folder_path(j).name);
%     end
% 
%     for j=1:length(python_folders)
%         folder_path = fullfile(root_folder,folder_days{i},'Python',folder_days{i});
%     end
end

files1=[];
for i=1:length(files)
    if length(regexp(files{i},'Data'))>0
        files1=[files1;files(i)];
    end
end
files=files1;

Trials={};
for ii=1:length(files)
    disp(ii/length(files)*100)

    load(files{ii})



    idx=find(TrialData.TaskState==3);
    temp = cell2mat(TrialData.SmoothedNeuralFeatures(idx));
    kin=TrialData.CursorState;
    decodes = TrialData.FilteredClickerState;
    kin = kin(:,idx);

    % perform the pooling here
    new_temp=[];
    [xx yy] = size(TrialData.Params.ChMap);
    for k=1:size(temp,2)
        tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
        tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
        tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
        tmp4 = temp(641:768,k);tmp4 = tmp4(TrialData.Params.ChMap);%lg
        pooled_data=[];
        for i=1:2:xx
            for j=1:2:yy
                delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                lg = (tmp4(i:i+1,j:j+1));lg=mean(lg(:));
                pooled_data = [pooled_data; delta; beta ;lg;hg];
            end
        end
        new_temp= [new_temp pooled_data];
    end
    temp=new_temp;
    neural_features = temp;


    Trials(ii).TargetDir = TrialData.TargetID;
    Trials(ii).NeuralFeatures = neural_features;
    Trials(ii).Kinematics = kin;
    Trials(ii).decodes = decodes;

    %         % load python data if exists
    %         python_folderpath=[files{ii}(1:60) 'Python\' files{ii}(61:69)];
    %         python_foldername=files{ii}(85:90);
    %         python_filename =   str2num(files{ii}(106:109))


end


save RealRobotBatchTrials Trials  -v7.3


%% HONGYI PATH ANALYSIS
clc;clear
close all

path_calc='C:\Users\Nikhlesh\Documents\MATLAB\Hongyi_Robot_Path_Analysis\off_calc_robot';
path_manual = 'C:\Users\Nikhlesh\Documents\MATLAB\Hongyi_Robot_Path_Analysis\Robot3D_OffData';

files_calc = findfiles('',path_calc)';
files_manual = findfiles('',path_manual)';

a=load(files_calc{11})
b=load(files_manual{12})

filepath = 'E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20211001\Robot\105318\BCI_Fixed';
files = findfiles('',filepath)'

for i=1:length(files)
    figure;hold on
    load(files{i})
    idx = find(TrialData.TaskState==3);
    kin = TrialData.CursorState;
    kin = kin(:,idx);
    plot3(kin(1,:),kin(2,:),kin(3,:),'b')
    plot3(kin(1,1),kin(2,1),kin(3,1),'.c','MarkerSize',20)
    target = TrialData.TargetPosition;
    plot3(target(1),target(2),target(3),'.g','MarkerSize',20)
    clicks_filtered = TrialData.FilteredClickerState;
    clicks = TrialData.ClickerState;



    calc = a.offCalc{i};
    man = b.storeCell{i};
    for j=1:size(kin,2)
        if calc(j)==0
            plot3(kin(1,j),kin(2,j),kin(3,j),'ok','MarkerSize',10);
        end
    end

    % calculate a point to be good if it moves the robot closer to the
    % target as compared to the previous point.
    err=[];
    err(1)=0;
    d = norm(target' - kin(1:3,1));
    dist_target=[];
    dist_target = [dist_target d];
    for j=2:size(kin,2)
        d1 = norm(target' - kin(1:3,j));
        dist_target = [dist_target d1];
        if d1<=d
            err(j)=0;
        elseif d1>d && clicks(j)==7
            err(j)=0;
        else
            err(j)=1;
        end
        d=d1;
    end
    for j=1:size(kin,2)
        if err(j)==0
            plot3(kin(1,j),kin(2,j),kin(3,j),'ok','MarkerSize',10);
        end
    end
    figure;plot(dist_target);
    hold on
    stem(err*200)
    xlabel('Time bins')
    ylabel('Distance to target')

end


% plotting the trajectory along with the filtered clicker state
figure;
hold on
plot3(kin(1,:),kin(2,:),kin(3,:),'b','LineWidth',1)
plot3(kin(1,1),kin(2,1),kin(3,1),'.g','MarkerSize',100)
plot3(target(1),target(2),target(3),'.r','MarkerSize',100)

for j=1:size(kin,2)
    if clicks_filtered(j)<7
        if  clicks_filtered(j)>0
            plot3(kin(1,j),kin(2,j),kin(3,j),'ok','MarkerSize',10);
        else
            plot3(kin(1,j),kin(2,j),kin(3,j),'.k','MarkerSize',10);
        end
    end
end
set(gcf,'Color','w')
set(gca,'FontSize',16)
set(gca,'LineWidth',1)
xlabel('X-axis')
ylabel('Y-axis')
zlabel('Z-axis')


%% GETTING THE LATERAL R2G TASK FOR JENSEN AND CO

clc;clear
close all
filepath = 'E:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
folders = {'20210929','20211001',...
    '20211006','20211008','20211013','20211015'};


files=[];
for i=1:length(folders)
    disp([i/length(folders)])
    foldername = fullfile(filepath,folders{i},'RobotLateralR2G');
    if exist(foldername)
        tmp = findfiles('mat',foldername,1);
        for j=1:length(tmp)
            if regexp(tmp{j},'BCI_Fixed')
                files=[files;tmp(j)];
            end
        end
    end
end


% split the data into subparts i.e., subtask1 and subtask2.
% for each subtask, get the kinematic data, the taget location, filtered
% clicker decode and the actual clicker decode
trials={};
for ii=1:length(files)
    disp([ii/length(files)])
    load(files{ii})
    subtask = TrialData.Subtask;
    id = TrialData.TargetID;
    idx1 = find(subtask==1);
    idx2 = find(subtask==2);
    targets = TrialData.Params.ReachTargetPositions{id};
    target_radius = TrialData.Params.RobotTargetRadius;
    feat = cell2mat(TrialData.SmoothedNeuralFeatures(TrialData.TaskState==3));
    %feat = feat([129:256 513:640 769:end],:);
    temp=feat;

    % perform the pooling here
    new_temp=[];
    [xx yy] = size(TrialData.Params.ChMap);
    for k=1:size(temp,2)
        tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
        tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
        tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
        pooled_data=[];
        for i=1:2:xx
            for j=1:2:yy
                delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                pooled_data = [pooled_data; delta; beta ;hg];
            end
        end
        new_temp= [new_temp pooled_data];
    end
    temp=new_temp;
    neural_features = temp;

    % get the kinematics
    kin = TrialData.CursorState(1:3,TrialData.TaskState==3);

    % get the decodes
    decodes = TrialData.ClickerState;
    filtered_decodes = TrialData.ClickerState;

    % store the information
    %trials{ii}.subTask1.neural_features = neural_features(:,idx1);
    trials{ii}.subTask1.target_location = targets(1,:);
    trials{ii}.subTask1.target_radius = target_radius(1);
    trials{ii}.subTask1.kinematics = kin(:,idx1);
    trials{ii}.subTask1.decodes =  decodes(idx1);
    trials{ii}.subTask1.Filtered_decodes = filtered_decodes(idx1);
    %trials{ii}.subTask2.neural_features = neural_features(:,idx2);
    trials{ii}.subTask2.target_location = targets(2,:);
    trials{ii}.subTask2.target_radius = target_radius(2);
    trials{ii}.subTask2.kinematics = kin(:,idx2);
    trials{ii}.subTask2.decodes = decodes(idx2);
    trials{ii}.subTask2.Filtered_decodes = filtered_decodes(idx2);
end


trials=trials';

cd(filepath)
save robot_r2g_2Task_trials trials -v7.3

%% EMG TESTING FOR INTERFERNCE

clc;clear
filepath=('E:\DATA\ecog data\ECoG BCI\Testing BlackRock')
cd(filepath)

%blackrock2mat


cd('E:\DATA\ecog data\ECoG BCI\EMF_interference')
data_off = load('RobotTurnedOff_SuperNear005.mat');
lfp_off = data_off.lfp;
data_on = load('RobotTurnedOn004.mat');
lfp_on = data_on.lfp;

lfp_off = zscore(lfp_off);
lfp_on = zscore(lfp_on);
lfp_on = lfp_on(:,1:96);
lfp_off = lfp_off(:,1:96);
%
% [r,lags]=xcorr(lfp_off(:,1),lfp_on(:,1));
% figure;plot(lags,r)
%
%
% figure;stem(mean(lfp_on))
% hold on
% stem(mean(lfp_off))
%
% figure;stem(std(lfp_on))
% hold on
% stem(std(lfp_off))
%
%
% figure;plot(lfp_off(1:end,1));
% hold on
% plot(lfp_on(42:end,1))


a=size(lfp_on,1);
b=size(lfp_off,1);
if a>b
    lfp_on = lfp_on(1:b,:);
else
    lfp_off = lfp_off(1:a,:);
end


% low pass filter anything below 200Hz
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',200,'PassbandRipple',0.2, ...
    'SampleRate',1000);
fvtool(lpFilt)
lfp_on = filtfilt(lpFilt,lfp_on);
lfp_off = filtfilt(lpFilt,lfp_off);


% band-stop filter ro remove line noise and its first three harmonics
bad = [60 120 180 240 ];
for i=1:length(bad)
    disp(i)
    bsFilt = designfilt('bandstopiir','FilterOrder',4, ...
        'HalfPowerFrequency1',bad(i)-1,'HalfPowerFrequency2',bad(i)+1, ...
        'SampleRate',1000);
    lfp_on = filtfilt(bsFilt,lfp_on);
    lfp_off = filtfilt(bsFilt,lfp_off);
end



bins = [1:2000:size(lfp_on,1)];
p1=[];
p2=[];
for i=1:96
    disp(i)
    a=lfp_on(:,i);
    b=lfp_off(:,i);
    params_a=[];
    params_b=[];
    for j=1:length(bins)-1
        a1 = a(bins(j):bins(j+1)-1);
        b1 = b(bins(j):bins(j+1)-1);
        %m1 = ar(a1,2);
        %m2 = ar(b1,2);
        %params_a(j,:) = m1.A(2:end);
        %params_b(j,:) = m2.A(2:end);
        %[Pa,Fa] = pwelch(a1,[],[],[],1e3);
        %[Pb,Fb] = pwelch(b1,[],[],[],1e3);
        %v_distispf(Pa',Pb','x');

        [Pa,Fa] = fft_compute(a1,1e3);
        [Pb,Fa] = fft_compute(b1,1e3);
        v_distispf((Pa)',Pb','x');

        params_a(j,:) = 10*log10(abs(Pa));
        params_b(j,:) = 10*log10(abs(Pb));
    end
    %[h,p] = ttest2(params_a(:,1),params_b(:,1));
    %p1(i)=p;
    %[h,p] = ttest(params_a(:,2),params_b(:,2));
    %p2(i)=p;
    p1(i,:,:) = params_a;
    p2(i,:,:) = params_b;
end
figure;hist(p1);
figure;hist(p2);


figure;boxplot([params_a(:,1) params_b(:,1)])
figure;boxplot([params_a(:,2) params_b(:,2)])




% get the power at 9Hz, 1.5Hz, 3Hz
f1 =  find(Fa==9);
f2 = find(Fa==1.5);
f3 = find(Fa==3);

% loop over channels
pf1=[];
pf2=[];
pf3=[];
for i=1:96
    p1f1 = squeeze(p1(i,:,f1));
    p2f1 = squeeze(p2(i,:,f1));
    [h p tb st]=ttest(p1f1,p2f1);
    pf1(i) = p;

    p1f2 = squeeze(p1(i,:,f2));
    p2f2 = squeeze(p2(i,:,f2));
    [h p tb st]=ttest(p1f2,p2f2);
    pf2(i) = p;

    p1f3 = squeeze(p1(i,:,f3));
    p2f3 = squeeze(p2(i,:,f3));
    [h p tb st]=ttest(p1f3,p2f3);
    pf3(i) = p;
end

figure;hist(pf1);xlim([0 1]);vline(0.05,'r')
xlabel('P-value')
ylabel('Count')
title('T-test of differences in power at 9Hz across channels')
set(gcf,'Color','w')
set(gca,'FontSize',12)
set(gca,'LineWidth',1)
figure;hist(pf2);xlim([0 .2]);vline(0.05,'r')
xlabel('P-value')
ylabel('Count')
title('T-test of differences in power at 1.5Hz across channels')
set(gcf,'Color','w')
set(gca,'FontSize',12)
set(gca,'LineWidth',1)
figure;hist(pf3);xlim([0 .4]);vline(0.05,'r');
xlabel('P-value')
ylabel('Count')
title('T-test of differences in power at 3Hz across channels')
set(gcf,'Color','w')
set(gca,'FontSize',12)
set(gca,'LineWidth',1)

% plotting power spectra
figure;subplot(2,2,1)
plot(1:2000,a1,'k','LineWidth',1)
axis tight
title('Snippet at Ch. 23, Robot ON')
xlabel('Time in ms')
ylabel('Z-score')
set(gcf,'Color','w')
set(gca,'LineWidth',1)
set(gca,'FontSize',12)
subplot(2,2,2)
plot(Fa,10*log10(abs(Pa)),'LineWidth',1);
grid on
xlim([0 50])
title('Power Spectrum FFT, Ch 23, ON')
xlabel('Freq. (Hz)')
ylabel('Power')
set(gcf,'Color','w')
set(gca,'LineWidth',1)
set(gca,'FontSize',12)
h=vline([1.5 3 9]);
set(h,'LineWidth',.2)

subplot(2,2,3)
plot(1:2000,b1,'k','LineWidth',1)
axis tight
title('Snippet at Ch. 23, Robot OFF')
xlabel('Time in ms')
ylabel('Z-score')
set(gcf,'Color','w')
set(gca,'LineWidth',1)
set(gca,'FontSize',12)
subplot(2,2,4)
plot(Fa,10*log10(abs(Pb)),'LineWidth',1);
grid on
xlim([0 50])
h=vline([1.5 3 9]);
set(h,'LineWidth',.2)
title('Power Spectrum FFT, Ch 23, OFF')
xlabel('Freq. (Hz)')
ylabel('Power')
set(gcf,'Color','w')
set(gca,'LineWidth',1)
set(gca,'FontSize',12)







x=a1;
Fs=1e3;
N = length(x);
xdft = fft(x);
xdft = xdft(1:N/2+1);
psdx = (1/(Fs*N)) * abs(xdft).^2;
psdx(2:end-1) = 2*psdx(2:end-1);
freq = 0:Fs/length(x):Fs/2;
figure
plot(freq,10*log10(psdx))
grid on
title('Periodogram Using FFT')
xlabel('Frequency (Hz)')
ylabel('Power/Frequency (dB/Hz)')



x=b1;
Fs=1e3;
N = length(x);
xdft = fft(x);
xdft = xdft(1:N/2+1);
psdx = (1/(Fs*N)) * abs(xdft).^2;
psdx(2:end-1) = 2*psdx(2:end-1);
freq = 0:Fs/length(x):Fs/2;
hold on
plot(freq,10*log10(psdx))
xlim([0 50])


%% getting data to show artifacts for BR rep

clc;clear
close all

addpath(genpath('C:\Users\nikic\OneDrive\Documents\GitHub\Testing_Blackrock_NN'))


filename = '20220603-111209-009.ns2'
filepath = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220603\Blackrock\20220603-111209';
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220603\Blackrock\20220603-111209')
file = fullfile(filepath,filename);

data = blackrock2mat;

anin=data.anin;
lfp=data.lfp;

tt=(0:(size(lfp,1)-1))/1e3;
figure;plot(tt,lfp(:,3))
axis tight
set(gcf,'Color','w')
xlabel('Time in s')
title('Ch 3')
set(gca,'FontSize',14)

figure;plot(lfp(:,3))
[aa bb]=ginput
aa=round(aa)

lfp_noise = lfp(aa(1):aa(2),:);
tt=(0:(size(lfp_noise,1)-1))/1e3;
figure;plot(tt,lfp_noise(:,3))

lfp_clean = lfp(aa(2):aa(2)+0.8e5,:);

% perform a FFT on the noise part to see what is happening
%[psdx1,ffreq1]=fft_compute(lfp_clean(:,3),1e3,0);
%[psdx2,ffreq2]=fft_compute(lfp_noise(:,3),1e3,0);


X=zscore(lfp_clean(:,15));
X1=zscore(lfp_noise(:,15));
[Pxx,F] = pwelch(X,[],[],[],1e3);
[Pxx1,F1] = pwelch(X1,[],[],[],1e3);
figure;
hold on
plot(F,log10(Pxx),'LineWidth',1)
plot(F1,log10(Pxx1),'LineWidth',1)
set(gcf,'Color','w')
xlabel('Freq')
ylabel('Power')
set(gca,'FontSize',14)
%vline([60 120 180 240])
%xlim([0 3])
legend('Clean signal','Noisy signal')

%% MULTI STATE DECODING, WINTER 2022 
% pass the collected data through a lstm and see decoding outputs


clear;clc


addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
cd(root_path)
%foldernames = {'20220803','20220810','20220812'};
foldernames = {'20221129'};

% load the lstm 
load net_bilstm_20220824
net_bilstm = net_bilstm_20220824;

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

% low pass filters
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',25,'PassbandRipple',0.2, ...
    'SampleRate',1e3);

for i=1:length(foldernames)
    disp(i)
    filepath = fullfile(root_path,foldernames{i},'Robot3DArrow');
    folders_idx = [3:5];
    [acc_lstm_sample,lstm_output]...
        = get_lstm_performance_multistate(filepath,net_bilstm,Params,lpFilt,...
        folders_idx);
% 
%     acc_days = [acc_days diag(acc_lstm_sample)];
%     acc_mlp_days = [acc_mlp_days diag(acc_mlp_sample)];
end

figure;hold on
actions={'Rt thumb and head','Rt thumb and Lt Leg','Lt Thumb and Lt leg','Lt thumb and Head'};
for i=1:size(acc_lstm_sample,1)
    subplot(4,1,i)
    bar(acc_lstm_sample(i,:)*100,'FaceColor',[.6 .6 .6 ])
    title(actions{i})
    set(gca,'LineWidth',1)
    set(gca,'FontSize',14)
    xlim([.5 7.5])
    ylim([0 .7*100])
    ylabel('Accuracy')
    if i<4
        xticks ''
    else
        xticks = 1:7;
        xticklabels({'Rt Thumb','Lt Leg','Lt Thumb','Head','Tongue','Lips','Both Middle'})
    end
end
set(gcf,'Color','w')



% plot the decodes overall Runfeng method
figure;hold on
actions={'Rt thumb and head','Rt thumb and Lt Leg','Lt Thumb and Lt leg','Lt thumb and Head'};
for i=1:size(acc_lstm_sample,1)
    subplot(4,1,i)
    tmp = lstm_output{i};
    plot(tmp,'Color',[.5 .5 .5 .5]);
    hold on
    plot(mean(tmp,2),'b','LineWidth',1)
    title(actions{i})
    set(gca,'LineWidth',1)
    set(gca,'FontSize',14)
    xlim([.5 7.5])    
    ylabel('LSTM O/P')
    if i<4
        xticks ''
    else
        xticks = 1:7;
        xticklabels({'Rt Thumb','Lt Leg','Lt Thumb','Head','Tongue','Lips','Both Middle'})
    end
end
set(gcf,'Color','w')


