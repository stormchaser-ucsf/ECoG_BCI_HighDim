

% plan here is to look at changes in neural feature discriminability from
% early in training to online control, tracked across days

% hypothesis is that B1 gets better at generating those 'spatial pops' that
% discriminate betwween the various actions
clc;clear
close all

root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';

% for only 6 DoF original:
%foldernames = {'20210526','20210528','20210602','20210609_pm','20210611'};

foldernames = {'20210615','20210616','20210623','20210625','20210630','20210702',...
    '20210707','20210716','20210728','20210804','20210806','20210813','20210818',...
    '20210825','20210827','20210901','20210903','20210910','20210917','20210924','20210929',...
    '20211001''20211006','20211008','20211013','20211015','20211022','20211027','20211029','20211103',...
    '20211105','20211117','20211119','20220126','20220128','20220202','20220204','20220209','20220211',...
    '20220218','20220223','20220225','20220302'};
cd(root_path)


% 20210423 -> 111360, CenterOut -> right hand focus - rt -> rt thumb, top -> rt index finger, bottom -> rt middle finger, left -> left thumb
% do regression to show that there is not much information towards
% regression as there is towards classification ?



%% looking at changes in real time neural features in response to errors


%% looking at changes in cosine distance between decoders across learning from first init.

%20210615 is the first day, look at decoder relationships betwen imagined
%and online neural network weights for that day


% 114537, 20210615 -> seeded from imagined movement data with pooling
% 115420, 20210615 -> batch update to the decoder
% 135913 -> seeded part 2, same day
% 140642 -> batch update 2
clc;clear

addpath('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\clicker\7DoF_Classifiers')

% get the seeding decoders
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210615\Robot3DArrow\114537\BCI_Fixed\Data0006.mat')
net{1} = TrialData.Params.NeuralNetFunction;
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210615\Robot3DArrow\135913\BCI_Fixed\Data0006.mat')
net{3} = TrialData.Params.NeuralNetFunction;
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210630\Robot3DArrow\103756\BCI_Fixed\Data0009.mat')
net{5} = TrialData.Params.NeuralNetFunction;


% get the batch update decoder
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210615\Robot3DArrow\115420\BCI_Fixed\Data0004.mat')
net{2} = TrialData.Params.NeuralNetFunction;
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210615\Robot3DArrow\140642\BCI_Fixed\Data0004.mat')
net{4} = TrialData.Params.NeuralNetFunction;
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210630\Robot3DArrow\110415\BCI_Fixed\Data0013.mat')
net{6} = TrialData.Params.NeuralNetFunction;
net{7} = 'MLP_PreTrained_7DoF_Days1to11';

cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')

% get similarity over a range of random entries
D=zeros(length(net));
for  j=1:length(net)
    decA = net{j};
    for k=j+1:length(net)
        decB = net{k};
        d=[];
        for i=1:1000
            X = randn(96,1);
            [Y,Xf,Af,a1,a2,a3,a4] = feval(decA,X);
            x1=[a1;a2;a3;Y];
            [Y,Xf,Af,a1,a2,a3,a4] = feval(decB,X);
            x2=[a1;a2;a3;Y];
            tmp =pdist([x1 x2]','cosine');
            %d(i)=sqrt(sum((x1-x2).^2))';
            d(i) = tmp;
        end
        D(j,k) = median(d);
        D(k,j) = median(d);
    end
end

D
Z=linkage(D,'complete')
figure;dendrogram(Z)
figure;plot(D(7,:))



% in addition, look at the distance between the distribution of neural
% data, per condition over recoring blocks and days to get a sense if there
% is stability in the neural data


%% looking at interclass differences at the decoder last layer for imagined vs. online batch

% have to do it day by day to see how decoder discriminability changes from
% an autoencoder perspective, do it for two to three actions : is it % the
% variance of activity that is getting tigher? or that the neural activity
% forms a new space or goes into a different manifold?

% IMPORTANT POINT: WHEN TRAINING THE DECODER DONT 2-NORM THE DATA, BUT THEN
% 2-NORM THE DATA WHEN FEEDING IT THRU THE DECODER THAT SEEMS TO HELP IN
% GETTING THOSE GAUSSIAN SHAPES WHEN DOING THE PCA ON THE SOFTMAX LAYER

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\clicker\7DoF_Classifiers')

% get the seeding decoders
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210615\Robot3DArrow\114537\BCI_Fixed\Data0006.mat')
net{1} = TrialData.Params.NeuralNetFunction;

% get the batch update decoder
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210615\Robot3DArrow\115420\BCI_Fixed\Data0004.mat')
net{2} = TrialData.Params.NeuralNetFunction;

%%%%% analysis on the training data
% get the files that were used in the seeding decoder
folders={'110604','111123','111649'};
day_date = '20210615';
files=[];
for i=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'Imagined');
    files = [files;findfiles('',folderpath)'];
end

% have to hold on train and testing on held out trials
len=length(files);
idx=randperm(len,round(0.8*len));
idx1=ones(len,1);
idx1(idx)=0;
train_trials = files(idx);
test_trials = files(logical(idx1));


% load the training data
condn_data = load_data_for_MLP(train_trials);
condn_data = load_data_for_MLP(files);
% get decoder from training data
decoder = get_decoder_for_MLP(condn_data);
cd('C:\Users\Nikhlesh\Documents\GitHub\ECoG_BCI_HighDim')
genFunction(decoder,'MLP_7DoF_Plasticity_20210615_Imagined')

% load the testing data
condn_data = load_data_for_MLP(test_trials);
% get the softmax values at the last layer for the testing trials
decoder_name='MLP_7DoF_Plasticity_20210615_Imagined';
%decoder_name=net{1};
[Y,labels] = get_softmax(condn_data,decoder_name);

% look at the PC space of the softmax layer, color it by label
[c,s,l]=pca(Y');
cmap = parula(7);
figure;hold on
for i=1:length(labels)
    plot3(s(i,1),s(i,2),s(i,3),'.','MarkerSize',10,'Color',cmap(labels(i),:));
end

% get the files that were used to batch update the decoder i.e., the online
% trials
folders = {'113909','114318','114537'};
%folders = {'115420'};
day_date = '20210615';
files=[];
for i=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'BCI_Fixed');
    files = [files;findfiles('',folderpath)'];
end
condn_data = load_data_for_MLP(files);
decoder_name=net{1};
[Y,labels] = get_softmax(condn_data,decoder_name);

% look at the PC space of the softmax layer, color it by label
[c,s,l]=pca(Y');
cmap = parula(7);
figure;hold on
for i=1:length(labels)
    plot3(s(i,1),s(i,2),s(i,3),'.','MarkerSize',10,'Color',cmap(labels(i),:));
end





%% keeping track of all the imagined plus online sessions for arrow experiment

clc;clear
session_data=[];
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\clicker\7DoF_Classifiers')
cd(root_path)

%day1
session_data(1).Day = '20210615';
session_data(1).folders = {'110604','111123','111649','113524','113909','114318',...
    '114537','115420','115703','132843','133545','134131','134735','135913','140642',...
    '140904','144621','144814','145829','150031','150224','150839'};
session_data(1).folder_type={'I','I','I','O','O','O','O','B','B','I','I','I','I','O','B',...
    'B','B','B','B','B','B','B'};
session_data(1).AM_PM = {'am','am','am','am','am','am','am','am','am',...
    'pm','pm','pm','pm','pm','pm','pm','pm','pm','pm','pm','pm','pm'};

%day2
session_data(2).Day = '20210616';
session_data(2).folders={'111251','111821','112750','113117','113759','114449',...
    '134638','135318','135829','140842','141045','141459','143736'};
session_data(2).folder_type={'I','I','O','O','O','B','I','I','I','O','O','B','B'};
session_data(2).AM_PM = {'am','am','am','am','am','am','pm','pm','pm','pm','pm','pm','pm'};


% day 3
session_data(3).Day = '20210623';
session_data(3).folders={'110830','111416','111854','112823','113026',...
    '133244','133928','134357','135435','135630','135830','140530','142530','142723'};
session_data(3).folder_type={'I','I','I','O','O','I','I','I','O','O','O','B','B','B'};
session_data(3).AM_PM = {'am','am','am','am','am','pm','pm','pm','pm','pm','pm','pm',...
    'pm','pm'};



% day 4
session_data(4).Day = '20210625';
session_data(4).folders={'111134','112108','112805','113645','114239','132902',...
    '134133','142139'};
session_data(4).folder_type={'I','I','I','O','B','O','B','B'};
session_data(4).AM_PM = {'am','am','am','am','am','pm','pm','pm'};

% day 5
session_data(5).Day = '20210630';
session_data(5).folders={'101857','102342','102825','103756','110415','133210',...
    '133905','134420','135813','140408'};
session_data(5).folder_type={'I','I','I','O','B','O','I','I','O','B'};
session_data(5).AM_PM = {'am','am','am','am','am','pm','pm','pm','pm','pm'};

% day 6
session_data(6).Day = '20210702';
session_data(6).folders={'135108','135915','140426','141920','142120','142320',...
    '142800','145811'};
session_data(6).folder_type={'I','I','I','O','O','O','B','B'};
session_data(6).AM_PM = {'pm','pm','pm','pm','pm','pm','pm','pm'};

% day 7
session_data(7).Day = '20210707';
session_data(7).folders={'103731','104916','105644','110518','111026','132803',...
    '133525','134019','135008'};
session_data(7).folder_type={'I','I','I','O','B','I','I','I','O'};
session_data(7).AM_PM = {'am','am','am','am','am','pm','pm','pm','pm'};

% need to get data for 20210709
% session_data(8).Day = '20210709';
% session_data(8).folders={'101301','102021','102634'};
% session_data(8).folder_type={'I','I','I'};
% session_data(8).AM_PM = {'am','am','am''am','am'};

% day 8
session_data(8).Day = '20210714';
session_data(8).folders={'101741','102514','103106','104621','132615','133137','133748',...
    '140047','140924','141733','142605','143752','145541','150310'};
session_data(8).folder_type={'I','I','I','O','B','B','B','B','B','I','I','O','O','O'};
session_data(8).AM_PM = {'am','am','am','am','pm','pm','pm','pm','pm','pm','pm',...
    'pm','pm','pm'};

% day 9
session_data(9).Day = '20210716';
session_data(9).folders={'102008','102726','103214','104134','104745','133339',...
    '133908','134306','134936'};
session_data(9).folder_type={'I','I','I','O','B','B','B','I','O'};
session_data(9).AM_PM = {'am','am','am','am','am','pm','pm','pm','pm'};

% day 10
session_data(10).Day = '20210728';
session_data(10).folders={'103034','103745','104244','105354','110143','132842','133727',...
    '134258'};
session_data(10).folder_type={'I','I','I','O','B','B','B','B'};
session_data(10).AM_PM = {'am','am','am','am','am','pm','pm','pm'};

% day 11
session_data(11).Day = '20210804';
session_data(11).folders={'102546','103317','103821','104809','105403','133255','134125',...
    '134652'};
session_data(11).folder_type={'I','I','I','O','B','B','B','B'};
session_data(11).AM_PM = {'am','am','am','am','am','pm','pm','pm'};

%day 12
session_data(12).Day = '20210806';
session_data(12).folders={'103003','103828','104406','105415','105859','110512','134206',...
    '134915','140110','140536','141223'};
session_data(12).folder_type={'I','I','I','O','O','B','B','B','B','B','B'};
session_data(12).AM_PM = {'am','am','am','am','am','am','pm','pm','pm','pm','pm'};




save session_data session_data -v7.3

%% analzying session specific data



%% using an MLP-AE to look at differences between imagined and online control
% projecting to the latent space here works best for 4 of the 7 actions and
% not all simultaneously due to noise

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
cd('C:\Users\nikic\OneDrive\Documents\GitHub\ECoG_BCI_HighDim')
addpath(genpath(pwd))

% Imagined movement data
%folders={'110604','111123','111649'};%20210615
%folders={'134638','135318','135829'};%20210616
%folders={'133244','133928','134357'};%20210623
%folders={'111134','112108','112805'};%20210625
%folders={'102546','103317','103821'};%20210804
%folders={'103003','103828','104406'};%20210806
folders={'135108','135915','140426'};%20210702
day_date = '20210702';
files=[];
for i=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'Imagined');
    files = [files;findfiles('',folderpath)'];
end


%%%% have to do procustus when mapping data from one session to another

% online data
%folders = {'113524','113909','114318','114537'};%20210615
%folders = {'140842','141045'};
%folders={'135435','135630','135830'};%20210623
%folders={'113645','114239'};
%folders={'112750','113117','113759'};%20210616
%folders={'132615','133137','133748','140047','140924'};%20210714
%folders={'113645','114239','132902','134133','142139'};%20210625
%folders={'133255','134125','134652'};%20210804
%folders={'105415','105859','110512'};%20210806
folders={'141920','142120','142320', '142800','145811'};%20210702

day_date = '20210702';
files=[];
for i=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'BCI_Fixed');
    files = [files;findfiles('',folderpath)'];
end

%load the data
condn_data = load_data_for_MLP(files);

% build the AE based on MLP and only for hG
%[net,Xtrain,Ytrain] = build_mlp_AE(condn_data);
[net,Xtrain,Ytrain] = build_mlp_AE_supervised(condn_data);


% now build a classifier on the outerlayers

% get activations in deepest layer but averaged over a trial
TrialZ=[];
idx=[];
imag=0;
for i=1:length(files)
    disp(i)
    file_loaded=1;
    try
        load(files{i});
    catch
        file_loaded=0;
    end
    if file_loaded
        features  = TrialData.SmoothedNeuralFeatures;
        kinax = TrialData.TaskState;
        kinax = [find(kinax==3)];
        if imag==0
            counter=TrialData.Params.ClickCounter;
            kinax=kinax(end-counter+1:end);
        end
        temp = cell2mat(features(kinax));
        chmap = TrialData.Params.ChMap;
        X = bci_pooling(temp,chmap);

        %2-norm the data
        for j=1:size(X,2)
            X(:,j)=X(:,j)./norm(X(:,j));
        end

        % feed it through the AE
        X = X(1:96,:);
        Z = activations(net,X','autoencoder');
        if imag==0
            if TrialData.SelectedTargetID == TrialData.TargetID
                %Z = Z(:,end-4:end);
                TrialZ = [TrialZ Z];
                idx=[idx repmat(TrialData.TargetID,1,size(Z,2))];
                %Z = mean(Z,2);
                %TrialZ = [TrialZ Z];
                %idx=[idx TrialData.TargetID];
            end
        else
            TrialZ = [TrialZ Z];
            idx=[idx repmat(TrialData.TargetID,1,size(Z,2))];
        end
    end
end

% plot the trial averaged activity in the latent space
Z=TrialZ;
%[c,s,l]=pca(Z');
%Z=s';
cmap = parula(length(unique(idx)));
figure;hold on
for i=1:size(cmap,1)
    %if i==1||i==6||i==7||i==4||i==2
    idxx = find(idx==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:),'MarkerSize',20);
    %end
end
xlabel('Latent 1')
ylabel('Latent 2')
zlabel('Latent 3')

if imag==1
    title('Imagined Latent Space')
else
    title('Proj. Online Data through Latent Space')
end
set(gcf,'Color','w')
set(gca,'LineWidth',1)
set(gca,'FontSize',12)

% get pairwise mahalanbois distance
len = length(unique(idx));
D = zeros(len);
for i=1:len
    idxx = find(idx==i);
    A=Z(:,idxx);
    for j=i+1:len
        idxx = find(idx==j);
        B=Z(:,idxx);
        D(i,j) = mahal2(A',B',2);
        D(j,i) = D(i,j);
    end
end
dist_online = squareform(D);

figure;boxplot([dist_imagined' dist_online'])
box off
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Distance')
title('Inter-class distances')
set(gca,'LineWidth',1)
set(gca,'FontSize',12)

[h p tb st]=ttest(dist_imagined,dist_online)
mean([dist_imagined' dist_online'])



%% using an MLP-AE + decoding layer for class specific latent differences
%to look at differences between imagined and online control
% this works only when updating the decoding layer

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\Nikhlesh\Documents\GitHub\ECoG_BCI_HighDim'))


% Imagined movement data
folders={'111251','111821'};
%folders={'134638','135318','135829'};
%folders={'133244','133928','134357'};%20210623
%folders={'111134','112108','112805'}
day_date = '20210616';
files=[];
for i=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'Imagined');
    files = [files;findfiles('',folderpath)'];
end


%%%% have to do procustus when mapping data from one session to another

% online data
folders = {'112750','113117','113759'};
%folders = {'140842','141045'};
%folders={'135435','135630','135830'};%20210623
%folders={'113645','114239'};
day_date = '20210616';
files=[];
for i=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'BCI_Fixed');
    files = [files;findfiles('',folderpath)'];
end

% batch update files
folders = {'114449'};
day_date = '20210616';
files=[];
for i=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'BCI_Fixed');
    files = [files;findfiles('',folderpath)'];
end

%load the data
condn_data = load_data_for_MLP(files);

% build the AE based on MLP and only for hG
[net,Xtrain,Ytrain] = build_mlp_AE(condn_data);

% re-build the AE latent space after changing wts thru a softmax layer
[net1] =  add_decoding_AE(net,condn_data);


% perform a batch update: update the softmax weights using new online data
[net2] =  add_decoding_AE_batch(net,net1,condn_data);

% get activations from the latent space
TrialZ=[];
idx=[];
imag=0;
batch=0;
data_correct=[];
for i=1:length(files)
    disp(i)
    file_loaded=1;
    try
        load(files{i});
    catch
        file_loaded=0;
    end
    if file_loaded
        features  = TrialData.SmoothedNeuralFeatures;
        kinax = TrialData.TaskState;
        kinax = [find(kinax==3)];
        if imag==0
            counter=TrialData.Params.ClickCounter;
            kinax=kinax(end-counter+1:end);
        end
        temp = cell2mat(features(kinax));
        chmap = TrialData.Params.ChMap;
        X = bci_pooling(temp,chmap);

        %2-norm the data
        for j=1:size(X,2)
            X(:,j)=X(:,j)./norm(X(:,j));
        end

        %feed it through the AE
        X = X(1:32,:);
        Z = activations(net,X','autoencoder');

        % store


        % pass it next through softmax layer
        %         if batch==0
        %             Z = activations(net1,Z','Classif');
        %         else
        %             Z = activations(net2,Z','Classif');
        %         end

        % straight pass thru softmax layer
        %Z=activations(net1,X','Classif');

        if imag==0
            if TrialData.SelectedTargetID == TrialData.TargetID
                TrialZ = [TrialZ Z];
                idx=[idx repmat(TrialData.TargetID,1,size(Z,2))];
                %Z = mean(Z,2);
                %idx=[idx TrialData.TargetID];
            end
        else
            TrialZ = [TrialZ Z];
            idx=[idx repmat(TrialData.TargetID,1,size(Z,2))];
        end
    end
end



% plot the trial averaged activity in the latent space
Z=TrialZ;
%[c,s,l]=pca(Z');
%Z=s';
cmap = parula(length(unique(idx)));
figure;hold on
for i=1:size(cmap,1)
    %if i==1||i==6||i==7||i==4||i==3
    idxx = find(idx==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:),'MarkerSize',20);
    %end
end
xlabel('PC 1')
ylabel('PC 2')
zlabel('PC 3')

if imag==1
    title('Imagined Decoder Space')
else
    if batch==0
        title('Proj. Online Data through Imagined Decoder Space')
    else
        title('Proj. Online Data through Imagined Decoder Space- BATCH')
    end
end
set(gcf,'Color','w')
set(gca,'LineWidth',1)
set(gca,'FontSize',12)

% get pairwise mahalanbois distance
len = length(unique(idx));
D = zeros(len);
for i=1:len
    idxx = find(idx==i);
    A=Z(:,idxx);
    for j=i+1:len
        idxx = find(idx==j);
        B=Z(:,idxx);
        D(i,j) = mahal2(A',B',2);
        D(j,i) = D(i,j);
    end
end
dist_batch = squareform(D);

figure;boxplot([dist_imagined' dist_online' dist_batch'])
box off
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Distance')
title('Inter-class distances')
set(gca,'LineWidth',1)
set(gca,'FontSize',12)

[h p tb st]=ttest(dist_imagined,dist_online)


%% CONTINUING WITH APPROACH ABOVE, LOOKING AT LATENT SPACE ACROSS DAYS
% build imagined space on day 1, project dat 2 data onto it..
% cumulatively build the spaces up to see when it all stabilizes

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
cd('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim')
addpath(genpath(pwd))
addpath('C:\Users\nikic\Documents\MATLAB')

% get the folder names etc
load session_data


% Imagined movement data
folders={'110604','111123','111649'};%'20210615'
%folders={'134638','135318','135829'};20210616
%folders={'133244','133928','134357'};%20210623
%folders={'111134','112108','112805'}
day_date = '20210615';
files=[];
for i=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'Imagined');
    files = [files;findfiles('',folderpath)'];
end


%%%% have to do procustus when mapping data from one session to another

% online data
%folders = {'113524','113909','114318','114537'};%20210615
%folders = {'140842','141045'};
%folders={'135435','135630','135830'};%20210623
%folders={'112750','113117','113759'};%20210616
%folders={'140842','141045','141459','143736'};%20210616
%folders={'113645','114239'};%'20210625'
%folders={'113645','114239','132902',...
%'134133','142139'};%20210625
folders={'141920','142120','142320','142800','145811'};%'20210702'


day_date = '20210702';
files=[];
for i=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'BCI_Fixed');
    files = [files;findfiles('',folderpath)'];
end

%ratio of correct bins to total bins in online data
res = ratio_correct_bins(files);
mean(res.data)

%load the data
condn_data = load_data_for_MLP(files);

% build the AE based on MLP and only for hG
[net,Xtrain,Ytrain] = build_mlp_AE(condn_data);

% now build a classifier on the outerlayers

% get activations in deepest layer but averaged over a trial
TrialZ=[];
idx=[];
imag=0;
for i=1:length(files)
    disp(i)
    file_loaded=1;
    try
        load(files{i});
    catch
        file_loaded=0;
    end
    if file_loaded
        features  = TrialData.SmoothedNeuralFeatures;
        kinax = TrialData.TaskState;
        kinax = [find(kinax==3)];
        if imag==0
            counter=TrialData.Params.ClickCounter;
            kinax=kinax(end-counter+1:end);
        end
        temp = cell2mat(features(kinax));
        chmap = TrialData.Params.ChMap;
        X = bci_pooling(temp,chmap);

        %2-norm the data
        for j=1:size(X,2)
            X(:,j)=X(:,j)./norm(X(:,j));
        end

        % feed it through the AE
        X = X(65:96,:);
        %X = X(1:96,:);
        Z = activations(net,X','autoencoder');
        if imag==0
            if TrialData.SelectedTargetID == TrialData.TargetID
                TrialZ = [TrialZ Z];
                idx=[idx repmat(TrialData.TargetID,1,size(Z,2))];
                %Z = mean(Z,2);
                %idx=[idx TrialData.TargetID];
            end
        else
            TrialZ = [TrialZ Z];
            idx=[idx repmat(TrialData.TargetID,1,size(Z,2))];
        end
    end
end

% plot the trial averaged activity in the latent space
Z=TrialZ;
%[c,s,l]=pca(Z');
%Z=s';
cmap = parula(length(unique(idx)));
figure;hold on
for i=1:size(cmap,1)
    %if i==1||i==6||i==7||i==4||i==2
    idxx = find(idx==i);
    plot3(Z(1,idxx),Z(2,idxx),Z(3,idxx),'.','color',cmap(i,:),'MarkerSize',20);
    %end
end
xlabel('Latent 1')
ylabel('Latent 2')
zlabel('Latent 3')

if imag==1
    title('Imagined Latent Space')
else
    title('Proj. Online Data through Imagined Latent Space')
end
set(gcf,'Color','w')
set(gca,'LineWidth',1)
set(gca,'FontSize',12)

% get silhoutte index
%silh = silhouette(Z',idx');
%figure;hist(silh)



% get pairwise mahalanbois distance
len = length(unique(idx));
D = zeros(len);
for i=1:len
    idxx = find(idx==i);
    A=Z(:,idxx);
    for j=i+1:len
        idxx = find(idx==j);
        B=Z(:,idxx);
        D(i,j) = mahal2(A',B',2);
        D(j,i) = D(i,j);
    end
end
dist_online = squareform(D);

figure;boxplot([dist_imagined' dist_online'])
box off
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Distance')
title('Inter-class distances')
set(gca,'LineWidth',1)
set(gca,'FontSize',12)

[h p tb st]=ttest(dist_imagined,dist_online)


%% PUTTING IT ALL TOGETHER IN TERMS OF BUILDING THE AE
%(MAIN MAIN)

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
cd(root_path)
load session_data
dist_online_total=[];
dist_imag_total=[];
dist_batch_total=[];
var_imag_total=[];
mean_imag_total=[];
var_online_total=[];
mean_online_total=[];
var_batch_total=[];
mean_batch_total=[];
res=[];
mahab_full_online=[];
mahab_full_imagined=[];
mahab_full_batch=[];
session_data = session_data([1:9 11]); % removing bad days
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    %folders_online = logical((strcmp(session_data(i).folder_type,'B')) + (strcmp(session_data(i).folder_type,'O')));
    if i~=6
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_imag(folders_am==0)=0;
        folders_online(folders_am==0)=0;
    end

    if i==3 || i==6 || i==8
        folders_pm = strcmp(session_data(i).AM_PM,'pm');
        folders_batch(folders_pm==0)=0;
        if i==8
            idx = find(folders_batch==1);
            folders_batch(idx(3:end))=0;
        end
    else
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_batch(folders_am==0) = 0;
    end


    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);

    %%%%%%imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    condn_data = load_data_for_MLP(files);

    %     % save the data
    %     filename = ['condn_data_Imagined_Day' num2str(i)];
    %     save(filename, 'condn_data', '-v7.3')

    % get the mahab distance in the full dataset
    Dimagined = mahal2_full(condn_data);
    Dimagined = triu(Dimagined);
    Dimagined = Dimagined(Dimagined>0);
    mahab_full_imagined = [mahab_full_imagined Dimagined];

    %     % build the AE based on MLP and only for hG
    %    [net,Xtrain,Ytrain] = build_mlp_AE(condn_data);
    %     %[net,Xtrain,Ytrain] = build_mlp_AE_supervised(condn_data);
    %
    % get activations in deepest layer but averaged over a trial
    %     imag=1;
    %     [TrialZ_imag,dist_imagined,mean_imagined,var_imagined,idx_imag,~,condn_data_recon] = ...
    %         get_latent_regression(files,net,imag);
    %     dist_imag_total = [dist_imag_total;dist_imagined];
    %     mean_imag_total=[mean_imag_total;pdist(mean_imagined)];
    %     var_imag_total=[var_imag_total;var_imagined'];

    %%%%%%online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end


    %load the data
    condn_data = load_data_for_MLP(files);

    %     % save the data
    %     filename = ['condn_data_Online_Day' num2str(i)];
    %     save(filename, 'condn_data', '-v7.3')

    % get the mahab distance in the full dataset
    Donline = mahal2_full(condn_data);
    Donline = triu(Donline);
    Donline = Donline(Donline>0);
    mahab_full_online = [mahab_full_online Donline];

    %     % get activations in deepest layer
    %     imag=0;
    %     [TrialZ_online,dist_online,mean_online,var_online,idx_online,~,condn_data_recon_online]...
    %         = get_latent_regression(files,net,imag);
    %     dist_online_total = [dist_online_total;dist_online];
    %     mean_online_total=[mean_online_total;pdist(mean_online)];
    %     var_online_total=[var_online_total;var_online'];



    %%%%%%batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end


    %load the data
    condn_data = load_data_for_MLP(files);

    %     % save the data
    %     filename = ['condn_data_Batch_Day' num2str(i)];
    %     save(filename, 'condn_data', '-v7.3')

    % get the mahab distance in the full dataset
    Donline = mahal2_full(condn_data);
    Donline = triu(Donline);
    Donline = Donline(Donline>0);
    mahab_full_batch = [mahab_full_batch Donline];

    % get activations in deepest layer
    %     imag=0;
    %     [TrialZ_batch,dist_batch,mean_batch,var_batch,idx_batch] = get_latent_regression(files,net,imag);
    %     dist_batch_total = [dist_batch_total;dist_batch];
    %     mean_batch_total=[mean_batch_total;pdist(mean_batch)];
    %     var_batch_total=[var_batch_total;var_batch'];

    % plotting imagined and online in latent space
    %     idxa = find(idx_imag==4);
    %     idxb = find(idx_online==4);
    %     idxa = idxa(randperm(length(idxa),length(idxb)));
    %     figure;hold on
    %     plot3(TrialZ_imag(1,idxa),TrialZ_imag(2,idxa),TrialZ_imag(3,idxa),'.','MarkerSize',20)
    %     plot3(TrialZ_online(1,idxb),TrialZ_online(2,idxb),TrialZ_online(3,idxb),'.','MarkerSize',20)
    %     c1 = TrialZ_imag(:,idxa);
    %     c2 = TrialZ_online(:,idxb);
    %     c1=cov(c1');
    %     c2=cov(c2');

    %      plot
    %
    %     figure;boxplot([dist_imagined' dist_online'])
    %     box off
    %     set(gcf,'Color','w')
    %     xticks(1:2)
    %     xticklabels({'Imagined Data','Online Data'})
    %     ylabel('Distance')
    %     title('Inter-class distances')
    %     set(gca,'LineWidth',1)
    %     set(gca,'FontSize',12)

    %     [h p tb st]=ttest(dist_imagined,dist_online);
    %     disp([p mean([dist_imagined' dist_online'])]);
    %     res=[res;[p mean([dist_imagined' dist_online'])]];
end

close all

save mahab_distances_Full_B1 -v7.3

% compare and contrast the reconstruction with the original on an average
% basis
% to reconstruct data back
for i=1:7
    pooled_data = condn_data{i};
    pooled_data = pooled_data(:,3:3:end);
    tmp = mean(pooled_data,1);
    pooled_data_orig=reshape(tmp,8,4)';

    pooled_data_ae = condn_data_recon_online{i};
    tmp = mean(pooled_data_ae,1);
    pooled_data_recon=reshape(tmp,8,4)';

    clims_min = min([pooled_data_recon(:); pooled_data_orig(:)]);
    clims_max = max([pooled_data_recon(:); pooled_data_orig(:)]);

    figure;
    subplot(2,1,1)
    imagesc(pooled_data_orig)
    clim([clims_min clims_max])
    subplot(2,1,2)
    imagesc(pooled_data_recon)
    clim([clims_min clims_max])
    sgtitle(num2str(i))
end

% getting data from python, reconstructing back to full channel space and
% then plotting on B1 brain
% TRANSOFORMATION FRROM POOLED TO ORIGINAL
% x- > 2x-1,2x. y -> 2y-1,2y.
imaging_B1
figure;
imagesc(tmp3)
recon_data = zeros(size(chmap));
for i=1:size(tmp3,1) % going down the rows
    for j=1:size(tmp3,2) % going across the cols
        a=tmp3(i,j);
        i_idx = [2*i-1,2*i];
        j_idx = [2*j-1,2*j];
        recon_data(i_idx(1), j_idx(1))=a;
        recon_data(i_idx(1), j_idx(2))=a;
        recon_data(i_idx(2), j_idx(1))=a;
        recon_data(i_idx(2), j_idx(2))=a;
    end
end
figure;imagesc(recon_data)
recon_data = flipud(recon_data)';
recon_data = recon_data(:);
% plotting on brain as heat map
c_h = ctmr_gauss_plot(cortex,elecmatrix(1:128,:),recon_data(:),'lh',1,1,1);
e_h = el_add(elecmatrix([1:length(ch)],:), 'color', 'w', 'msize',2);
% plotting electrode sizes
val = linspace(min(recon_data),max(recon_data),128);
sz = linspace(1,10,128);
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh');
for j=1:length(recon_data)
    [aa,bb] = min(abs(val-recon_data(j)));
    ms = sz(bb)+1;
    e_h = el_add(elecmatrix(j,:), 'color', 'b','msize',ms);
end



close all
figure;
hold on
plot(smooth(median(mahab_full_imagined(:,1:end)),2))
plot(smooth(median(mahab_full_online(:,1:end)),2))
plot(smooth(median(mahab_full_batch(:,1:end)),2))


figure;
hold on
plot((mean(mahab_full_imagined(:,1:end))))
plot((mean(mahab_full_online(:,1:end))))
plot((mean(mahab_full_batch(:,1:end))))

clear tmp
w = [1/2 1/2];
tmp(:,1) = mean(mahab_full_imagined(:,1:end));
tmp(:,2) = mean(mahab_full_online(:,1:end));
tmp(:,3) = mean(mahab_full_batch(:,1:end));

for i=1:size(tmp,2)
    %xx = filter(w,1,[tmp(1,i) ;tmp(:,i)]);
    xx = filter(w,1,[tmp(:,i) ;tmp(end,i)]);
    tmp(:,i) = xx(2:end);
end

% plotting with regression lines
figure;
hold on
x=1:10;
x1=[ones(length(x),1) x'];
% imagined
y=smooth(median(mahab_full_imagined(:,1:end)),2);
plot(x,y,'.','MarkerSize',20,'Color','b')
[B,BINT,R,RINT,STATS1] = regress(y,x1);
yhat=x1*B;
plot(x,yhat,'b','LineWidth',1)
% online
y=smooth(median(mahab_full_online(:,1:end)),2);
plot(x,y,'.','MarkerSize',20,'Color','k')
[B,BINT,R,RINT,STATS2] = regress(y,x1);
yhat=x1*B;
plot(x,yhat,'k','LineWidth',1)
% batch
y=smooth(median(mahab_full_batch(:,1:end)),2);
plot(x,y,'.','MarkerSize',20,'Color','r')
[B,BINT,R,RINT,STATS3] = regress(y,x1);
yhat=x1*B;
plot(x,yhat,'r','LineWidth',1)



res

figure;
plot(mean(dist_online_total'))
set(gcf,'Color','w')
title('Across Day Learning')
ylabel('Mahalanobis Dist.')
xlabel('Day')
xlim([0.5 11.5])
tmp = mean(dist_online_total');
figure;
tmp1 = tmp(1:4);
tmp2 = tmp(5:end);
tmp1(end+1:length(tmp2))=NaN;
boxplot([tmp1' tmp2'])
xticklabels({'Early Days','Late Days'})
ylabel('Mahalanobis Dist')
title('Online Bins proj. thru Imagined Manifold')
set(gcf,'Color','w')


% stats on distances, early days and late days online mahalanobis distances
early_days_online = dist_online_total(1:4,:);
early_days_online=early_days_online(:);
late_days_online = dist_online_total(5:end,:);
late_days_online=late_days_online(:);
early_days_online(end+1:length(late_days_online))=NaN;
figure;boxplot([early_days_online late_days_online ]);


tmp = mean(dist_imag_total');
figure;
tmp1 = tmp(1:4);
tmp2 = tmp(5:end);
tmp1(end+1:length(tmp2))=NaN;
boxplot([tmp1' tmp2'])
xticklabels({'Early Days','Late Days'})
ylabel('Mahalanobis Dist')
title('Imag Bins proj. thru Imagined Manifold')
set(gcf,'Color','w')

figure;
boxplot([dist_imag_total(1,:)' dist_online_total(1,:)' ])


%plotting the difference in manifold angles between mvmt and time
ang = [mean(dist_imag_total,2) mean(dist_online_total,2) ];
%ang=fliplr(ang);
figure;hold on
%scatter(ones(length(ang),1)+0.05*randn(length(ang),1),ang(:,1));
%scatter(2*ones(length(ang),1)+0.05*randn(length(ang),1),ang(:,2));
idx = 0.05*randn(length(ang),2);
%idx = zeros(size(idx));
scatter(ones(length(ang),1)+idx(:,1),ang(:,1),100);
scatter(2*ones(length(ang),1)+idx(:,2),ang(:,2),100)
xlim([0.5 2.5])
col=winter(length(ang));
for i=1:length(ang)
    plot([1 2]+idx(i,:),(ang(i,:)),'Color',[.5 .5 .5 .5],'LineWidth',1)
end
ylim([5 65]) %0 to 40 for regression
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Mahalanobis distance')
set(gca,'FontSize',14)

%plotting changes in the variance of the latent distributions
figure;boxplot(log([var_imag_total(:) var_online_total(:) var_batch_total(:)]))
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Variance in latent space')
set(gca,'FontSize',14)
set(gcf,'Color','w')
set(gca,'LineWidth',2)
box off

ang=log([mean(var_imag_total,2) mean(var_batch_total,2)]);
figure;hold on
idx = 0.05*randn(length(ang),2);
scatter(ones(length(ang),1)+idx(:,1),ang(:,1),100);
scatter(2*ones(length(ang),1)+idx(:,2),ang(:,2),100)
xlim([0.5 2.5])
col=parula(length(ang));
for i=1:length(ang)
    %plot([1 2]+idx(i,:),(ang(i,:)),'Color',[.5 .5 .5 .5],'LineWidth',1)
    plot([1 2]+idx(i,:),(ang(i,:)),'Color',col(i,:),'LineWidth',1)
end
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Variance in latent space')
set(gca,'FontSize',14)
[h p tb st]=ttest(ang(:,1),ang(:,2))

figure;
plot(ang(:,2)-ang(:,1),'.k','MarkerSize',20)
tmp=ang(:,2)-ang(:,1);

% plotting changes in the mean distance between distributions over learning
figure;boxplot([mean_imag_total(:) mean_batch_total(:)])
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Mean Diff in latent space')
set(gca,'FontSize',14)
set(gcf,'Color','w')
set(gca,'LineWidth',2)
box off

ang=[mean(mean_imag_total,2) mean(mean_online_total,2)];
figure;hold on
idx = 0.05*randn(length(ang),2);
scatter(ones(length(ang),1)+idx(:,1),ang(:,1),100);
scatter(2*ones(length(ang),1)+idx(:,2),ang(:,2),100)
xlim([0.5 2.5])
col=parula(length(ang));
for i=1:length(ang)
    %plot([1 2]+idx(i,:),(ang(i,:)),'Color',[.5 .5 .5 .5],'LineWidth',1)
    plot([1 2]+idx(i,:),(ang(i,:)),'Color',col(i,:),'LineWidth',1)
end
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Distance b/w means in latent space')
set(gca,'FontSize',14)
[h p tb st]=ttest(ang(:,1),ang(:,2))

figure;
plot(ang(:,2)-ang(:,1),'.k','MarkerSize',20)
tmp=ang(:,2)-ang(:,1);
[bhat p wh se ci t_stat]=robust_fit((1:length(tmp))',tmp,1);
hold on
plot([ (1:length(tmp))'],...
    [ ones(size(tmp,1),1) (1:length(tmp))']*bhat,'k','LineWidth',1);
xlim([0.5 12])
xlabel('Days')
ylabel('Delta Online vs. Imagined')
title('Mean Separation in Latent Space')
set(gca,'FontSize',14)
set(gcf,'Color','w')
box off

% boostrapped test
bhat_boot=[];
parfor iter=1:500
    x=1:length(tmp);
    x=x(randperm(length(x)));
    [bhat1 p wh se ci t_stat]=robust_fit(x',tmp,1);
    bhat_boot(iter)=bhat1(2);
end
sum(bhat_boot>bhat(2))/length(bhat_boot)

x= [ ones(size(tmp,1),1) (1:length(tmp))'];
y = tmp(:,2);
[B,BINT,R,RINT,STATS] = regress(y,x);
STATS(3)


x= [ ones(size(tmp(:,1),1),1) (1:length(tmp(:,1)))'];
y = tmp(:,2);
[B,BINT,R,RINT,STATS] = regress(y,x);
STATS(3)


figure;plot(tmp(:,2),'.','MarkerSize',20)

%%%% REGRESSION LINES AS FUNCITON OF DAY %%%%%
clear tmp
w = [1/2 1/2 ];
tmp(:,1) = mean(mahab_full_imagined(:,1:end));
tmp(:,2) = mean(mahab_full_online(:,1:end));
tmp(:,3) = mean(mahab_full_batch(:,1:end));

for i=1:size(tmp,2)
    xx = filter(w,1,[tmp(1,i);tmp(:,i)]);
    %xx = filter(w,1,[tmp(:,i) ;tmp(end,i)]);
    tmp(:,i) = xx(2:end);
end


%save res_python_2D_B1 -v7.3

% stats on the Mahab distances 
%organize it as experiment, day, mahab dist
data=[];
for i=1:length(tmp)
    exp_type = [1 2 3]';
    day_name = [i i i]';
    latent_mahab_dist = tmp(i,:)';
    data = [data ; [exp_type day_name latent_mahab_dist]];
end
data_latent_dist_B1 = data;
save data_latent_dist_B1 data_latent_dist_B1 -v7.3
anova1(tmp)
% getting mean separation
mean(tmp)
std(tmp)/sqrt(10)

% doing paired t-tests
[h p tb st]=ttest(tmp(:,3),tmp(:,2))
[h p tb st]=ttest(tmp(:,3),tmp(:,1))
[h p tb st]=ttest(tmp(:,2),tmp(:,1))

% plotting the regression for Mahab distance increases as a function of day
figure;
xlim([0 11])
hold on
x= [ ones(size(tmp(:,1),1),1) (1:length(tmp(:,1)))'];
% imag
plot(1:10,tmp(:,1),'.b','MarkerSize',20)
y = tmp(:,1);
[B,BINT,R,RINT,STATS1] = regress(y,x);
yhat = x*B;
plot(1:10,yhat,'b','LineWidth',1)
% online
plot(1:10,tmp(:,2),'.k','MarkerSize',20)
y = tmp(:,2);
[B,BINT,R,RINT,STATS2] = regress(y,x);
yhat = x*B;
plot(1:10,yhat,'k','LineWidth',1)
% batch
plot(1:10,tmp(:,3),'.r','MarkerSize',20)
y = tmp(:,3);
[B,BINT,R,RINT,STATS3] = regress(y,x);
yhat = x*B;
plot(1:10,yhat,'r','LineWidth',1)
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xticks([1:10])
% yticks([5:5:35])
% ylim([5 35])

% using robust regression in matlab
figure;
xlim([0 11])
hold on
x= [ ones(size(tmp(:,1),1),1) (1:length(tmp(:,1)))'];
% imag
plot(1:10,tmp(:,1),'.b','MarkerSize',20)
y = tmp(:,1);
lm=fitlm(x(:,2:end),y,'Robust','on')
B=lm.Coefficients.Estimate;
yhat = x*B;
plot(1:10,yhat,'b','LineWidth',1)
% online
plot(1:10,tmp(:,2),'.k','MarkerSize',20)
y = tmp(:,2);
lm=fitlm(x(:,2:end),y,'Robust','on')
B=lm.Coefficients.Estimate;
yhat = x*B;
plot(1:10,yhat,'k','LineWidth',1)
% batch
plot(1:10,tmp(:,3),'.r','MarkerSize',20)
y = tmp(:,3);
lm=fitlm(x(:,2:end),y,'Robust','on')
B=lm.Coefficients.Estimate;
yhat = x*B;
plot(1:10,yhat,'r','LineWidth',1)
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xticks([1:10])
% yticks([5:5:35])
% ylim([5 35])




% plotting the robust regression for Mahab distance increases as a function of day
figure;
xlim([0 11])
hold on
x= [ ones(size(tmp(:,1),1),1) (1:length(tmp(:,1)))'];
% imag
plot(1:10,tmp(:,1),'.b','MarkerSize',20)
[bhat p1 wh se ci t_stat]=robust_fit((1:length(tmp))',tmp(:,1),2);
yhat = x*bhat;
plot(1:10,yhat,'b','LineWidth',1)
% online
plot(1:10,tmp(:,2),'.k','MarkerSize',20)
[bhat p2 wh se ci t_stat]=robust_fit((1:length(tmp))',tmp(:,2),2);
yhat = x*bhat;
plot(1:10,yhat,'k','LineWidth',1)
% batch
plot(1:10,tmp(:,3),'.r','MarkerSize',20)
[bhat p3 wh se ci t_stat]=robust_fit((1:length(tmp))',tmp(:,3),2);
yhat = x*bhat;
plot(1:10,yhat,'r','LineWidth',1)


% bootstrapped test for regression coefficient
y = tmp(:,3);
[B2,BINT,R,RINT,STATS2] = regress(y,x);
bhat_boot=[];
for i=1:2000
    ytmp = y(randperm(numel(y)));
    [B_tmp,~,Rtmp,~,STATS_tmp] = regress(ytmp,x);
    bhat_boot(i) = B_tmp(2);
end
figure;hist(bhat_boot)
vline(B2(2),'r')
sum(bhat_boot>B2(2))/length(bhat_boot)

% mixed effect model for batch and Init seed slopes
day_name=[];
mahab_dist=[];
for i=1:size(tmp,1)
    day_name = [day_name;i;i];
    %day_name = [day_name;i];
    mahab_dist = [mahab_dist;tmp(i,2:3)'];
    %mahab_dist = [mahab_dist;tmp(i,1)'];
end
data = table(day_name,mahab_dist);
glm = fitglme(data,'mahab_dist ~ 1+ day_name');
stat = glm.Coefficients.tStat(2);
stat_boot=[];
for i=1:500
    disp(i)
    day_name_tmp = day_name(randperm(numel(day_name)));
    data_tmp = table(day_name_tmp,mahab_dist);
    glm_tmp = fitglme(data_tmp,'mahab_dist ~ 1 + day_name_tmp');
    stat_boot(i) = glm_tmp.Coefficients.tStat(2);
end
figure;hist(stat_boot)
vline(stat)
sum(stat_boot>stat)/length(stat_boot)


% plotting the stats for the variance in latent-space for B1. Load the data
% from python where each sampel represents the matrix determinant for each
% movement, and collated across days

% scatter
m1 = tmp(:,1);
m1b = sort(bootstrp(1000,@mean,m1));
m11 = mean(tmp(:,1));
m2 = tmp(:,2);
m2b = sort(bootstrp(1000,@mean,m2));
m22 = mean(tmp(:,2));
m3 = tmp(:,3);
m3b = sort(bootstrp(1000,@mean,m3));
m33 = mean(tmp(:,3));
x=1:3;
y=[mean(m1) mean(m2) mean(m3)];
neg = [y(1)-m1b(25) y(2)-m2b(25) y(3)-m3b(25)];
pos = [m1b(975)-y(1) m2b(975)-y(2) m3b(975)-y(3)];
figure;
hold on
cmap = brewermap(10,'Blues');
%cmap = (turbo(7));
x1=1+0.1*randn(size(m1));
scatter(x1,m1)
x1=2+0.1*randn(size(m2));
scatter(x1,m2)
x1=3+0.1*randn(size(m3));
scatter(x1,m3)
for i=1:3
    errorbar(x(i),y(i),neg(i),pos(i),'Color','k','LineWidth',1)
    plot(x(i),y(i),'o','MarkerSize',5,'Color','k','LineWidth',2,'MarkerFaceColor',[.25 .25 .25])
end
xlim([.5 3.5])
xticks(1:3)
xticklabels({'Imagined','Online','Batch'})
set(gcf,'Color','w')
set(gca,'LineWidth',1)
% ylim([0.5 1])
% yticks(0:.1:1)
set(gca,'FontSize',12)


% getting the mixed effect model stats for latent variance in  B1
%save latent_variance_B1 -v7.3
% fitting the data into a RM anova format
tmp=[tmp];
Data_RM_B2_latentVar=[];
for i=1:length(tmp)
    exp_type = [1 2 3]';
    subject = [i i i]';
    lat_var = tmp(i,:)';
    Data_RM_B2_latentVar = [Data_RM_B2_latentVar;[exp_type subject lat_dist]];
end
save Data_RM_B2_latentVar Data_RM_B2_latentVar -v7.3

% putting into a mixed effect model format
day_name=[];
latent_var=[];
exp_type=[];
subject=[];
I = 1:7:length(tmp);
for i=1:length(I)
    tmp1 = tmp(I(i):I(i)+6,:);
    for j=1:length(tmp1)
        exp_type = [exp_type;[1 2 3]'];
        latent_var = [latent_var;(tmp1(j,:))'];
        day_name = [day_name;[i i i]'];
        subject = [subject;[1 1 1]'];
    end
end
data=table(exp_type,latent_var,day_name,subject);
glme = fitglme(data,'latent_var ~ 1 + exp_type +(1|day_name)+(1|subject)')

% doing paired t-tests
I = sort([find(data.exp_type==3);find(data.exp_type==2)]);
data1=data(I,:);
glme = fitglme(data1,'latent_var ~ 1 + exp_type +(1|day_name)')




%save bci_manifold_results_learning -v7.3


%% (MAIN) combining stats across B1 and B2 for Mahab distances

clear;clc
%b2=load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2\res_python_B2_latent.mat');
b2=load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\res_python_B2_latent_pt2.mat');
b1=load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\res_python_2D_B1.mat');

tmp1=b1.tmp;
tmp2=b2.tmp;

mean(tmp2)
std(tmp2)/sqrt(length(tmp2))

subject=[];
exp_type=[];
mah_dist=[];
for i=1:length(tmp1)
    subject=[subject;[1 1 1]'];
    exp_type=[exp_type;[1 2 3]'];
    mah_dist=[mah_dist ;tmp1(i,:)'];
end
for i=1:length(tmp2)
    subject=[subject;[2 2 2]'];
    exp_type=[exp_type;[1 2 3]'];
    mah_dist=[mah_dist ;tmp1(i,:)'];
end
data = table(mah_dist,exp_type,subject);
glm = fitglme(data,'mah_dist ~ 1+ exp_type + (1|subject)')

% fitting the data into a RM anova format
tmp=[tmp2];
Data_RM_B1B2=[];
for i=1:length(tmp)
    exp_type = [1 2 3]';
    subject = [i i i]';
    lat_dist = tmp(i,:)';
    Data_RM_B1B2 = [Data_RM_B1B2;[exp_type subject lat_dist]];
end
save Data_RM_B1B2 Data_RM_B1B2 -v7.3


%% (MAIN) loading individual trial data for trainign autoencoders

clc;clear
close all
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
cd(root_path)
load session_data
dist_online_total=[];
dist_imag_total=[];
var_imag_total=[];
mean_imag_total=[];
var_online_total=[];
mean_online_total=[];
res=[];
mahab_full_online=[];
mahab_full_imagined=[];
session_data = session_data([1:9 11]); % removing bad days
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    %folders_online = logical((strcmp(session_data(i).folder_type,'B')) + (strcmp(session_data(i).folder_type,'O')));
    if i~=6
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_imag(folders_am==0)=0;
        folders_online(folders_am==0)=0;
    end

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);

    %%%%%%imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the single trial data
    trial_data = load_data_for_MLP_TrialLevel(files);

    % get the mahab distance in the full dataset
    Dimagined = mahal2_full(condn_data);
    Dimagined = triu(Dimagined);
    Dimagined = Dimagined(Dimagined>0);
    mahab_full_imagined = [mahab_full_imagined Dimagined];

    %     % build the AE based on MLP and only for hG
    [net,Xtrain,Ytrain] = build_mlp_AE(condn_data);
    %     %[net,Xtrain,Ytrain] = build_mlp_AE_supervised(condn_data);
    %
    % get activations in deepest layer but averaged over a trial
    imag=1;
    [TrialZ_imag,dist_imagined,mean_imagined,var_imagined,idx_imag] = get_latent_regression(files,net,imag);
    dist_imag_total = [dist_imag_total;dist_imagined];
    mean_imag_total=[mean_imag_total;pdist(mean_imagined)];
    var_imag_total=[var_imag_total;var_imagined'];

    %%%%%%online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end


    %load the single trial data
    trial_data = load_data_for_MLP_TrialLevel(files);


    %load the data
    condn_data = load_data_for_MLP(files);

    % get the mahab distance in the full dataset
    Donline = mahal2_full(condn_data);
    Donline = triu(Donline);
    Donline = Donline(Donline>0);
    mahab_full_online = [mahab_full_online Donline];

    % get activations in deepest layer
    imag=0;
    [TrialZ_online,dist_online,mean_online,var_online,idx_online] = get_latent_regression(files,net,imag);
    dist_online_total = [dist_online_total;dist_online];
    mean_online_total=[mean_online_total;pdist(mean_online)];
    var_online_total=[var_online_total;var_online'];

    % plotting imagined and online in latent space
    %     idxa = find(idx_imag==4);
    %     idxb = find(idx_online==4);
    %     idxa = idxa(randperm(length(idxa),length(idxb)));
    %     figure;hold on
    %     plot3(TrialZ_imag(1,idxa),TrialZ_imag(2,idxa),TrialZ_imag(3,idxa),'.','MarkerSize',20)
    %     plot3(TrialZ_online(1,idxb),TrialZ_online(2,idxb),TrialZ_online(3,idxb),'.','MarkerSize',20)
    %     c1 = TrialZ_imag(:,idxa);
    %     c2 = TrialZ_online(:,idxb);
    %     c1=cov(c1');
    %     c2=cov(c2');

    %      plot
    %
    %     figure;boxplot([dist_imagined' dist_online'])
    %     box off
    %     set(gcf,'Color','w')
    %     xticks(1:2)
    %     xticklabels({'Imagined Data','Online Data'})
    %     ylabel('Distance')
    %     title('Inter-class distances')
    %     set(gca,'LineWidth',1)
    %     set(gca,'FontSize',12)

    [h p tb st]=ttest(dist_imagined,dist_online);
    disp([p mean([dist_imagined' dist_online'])]);
    res=[res;[p mean([dist_imagined' dist_online'])]];
end


%% PUTTING IT ALL TOGETHER IN TERMS OF BUILDING THE AE WITH PROCRUSTES
%(MAIN)

clc;clear
close all
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
cd(root_path)
load session_data
dist_online_total=[];
dist_imag_total=[];
var_imag_total=[];
mean_imag_total=[];
var_online_total=[];
mean_online_total=[];
res=[];
mahab_full_online=[];
mahab_full_imagined=[];
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    %folders_online = strcmp(session_data(i).folder_type,'O');
    folders_online = logical((strcmp(session_data(i).folder_type,'B')) + (strcmp(session_data(i).folder_type,'O')));
    %if i~=6
    folders_am = strcmp(session_data(i).AM_PM,'am');
    folders_pm = strcmp(session_data(i).AM_PM,'pm');
    %folders_imag(folders_am==0)=0;
    %folders_online(folders_am==0)=0;
    %end

    imag_idx_am = find( (folders_imag .* folders_am)==1 );
    imag_idx_pm = find( (folders_imag .* folders_pm)==1 );
    online_idx_am = find( (folders_online .* folders_am)==1 );
    online_idx_pm = find( (folders_online .* folders_pm)==1 );

    %load morning session imagined
    condn_data_am=[];
    folders = session_data(i).folders(imag_idx_am);
    day_date = session_data(i).Day;
    files_am=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files_am = [files_am;findfiles('',folderpath)'];
    end
    if length(files_am)>0
        condn_data_am = load_data_for_MLP(files_am);
    end

    % load afternoon session imagined
    condn_data_pm=[];
    folders = session_data(i).folders(imag_idx_pm);
    day_date = session_data(i).Day;
    files_pm=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files_pm = [files_pm;findfiles('',folderpath)'];
    end
    if length(files_pm)>0
        condn_data_pm = load_data_for_MLP(files_pm);
    end

    % procrutus transformation between am and pm
    %     if length(condn_data_am)>0 &&  length(condn_data_pm)>0
    %         condn_data=[];
    %         for ii=1:length(condn_data_am)
    %             a=condn_data_am{ii};
    %             b=condn_data_pm{ii};
    %             if size(a,1) < size(b,1)
    %                 idx = randperm(size(b,1),size(a,1));
    %                 b1 = b(idx,:);
    %                 idx = randperm(size(a,1),size(a,1));
    %                 a1 = a(idx,:);
    %                 [d,z,transform] = procrustes(a1,b1,"scaling",false);
    %                 bout = transform.b*b*transform.T+transform.c(1,:);
    %             elseif size(b,1) < size(a,1)
    %                 idx = randperm(size(a,1),size(b,1));
    %                 a1 = a(idx,:);
    %                 idx = randperm(size(b,1),size(b,1));
    %                 b1 = b(idx,:);
    %                 [d,z,transform] = procrustes(a1,b1,"scaling",false);
    %                 bout = transform.b*b*transform.T+transform.c(1,:);
    %             elseif size(b,1) == size(a,1)
    %                 [d,z,transform] = procrustes(a,b,"scaling",false);
    %                 bout = transform.b*b*transform.T+transform.c(1,:);
    %             end
    %             condn_data{ii} = [a;bout];
    %         end
    %     elseif length(condn_data_am) > 0 && length(condn_data_pm) == 0
    %         condn_data = condn_data_am;
    %     elseif length(condn_data_pm) > 0 &&  length(condn_data_am) == 0
    %         condn_data = condn_data_pm;
    %     end

    % getting rid of the mean shift between am and pm sessions
    if length(condn_data_am)>0 &&  length(condn_data_pm)>0
        condn_data=[];
        for ii=1:length(condn_data_am)
            a=condn_data_am{ii};
            b=condn_data_pm{ii};
            dif = mean(b) - mean(a);
            b = b-dif;
            condn_data{ii} = [a;b];
        end
    elseif length(condn_data_am) > 0 && length(condn_data_pm) == 0
        condn_data = condn_data_am;
    elseif length(condn_data_pm) > 0 &&  length(condn_data_am) == 0
        condn_data = condn_data_pm;
    end


    % get the mahab distance in the full dataset
    Dimagined = mahal2_full(condn_data);
    Dimagined = triu(Dimagined);
    Dimagined = Dimagined(Dimagined>0);
    mahab_full_imagined = [mahab_full_imagined Dimagined];

    % build the AE based on MLP and only for hG
    [net,Xtrain,Ytrain] = build_mlp_AE(condn_data);
    %[net,Xtrain,Ytrain] = build_mlp_AE_supervised(condn_data);

    % get activations in deepest layer but averaged over a trial
    imag=1;
    files=[files_am;files_pm];
    [TrialZ_imag,dist_imagined,mean_imagined,var_imagined,idx_imag] = ...
        get_latent_regression_procrustes(condn_data,net,imag);
    dist_imag_total = [dist_imag_total;dist_imagined];
    mean_imag_total=[mean_imag_total;pdist(mean_imagined)];
    var_imag_total=[var_imag_total;var_imagined'];

    %load morning session online
    folders = session_data(i).folders(online_idx_am);
    day_date = session_data(i).Day;
    files_am=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files_am = [files_am;findfiles('',folderpath)'];
    end
    if length(files_am)>0
        condn_data_am = load_data_for_MLP(files_am);
    end

    % load afternoon session online
    folders = session_data(i).folders(online_idx_pm);
    day_date = session_data(i).Day;
    files_pm=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files_pm = [files_pm;findfiles('',folderpath)'];
    end
    if length(files_pm)>0
        condn_data_pm = load_data_for_MLP(files_pm);
    end

    %     % procrutus transformation between am and pm
    %      if length(condn_data_am)>0 &&  length(condn_data_pm)>0
    %         condn_data=[];
    %         for ii=1:length(condn_data_am)
    %             a=condn_data_am{ii};
    %             b=condn_data_pm{ii};
    %             if size(a,1) < size(b,1)
    %                 idx = randperm(size(b,1),size(a,1));
    %                 b1 = b(idx,:);
    %                 idx = randperm(size(a,1),size(a,1));
    %                 a1 = a(idx,:);
    %                 [d,z,transform] = procrustes(a1,b1,"scaling",false);
    %                 bout = transform.b*b*transform.T+transform.c(1,:);
    %             elseif size(b,1) < size(a,1)
    %                 idx = randperm(size(a,1),size(b,1));
    %                 a1 = a(idx,:);
    %                 idx = randperm(size(b,1),size(b,1));
    %                 b1 = b(idx,:);
    %                 [d,z,transform] = procrustes(a1,b1,"scaling",false);
    %                 bout = transform.b*b*transform.T+transform.c(1,:);
    %             elseif size(b,1) == size(a,1)
    %                 [d,z,transform] = procrustes(a,b,"scaling",false);
    %                 bout = transform.b*b*transform.T+transform.c(1,:);
    %                 dif = mean(b) - mean(a);
    %                 b = b-dif;
    %             end
    %             condn_data{ii} = [a;bout];
    %         end
    %     elseif length(condn_data_am) > 0 && length(condn_data_pm) == 0
    %         condn_data = condn_data_am;
    %     elseif length(condn_data_pm) > 0 &&  length(condn_data_am) == 0
    %         condn_data = condn_data_pm;
    %     end

    % getting rid of the mean shift between am and pm sessions
    if length(condn_data_am)>0 &&  length(condn_data_pm)>0
        condn_data=[];
        for ii=1:length(condn_data_am)
            a=condn_data_am{ii};
            b=condn_data_pm{ii};
            dif = mean(b) - mean(a);
            b = b-dif;
            condn_data{ii} = [a;b];
        end
    elseif length(condn_data_am) > 0 && length(condn_data_pm) == 0
        condn_data = condn_data_am;
    elseif length(condn_data_pm) > 0 &&  length(condn_data_am) == 0
        condn_data = condn_data_pm;
    end

    % get the mahab distance in the full dataset
    Donline = mahal2_full(condn_data);
    Donline = triu(Donline);
    Donline = Donline(Donline>0);
    mahab_full_online = [mahab_full_online Donline];

    % get activations in deepest layer
    imag=0;
    [TrialZ_online,dist_online,mean_online,var_online,idx_online] = ...
        get_latent_regression_procrustes(condn_data,net,imag);
    dist_online_total = [dist_online_total;dist_online];
    mean_online_total=[mean_online_total;pdist(mean_online)];
    var_online_total=[var_online_total;var_online'];

    % plotting imagined and online in latent space
    %     idxa = find(idx_imag==4);
    %     idxb = find(idx_online==4);
    %     idxa = idxa(randperm(length(idxa),length(idxb)));
    %     figure;hold on
    %     plot3(TrialZ_imag(1,idxa),TrialZ_imag(2,idxa),TrialZ_imag(3,idxa),'.','MarkerSize',20)
    %     plot3(TrialZ_online(1,idxb),TrialZ_online(2,idxb),TrialZ_online(3,idxb),'.','MarkerSize',20)
    %     c1 = TrialZ_imag(:,idxa);
    %     c2 = TrialZ_online(:,idxb);
    %     c1=cov(c1');
    %     c2=cov(c2');

    %      plot
    %
    %     figure;boxplot([dist_imagined' dist_online'])
    %     box off
    %     set(gcf,'Color','w')
    %     xticks(1:2)
    %     xticklabels({'Imagined Data','Online Data'})
    %     ylabel('Distance')
    %     title('Inter-class distances')
    %     set(gca,'LineWidth',1)
    %     set(gca,'FontSize',12)

    [h p tb st]=ttest(dist_imagined,dist_online);
    disp([p mean([dist_imagined' dist_online'])]);
    res=[res;[p mean([dist_imagined' dist_online'])]];
end


figure;
hold on
plot(smooth(mean(mahab_full_imagined(:,1:11)),3))
plot(smooth(mean(mahab_full_online(:,1:11)),3))
xlim([1 8])

res

figure;
plot(mean(dist_online_total'))
set(gcf,'Color','w')
title('Across Day Learning')
ylabel('Mahalanobis Dist.')
xlabel('Day')
xlim([0.5 11.5])
tmp = mean(dist_online_total');
figure;
tmp1 = tmp(1:4);
tmp2 = tmp(5:end);
tmp1(end+1:length(tmp2))=NaN;
boxplot([tmp1' tmp2'])
xticklabels({'Early Days','Late Days'})
ylabel('Mahalanobis Dist')
title('Online Bins proj. thru Imagined Manifold')
set(gcf,'Color','w')


% stats on distances, early days and late days online mahalanobis distances
early_days_online = dist_online_total(1:4,:);
early_days_online=early_days_online(:);
late_days_online = dist_online_total(5:end,:);
late_days_online=late_days_online(:);
early_days_online(end+1:length(late_days_online))=NaN;
figure;boxplot([early_days_online late_days_online ]);


tmp = mean(dist_imag_total');
figure;
tmp1 = tmp(1:4);
tmp2 = tmp(5:end);
tmp1(end+1:length(tmp2))=NaN;
boxplot([tmp1' tmp2'])
xticklabels({'Early Days','Late Days'})
ylabel('Mahalanobis Dist')
title('Imag Bins proj. thru Imagined Manifold')
set(gcf,'Color','w')

figure;
boxplot([dist_imag_total(1,:)' dist_online_total(1,:)' ])


%plotting the difference in manifold angles between mvmt and time
ang = [mean(dist_imag_total,2) mean(dist_online_total,2)];
%ang=fliplr(ang);
figure;hold on
%scatter(ones(length(ang),1)+0.05*randn(length(ang),1),ang(:,1));
%scatter(2*ones(length(ang),1)+0.05*randn(length(ang),1),ang(:,2));
idx = 0.05*randn(length(ang),2);
%idx = zeros(size(idx));
scatter(ones(length(ang),1)+idx(:,1),ang(:,1),100);
scatter(2*ones(length(ang),1)+idx(:,2),ang(:,2),100)
xlim([0.5 2.5])
col=winter(length(ang));
for i=1:length(ang)
    plot([1 2]+idx(i,:),(ang(i,:)),'Color',[.5 .5 .5 .5],'LineWidth',1)
end
ylim([5 65]) %0 to 40 for regression
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Mahalanobis distance')
set(gca,'FontSize',14)

%plotting changes in the variance of the latent distributions
figure;boxplot([var_imag_total(:) var_online_total(:)])
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Variance in latent space')
set(gca,'FontSize',14)
set(gcf,'Color','w')
set(gca,'LineWidth',2)
box off

ang=[mean(var_imag_total,2) mean(var_online_total,2)];
figure;hold on
idx = 0.05*randn(length(ang),2);
scatter(ones(length(ang),1)+idx(:,1),ang(:,1),100);
scatter(2*ones(length(ang),1)+idx(:,2),ang(:,2),100)
xlim([0.5 2.5])
col=parula(length(ang));
for i=1:length(ang)
    %plot([1 2]+idx(i,:),(ang(i,:)),'Color',[.5 .5 .5 .5],'LineWidth',1)
    plot([1 2]+idx(i,:),(ang(i,:)),'Color',col(i,:),'LineWidth',1)
end
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Variance in latent space')
set(gca,'FontSize',14)
[h p tb st]=ttest(ang(:,1),ang(:,2))

figure;
plot(ang(:,2)-ang(:,1),'.k','MarkerSize',20)
tmp=ang(:,2)-ang(:,1);

% plotting changes in the mean distance between distributions over learning
figure;boxplot([mean_imag_total(:) mean_online_total(:)])
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Mean Diff in latent space')
set(gca,'FontSize',14)
set(gcf,'Color','w')
set(gca,'LineWidth',2)
box off

ang=[mean(mean_imag_total,2) mean(mean_online_total,2)];
figure;hold on
idx = 0.05*randn(length(ang),2);
scatter(ones(length(ang),1)+idx(:,1),ang(:,1),100);
scatter(2*ones(length(ang),1)+idx(:,2),ang(:,2),100)
xlim([0.5 2.5])
col=parula(length(ang));
for i=1:length(ang)
    %plot([1 2]+idx(i,:),(ang(i,:)),'Color',[.5 .5 .5 .5],'LineWidth',1)
    plot([1 2]+idx(i,:),(ang(i,:)),'Color',col(i,:),'LineWidth',1)
end
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Distance b/w means in latent space')
set(gca,'FontSize',14)
[h p tb st]=ttest(ang(:,1),ang(:,2))

figure;
plot(ang(:,2)-ang(:,1),'.k','MarkerSize',20)
tmp=ang(:,2)-ang(:,1);
[bhat p wh se ci t_stat]=robust_fit((1:length(tmp))',tmp,1);
hold on
plot([ (1:length(tmp))'],...
    [ ones(size(tmp,1),1) (1:length(tmp))']*bhat,'k','LineWidth',1);
xlim([0.5 12])
xlabel('Days')
ylabel('Delta Online vs. Imagined')
title('Mean Separation in Latent Space')
set(gca,'FontSize',14)
set(gcf,'Color','w')
box off

% boostrapped test
bhat_boot=[];
parfor iter=1:500
    x=1:length(tmp);
    x=x(randperm(length(x)));
    [bhat1 p wh se ci t_stat]=robust_fit(x',tmp,1);
    bhat_boot(iter)=bhat1(2);
end
sum(bhat_boot>bhat(2))/length(bhat_boot)

x= [ ones(size(tmp,1),1) (1:length(tmp))'];
y = tmp;
[B,BINT,R,RINT,STATS] = regress(y,x);
STATS(3)





%% getting the proportion of correct bins within a session
% goal here is to see if there is learning i.e. the ratio of correct
% bins increases daily (MAIN IMPORTANT)

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\OneDrive\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
load session_data
acc_bins_ratio=[];
acc_lower_bound=[];
acc_upper_bound=[];
session_data = session_data([1:9 11]); % removing bad days
for i=1:length(session_data)
    disp(100*i/length(session_data))
    folder_names = session_data(i).folders;
    folder_type = session_data(i).folder_type;
    folder_am = session_data(i).AM_PM;
    idx=[];
    for j=1:length(folder_type)
        if i~=6 && i~=10
            if strcmp(folder_type{j},'O')  && strcmp(folder_am{j},'am')
                idx=[idx j];
            end
        elseif i==10
            if strcmp(folder_type{j},'O') || strcmp(folder_type{j},'B') && strcmp(folder_am{j},'am')
                idx=[idx j];
            end
        else
            if strcmp(folder_type{j},'O') %|| strcmp(folder_type{j},'B')
                idx=[idx j];
            end
        end
    end
    folders = folder_names(idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end

    %ratio of correct bins to total bins in online data
    res = ratio_correct_bins(files);
    acc_bins_ratio(i) = mean(res.data);
    bb=sort(bootstrp(1000,@mean,res.data));
    acc_lower_bound(i)=bb(25);
    acc_upper_bound(i)=bb(975);
end

figure;plot(acc_bins_ratio,'k','LineWidth',1')
ylim([0 1.05])
xlim([0 11.5])
set(gcf,'Color','w')
box off
xticks(1:length(session_data))
xticklabels(1:length(session_data))
set(gca,'LineWidth',1)
set(gca,'FontSize',12)
xlabel('Days')
ylabel('Median Accuracy with Mode Filter')
title('Performance of same-day trained decoder')
hold on
hline(1/7,'r')
%plot(acc_lower_bound,'--k')
%plot(acc_upper_bound,'--k')

% even when using the mean accuracy, the later days have sig. more accuracy
% than the earlier days in the online task
figure;boxplot([acc_bins_ratio(1:4)' acc_bins_ratio(end-3:end)'])

% can one visualize this in latent space? I.e., the variability in latent
% space has to be reducing in later days as compared to early days.

% build AE for early days on imagined, feed in online, all bins, color code
% by incorrect bins, get variance and mahab distance. Then do same for
% later days and get variance of clouds and mahab distance between imagined
% actions.

% first four days: build AE on imagined, then feed in data from online


%% (MAIN) looking at decoding performance from imagined -> online -> batch
% across days

clc;clear;
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data
addpath 'C:\Users\nikic\Documents\MATLAB'
session_data = session_data([1:9 11]); % removing bad days
acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
iterations=50;
plot_true=false;
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    if i~=6
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_imag(folders_am==0)=0;
        folders_online(folders_am==0)=0;
    end

    if i==3 || i==6 || i==8
        folders_pm = strcmp(session_data(i).AM_PM,'pm');
        folders_batch(folders_pm==0)=0;
        if i==8
            idx = find(folders_batch==1);
            folders_batch(idx(3:end))=0;
        end
    else
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_batch(folders_am==0) = 0;
    end

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);
    %disp([session_data(i).Day '  ' num2str(length(batch_idx))]);

    %%%%%% cross_val classification accuracy for imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    condn_data = load_data_for_MLP_TrialLevel(files);
    % save the data
    filename = ['condn_data_ImaginedTrials_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    % get cross-val classification accuracy
    [acc_imagined,train_permutations] = accuracy_imagined_data(condn_data, iterations);
    acc_imagined=squeeze(nanmean(acc_imagined,1));
    if plot_true
        figure;imagesc(acc_imagined)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
    end
    acc_imagined_days(:,i) = diag(acc_imagined);


    %%%%%% get classification accuracy for online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    % get the classification accuracy
    acc_online = accuracy_online_data(files);
    if plot_true
        figure;imagesc(acc_online)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
    end
    acc_online_days(:,i) = diag(acc_online);


    %%%%%% cross_val classification accuracy for batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    % get the classification accuracy
    acc_batch = accuracy_online_data(files);
    if plot_true
        figure;imagesc(acc_batch)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
    end
    acc_batch_days(:,i) = diag(acc_batch);
end

save hDOF_10days_accuracy_results_New -v7.3
%save hDOF_10days_accuracy_results -v7.3


%acc_online_days = (acc_online_days + acc_batch_days)/2;
figure;
ylim([0.2 1])
xlim([0.5 10.5])
hold on
plot(mean(acc_imagined_days,1))
plot(median(acc_online_days,1))
plot(median(acc_batch_days,1),'k')

% linear model for time to see if improvement in decoding accuracy
days=1:10;
y=mean(acc_imagined_days,1)';
figure;hold on
plot(days,y,'.k','MarkerSize',20)
x = [ones(length(days),1) days'];
[B,BINT,R,RINT,STATS] = regress(y,x);
yhat = x*B;
plot(days,yhat,'k','LineWidth',1)
% [bhat p wh se ci t_stat]=robust_fit((1:length(tmp))',tmp',1);
% yhat1 = x*bhat;
% plot(days,yhat1,'k','LineWidth',1)
xlim([.5 10.5])
xticks([1:10])
set(gcf,'Color','w')
yticks(0:.2:1)
ylim([0 1])
STATS(3)
lm = fitlm(x(:,2),y)


% as regression lines
figure;plot(mean(acc_imagined_days,1),'.','MarkerSize',20)

% stats
tmp = [median(acc_imagined_days,1)' median(acc_online_days,1)' ...
    median(acc_batch_days,1)'];

figure;boxplot(acc_imagined_days)
ylim([0.2 1])
xlim([0.5 10.5])
hold on
boxplot(acc_batch_days,'Colors','k')
a = get(get(gca,'children'),'children');

figure;
boxplot([acc_imagined_days(:) acc_online_days(:) acc_batch_days(:)])

m1 = (acc_imagined_days(:));
m1b = sort(bootstrp(1000,@mean,m1));
m11 = mean(acc_imagined_days,1);
m2 = (acc_online_days(:));
m2b = sort(bootstrp(1000,@mean,m2));
m22 = mean(acc_online_days,1);
m3 = (acc_batch_days(:));
m3b = sort(bootstrp(1000,@mean,m3));
m33 = mean(acc_batch_days,1);
x=1:3;
y=[mean(m1) mean(m2) mean(m3)];
neg = [y(1)-m1b(25) y(2)-m2b(25) y(3)-m3b(25)];
pos = [m1b(975)-y(1) m2b(975)-y(2) m3b(975)-y(3)];
figure;
hold on
cmap = brewermap(10,'Blues');
%cmap = (turbo(7));
for i=1:10
    plot(1+0.1*randn(1),m11(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',3,'Color',[cmap(end,:) .5])
    plot(2+0.1*randn(1),m22(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',3,'Color',[cmap(end,:) .5])
    plot(3+0.1*randn(1),m33(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',3,'Color',[cmap(end,:) .5])
end
for i=1:3
    errorbar(x(i),y(i),neg(i),pos(i),'Color','k','LineWidth',1)
    plot(x(i),y(i),'o','MarkerSize',20,'Color','k','LineWidth',1,'MarkerFaceColor',[.5 .5 .5])
end
xlim([.5 3.5])
ylim([0.5 1])
xticks(1:3)
xticklabels({'Imagined','Online','Batch'})
set(gcf,'Color','w')
set(gca,'LineWidth',1)
yticks(0:.1:1)
set(gca,'FontSize',12)

tmp = [ m11' m22' m33'];
figure;boxplot(tmp)

% fit a general linear regression model across days 
acc=[];
days_acc=[];
experiment=[];
tmp = mean(acc_imagined_days,1);
acc = [acc;tmp'];
experiment =[experiment;ones(length(tmp),1)];
tmp1 = mean(acc_online_days,1);
acc = [acc;tmp1'];
experiment =[experiment;2*ones(length(tmp),1)];
tmp2 = mean(acc_batch_days,1);
acc = [acc;tmp2'];
experiment =[experiment;3*ones(length(tmp),1)];

data = table(experiment,acc);
glm = fitglm(data,'acc ~ 1 + experiment');

% stats
[h p tb st] = ttest(tmp,tmp1)
[h p tb st] = ttest(tmp,tmp2)
[h p tb st] = ttest(tmp1,tmp2)

% using a GLM for each action across days 
acc=[];
days_acc=[];
experiment=[];
for i=1:size(acc_imagined_days,2)
    a = acc_imagined_days(:,i);
    acc = [acc;a];
    days_acc = [days_acc;i*ones(size(a))];
    experiment = [experiment;1*ones(size(a))];
end
for i=1:size(acc_online_days,2)
    a = acc_online_days(:,i);
    acc = [acc;a];
    days_acc = [days_acc;i*ones(size(a))];
    experiment = [experiment;2*ones(size(a))];
end
for i=1:size(acc_batch_days,2)
    a = acc_batch_days(:,i);
    acc = [acc;a];
    days_acc = [days_acc;i*ones(size(a))];
    experiment = [experiment;3*ones(size(a))];
end

data = table(days_acc,experiment,acc);
glme = fitglme(data,'acc ~ 1 + experiment +(1|days_acc)')

%test of medians between cl1 and cl2
a0=data.acc(data.experiment==1);
a1=data.acc(data.experiment==2);
a2=data.acc(data.experiment==3);

a1 = (median(acc_online_days,1))';
a2 = (median(acc_batch_days,1))';
stat = median(a2)-median(a1);
boot=[];
a=[a1;a2];
for i=1:1000
    idx = randperm(length(a));
    a11 = a(idx(1:10));
    a22 = a(idx(11:end));
    boot(i) = median(a11) - median(a22);
end
figure;hist((boot))
vline(stat)
sum((boot) > stat)/length(boot)

% X = [ones(10,1) (1:10)'];
% Y =  mean(acc_batch_days,1)';
% [B,BINT,R,RINT,STATS] = regress(Y,X)

figure;
boxplot([a0 a1 a2])

% get the accuracies relative to imagined movement within that day
a0 = mean(acc_imagined_days,1);
a1 = mean(acc_online_days,1);
a2 = mean(acc_batch_days,1);
figure;
plot(a0);
hold on
plot(a1);
plot(a2)
ylim([0 1])

a1 = (a1-a0)./a0;
a2 = (a2-a0)./a0;
figure;boxplot([a1' a2'])
hline(0)

b1_acc_rel_imagined_prop = [a1' a2'];
save b1_acc_rel_imagined_prop b1_acc_rel_imagined_prop

%% (MAIN) saving trial level data for later use with represenational drift

clc;clear;
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data
addpath 'C:\Users\nikic\Documents\MATLAB'
session_data = session_data([1:9 11]); % removing bad days
acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
iterations=50;
plot_true=false;
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    if i~=6
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_imag(folders_am==0)=0;
        folders_online(folders_am==0)=0;
    end

    if i==3 || i==6 || i==8
        folders_pm = strcmp(session_data(i).AM_PM,'pm');
        folders_batch(folders_pm==0)=0;
        if i==8
            idx = find(folders_batch==1);
            folders_batch(idx(3:end))=0;
        end
    else
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_batch(folders_am==0) = 0;
    end

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);
    %disp([session_data(i).Day '  ' num2str(length(batch_idx))]);

    %%%%%% cross_val classification accuracy for imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    condn_data = load_data_for_MLP_TrialLevel(files);
    % save the data
    filename = ['condn_data_ImaginedTrials_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    %%%%% online data %%%%%%%
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    condn_data = load_data_for_MLP_TrialLevel(files);
    % save the data
    filename = ['condn_data_OnlineTrials_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')


    %%%%%% batch data %%%%%%%
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end


    %load the data
    condn_data = load_data_for_MLP_TrialLevel(files);
    % save the data
    filename = ['condn_data_BatchTrials_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

end


%% (MAIN PART 1) CONTINUING PNP REPRESENTANTIONAL STUFF
% goal here is to look at generalizability as consecutive days are added to
% a MLP classifier, all combinations of consecutive days

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath 'C:\Users\nikic\Documents\MATLAB'
acc_days={};
for days = 1:9 % how many consecutive days
    disp(['processing ' num2str(days) ' day'])
    acc_tmp=[];
    for i=1:10-days+1
        train_days = i:i+days-1;
        I =  ones(10,1);
        I(train_days)=0;
        test_days = find(I==1);

        % load the training data
        train_data=[];
        for j=1:length(train_days)
            imag_filename = ['condn_data_ImaginedTrials_Day' num2str(train_days(j))];
            load(imag_filename)
            train_data=[train_data condn_data];
            online_filename = ['condn_data_OnlineTrials_Day' num2str(train_days(j))];
            load(online_filename)
            train_data=[train_data condn_data];
            batch_filename = ['condn_data_BatchTrials_Day' num2str(train_days(j))];
            load(batch_filename)
            train_data=[train_data condn_data];
        end

        % load the testing
        test_data=[];
        for j=1:length(test_days)
            %             imag_filename = ['condn_data_ImaginedTrials_Day' num2str(test_days(j))];
            %             load(imag_filename)
            %             test_data=[test_data condn_data];
            online_filename = ['condn_data_OnlineTrials_Day' num2str(test_days(j))];
            load(online_filename)
            test_data=[test_data condn_data];
            batch_filename = ['condn_data_BatchTrials_Day' num2str(test_days(j))];
            load(batch_filename)
            test_data=[test_data condn_data];
        end

        % build a classifier and get the accuracy
        acc = accuracy_repdrift_days(train_data,test_data,1);
        acc = squeeze(mean(acc,1));
        acc_tmp = [acc_tmp mean(diag(acc))];
    end
    acc_days{days} = acc_tmp;
end

acc=[];
for i=1:length(acc_days)
    tmp = acc_days{i};
    acc(i) = nanmean(tmp);
end
figure;stem(acc)


%% (MAIN  PART 2) CONTINUING PNP REPRESENTANTIONAL STUFF
% goal here is to look at generalizability as consecutive days are added to
% a MLP classifier, increasing order of conseuctive days

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath 'C:\Users\nikic\Documents\MATLAB'
acc_days={};
for days = 1:9 % how many consecutive days

    acc_tmp=[];
    train_days = 1:days;
    test_days = days+1:10;
    disp(['processing day ' num2str(train_days)])


    % load the training data
    train_data=[];
    for j=1:length(train_days)
        imag_filename = ['condn_data_ImaginedTrials_Day' num2str(train_days(j))];
        load(imag_filename)
        train_data=[train_data condn_data];
        online_filename = ['condn_data_OnlineTrials_Day' num2str(train_days(j))];
        load(online_filename)
        train_data=[train_data condn_data];
        batch_filename = ['condn_data_BatchTrials_Day' num2str(train_days(j))];
        load(batch_filename)
        train_data=[train_data condn_data];
    end

    % load the testing
    test_data=[];
    for j=1:length(test_days)
        %             imag_filename = ['condn_data_ImaginedTrials_Day' num2str(test_days(j))];
        %             load(imag_filename)
        %             test_data=[test_data condn_data];
        online_filename = ['condn_data_OnlineTrials_Day' num2str(test_days(j))];
        load(online_filename)
        test_data=[test_data condn_data];
        batch_filename = ['condn_data_BatchTrials_Day' num2str(test_days(j))];
        load(batch_filename)
        test_data=[test_data condn_data];
    end

    % build a classifier and get the accuracy
    acc = accuracy_repdrift_days(train_data,test_data,1);
    acc = squeeze(mean(acc,1));
    acc_tmp = [acc_tmp mean(diag(acc))];

    % store results
    acc_days{days} = acc_tmp;
end



acc=[];
for i=1:length(acc_days)
    tmp = acc_days{i};
    acc(i) = nanmean(tmp);
end
figure;plot(acc)

%% (MAIN) LOOKING AT SPATIAL POPS B/W IMAGINED, ONLINE AND BATCH

clc;clear;
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data
addpath 'C:\Users\nikic\Documents\MATLAB'
session_data = session_data([1:9 11]); % removing bad days
acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    if i~=6
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_imag(folders_am==0)=0;
        folders_online(folders_am==0)=0;
    end

    if i==3 || i==6 || i==8
        folders_pm = strcmp(session_data(i).AM_PM,'pm');
        folders_batch(folders_pm==0)=0;
        if i==8
            idx = find(folders_batch==1);
            folders_batch(idx(3:end))=0;
        end
    else
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_batch(folders_am==0) = 0;
    end

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);
    %disp([session_data(i).Day '  ' num2str(length(batch_idx))]);

    %%%%%% hG spatial map, ERPs for indicated target direction
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    % get ERPs and sig. channels
    [hg_erps_imag,sig_ch_imag] = get_erps_imag(files); % first 5 bins is state 1



    %%%%%% get erps from online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    [hg_erps_online,sig_ch_online] = get_erps_online(files); % first 5 bins is state 1
end

% plotting a single channel
ch = 114;
imag_data = hg_erps_imag{5};
imag_data = squeeze(imag_data(ch,:,:));
online_data = hg_erps_online{5};
online_data = squeeze(online_data(ch,:,:));
figure;
hold on
%plot(imag_data,'Color',[.8 .2 .2 .5])
plot(mean(imag_data,2),'r')
%plot(online_data,'Color',[.2 .2 .8 .5])
plot(mean(online_data,2),'b')
vline([5 10],'k')
legend({'Imag','Online'})

tmp = (online_data(11:35,:));
tmp1 = (imag_data(11:35,:));
[var(tmp(:)) var(tmp1(:))]

% plotting comaprisons
ch = 97;
figure;
subplot(2,1,1)
imag_data1 = hg_erps_imag{3};
imag_data1 = squeeze(imag_data1(ch,:,:));
imag_data2 = hg_erps_imag{1};
imag_data2 = squeeze(imag_data2(ch,:,:));
hold on
%plot(imag_data,'Color',[.8 .2 .2 .5])
plot(mean(imag_data1,2),'r')
%plot(online_data,'Color',[.2 .2 .8 .5])
plot(mean(imag_data2,2),'b')
vline([5 10],'k')
legend({'LT','RT'})
title('Imagined')
subplot(2,1,2)
imag_data1 = hg_erps_online{3};
imag_data1 = squeeze(imag_data1(ch,:,:));
imag_data2 = hg_erps_online{1};
imag_data2 = squeeze(imag_data2(ch,:,:));
hold on
%plot(imag_data,'Color',[.8 .2 .2 .5])
plot(mean(imag_data1,2),'r')
%plot(online_data,'Color',[.2 .2 .8 .5])
plot(mean(imag_data2,2),'b')
vline([5 10],'k')
legend({'LT','RT'})
title('Online')


%% HOW DOES MANIFOLD CHANGE ACROSS DAYS...REPRESENTATIONAL DRIFT
% Representational drift across days, how does latent space spread across
% days? Is it crystal clear and spaces are set but we just need more data
% across days to fill it up? The manifold rotates and changes across days,
% some drift due to baseline issues etc. and you need more days to capture
% the complexity while plasticity is happening at the same time? Use
% procrustus analyses here

%one way to do that is to project day two data onto day ones data and see
%whether projection lies within same pace in latent space i.e., how much
%procrustus distance between spaces across days...

clc;clear;
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\OneDrive\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
load session_data
addpath 'C:\Users\nikic\Documents\MATLAB'

files=[];
files1=[];

for i=2:3


    day_date = session_data(i).Day;
    for ii=1:length(session_data(i).folders)
        if strcmp(session_data(i).folder_type(ii),'I')
            folderpath =  fullfile(root_path,day_date,'Robot3DArrow',...
                session_data(i).folders{ii},'Imagined');
            files = [files;findfiles('',folderpath)'];
        else
            folderpath1 =  fullfile(root_path,day_date,'Robot3DArrow',...
                session_data(i).folders{ii},'BCI_Fixed');
            files1 =[files1;findfiles('',folderpath1)'];
        end
    end

end

% keep apart 20% of files for testing
idx_train = randperm(length(files1),round(0.80*length(files1)));
I = ones(size(files1));
I(idx_train)=0;
idx_test = find(I==1);
files_train = files1(idx_train);
files_test = files1(idx_test);

files_train = [files ;files_train];



% build an AE latent space for this model uaing just the imagined data
condn_data = load_data_for_MLP(files_train);
[net,Xtrain,Ytrain] = build_mlp_AE_supervised_total(condn_data);

% now project data from own day (held out) onto the manifold
imag=0;
[TrialZ_online,dist_online,mean_online,var_online,Zidx,acc] = get_latent(files_test,net,imag);
figure;
hold on
tid=find(Zidx==1);
plot3(TrialZ_online(1,tid),TrialZ_online(2,tid),TrialZ_online(3,tid),'.b','MarkerSize',20)
tid=find(Zidx==5);
plot3(TrialZ_online(1,tid),TrialZ_online(2,tid),TrialZ_online(3,tid),'.r','MarkerSize',20)
tid=find(Zidx==7);
plot3(TrialZ_online(1,tid),TrialZ_online(2,tid),TrialZ_online(3,tid),'.g','MarkerSize',20)
axis tight
xx = get(gca,'xlim');
yy = get(gca,'ylim');
zz = get(gca,'zlim');
view(47,31)
set(gcf,'Color','w')
xlabel('Latent 1')
ylabel('Latent 2')
zlabel('Latent 3')
set(gca,'FontSize',14)
title(['AE through Day ' num2str(i) ' acc of ' num2str(mean(diag(acc)))])
%title(['AE through Day 2-3 acc of ' num2str(nanmean(diag(acc)))])
%set(gcf,'Position',[680,50,934,946])


%load all of the online data from the all the consecutive days
acc_overall=[];
mahab_dist=[];
I = ones(length(session_data),1);
I(2:3)=0;
I=find(I==1);
for k=1:length(I)
    j=I(k);
    files=[];
    files1=[];
    day_date = session_data(j).Day;
    for ii=1:length(session_data(j).folders)
        if strcmp(session_data(j).folder_type(ii),'I')
            folderpath =  fullfile(root_path,day_date,'Robot3DArrow',...
                session_data(j).folders{ii},'Imagined');
            files = [files;findfiles('',folderpath)'];
        else
            folderpath1 =  fullfile(root_path,day_date,'Robot3DArrow',...
                session_data(j).folders{ii},'BCI_Fixed');
            files1 =[files1;findfiles('',folderpath1)'];
        end
    end

    % now project data from second day onto the manifold from the first
    % day
    imag=0;
    [TrialZ_online2,dist_online2,mean_online2,var_online2,Zidx2,acc1] = get_latent(files1,net,imag);
    close
    mahab_dist = [mahab_dist;mean(dist_online2)];

    % plot the mahab distribution
    figure;
    subplot(2,1,1)
    boxplot([dist_online' dist_online2'])
    xticks(1:2)
    xticklabels({'Day 2-3', ['Day ' num2str(j)]})
    set(gcf,'Color','w')
    set(gca,'FontSize',14)
    set(gca,'LineWidth',1)
    ylabel('Mahalanobis distance')

    %plotting just condition specific data
    subplot(2,1,2)
    hold on
    tid2=find(Zidx2==1);
    plot3(TrialZ_online2(1,tid2),TrialZ_online2(2,tid2),TrialZ_online2(3,tid2),'.b','MarkerSize',20)
    tid2=find(Zidx2==5);
    plot3(TrialZ_online2(1,tid2),TrialZ_online2(2,tid2),TrialZ_online2(3,tid2),'.r','MarkerSize',20)
    tid2=find(Zidx2==7);
    plot3(TrialZ_online2(1,tid2),TrialZ_online2(2,tid2),TrialZ_online2(3,tid2),'.g','MarkerSize',20)
    xlim(xx)
    ylim(yy)
    zlim(zz)
    view(47,31)
    set(gcf,'Color','w')
    xlabel('Latent 1')
    ylabel('Latent 2')
    zlabel('Latent 3')
    set(gca,'FontSize',14)
    tmp=(diag(acc1));
    %title(['Day ' num2str(j) ' thru AE Day ' num2str(i) ' with acc ' num2str(mean(tmp))])
    title(['Day ' num2str(j) ' thru AE Day 2-3 with acc ' num2str(mean(tmp))])
    acc_overall =[acc_overall diag(acc1)];
    set(gcf,'Position',[680,50,934,946])
end

acc_overall_compar = [diag(acc) acc_overall];
figure;
boxplot(acc_overall_compar)

figure;boxplot(mahab_dist)
hline(mean(dist_online),'r')
ylim([0 35])
xlim([0.5 1.5])
xticks(1)
xticklabels('Held out days')
ylabel('Mahab Distance')
set(gca,'FontSize',14)
set(gcf,'Color','w')

%80.56% accuracy of day 1 on its own data on top condition 1,5,7
%
% % old code when looking at things from a day to day basis
% j=4;
% files=[];
% files1=[];
% day_date = session_data(j).Day;
% for ii=1:length(session_data(j).folders)
%     if strcmp(session_data(j).folder_type(ii),'I')
%         folderpath =  fullfile(root_path,day_date,'Robot3DArrow',...
%             session_data(j).folders{ii},'Imagined');
%         files = [files;findfiles('',folderpath)'];
%     else
%         folderpath1 =  fullfile(root_path,day_date,'Robot3DArrow',...
%             session_data(j).folders{ii},'BCI_Fixed');
%         files1 =[files1;findfiles('',folderpath1)'];
%     end
% end
% % now project data from second day onto the manifold from the first
% % day
% imag=0;
% [TrialZ_online2,dist_online2,mean_online2,var_online2,Zidx2,acc1] = get_latent(files1,net,imag);
%
% % plotting the data on the manifold
% figure;
% hold on
% plot3(TrialZ_online(1,:),TrialZ_online(2,:),TrialZ_online(3,:),'.','MarkerSize',20)
% plot3(TrialZ_online2(1,:),TrialZ_online2(2,:),TrialZ_online2(3,:),'.','MarkerSize',20)
% figure;boxplot([dist_online' dist_online2'])
%
% %plotting just condition specific data
% figure;
% hold on
% tid=find(Zidx==1);
% plot3(TrialZ_online(1,tid),TrialZ_online(2,tid),TrialZ_online(3,tid),'.b','MarkerSize',20)
% tid=find(Zidx==5);
% plot3(TrialZ_online(1,tid),TrialZ_online(2,tid),TrialZ_online(3,tid),'.r','MarkerSize',20)
% tid=find(Zidx==7);
% plot3(TrialZ_online(1,tid),TrialZ_online(2,tid),TrialZ_online(3,tid),'.g','MarkerSize',20)
% axis tight
% xx = get(gca,'xlim');
% yy = get(gca,'ylim');
% zz = get(gca,'zlim');
% view(-134,36)
% set(gcf,'Color','w')
% xlabel('Latent 1')
% ylabel('Latent 2')
% zlabel('Latent 3')
% set(gca,'FontSize',14)
% figure;hold on
% tid2=find(Zidx2==1);
% plot3(TrialZ_online2(1,tid2),TrialZ_online2(2,tid2),TrialZ_online2(3,tid2),'.b','MarkerSize',20)
% tid2=find(Zidx2==5);
% plot3(TrialZ_online2(1,tid2),TrialZ_online2(2,tid2),TrialZ_online2(3,tid2),'.r','MarkerSize',20)
% tid2=find(Zidx2==7);
% plot3(TrialZ_online2(1,tid2),TrialZ_online2(2,tid2),TrialZ_online2(3,tid2),'.g','MarkerSize',20)
% xlim(xx)
% ylim(yy)
% zlim(zz)
% view(-134,36)
% set(gcf,'Color','w')
% xlabel('Latent 1')
% ylabel('Latent 2')
% zlabel('Latent 3')
% set(gca,'FontSize',14)
% tmp=(diag(acc1));
% title(['Day ' num2str(j) ' thru AE 78 with acc ' num2str(mean(tmp([1 5 7])))])
%
% % plotting
%


% regression lines for accuracy over days
days=1:9;
figure;hold on
plot(days,tmp,'.k','MarkerSize',20)
x = [ones(length(days),1) days'];
y = tmp';
[B,BINT,R,RINT,STATS] = regress(y,x);
yhat = x*B;
plot(days,yhat,'k','LineWidth',1)
% [bhat p wh se ci t_stat]=robust_fit((1:length(tmp))',tmp',1);
% yhat1 = x*bhat;
% plot(days,yhat1,'k','LineWidth',1)
xlim([.5 9.5])
xticks([1:9])
xticklabels({'1','1-2','1-3','1-4','1-5','1-6','1-7','1-8','1-9'})
%ylim([4 10])
set(gcf,'Color','w')
yticks(4:2:12)
ylim([4 12])


% robust regression lines for accuracy over days
days=1:9;
figure;hold on
plot(days,tmp,'.k','MarkerSize',20)
x = [ones(length(days),1) days'];
y = tmp';
[B,BINT,R,RINT,STATS] = regress(y,x);
yhat = x*B;
plot(days,yhat,'k','LineWidth',1)
[bhat p wh se ci t_stat]=robust_fit((1:length(tmp))',tmp',1);
yhat1 = x*bhat;
plot(days,yhat1,'b','LineWidth',1)
xlim([.5 9.5])
xticks([1:9])
xticklabels({'1','1-2','1-3','1-4','1-5','1-6','1-7','1-8','1-9'})
%ylim([4 10])
set(gcf,'Color','w')
yticks(4:2:12)
ylim([4 12])



% regression lines for accuracy over days excluding the last day
days=1:8;
tmp=tmp(1:8);
figure;hold on
plot(days,tmp,'.k','MarkerSize',20)
x = [ones(length(days),1) days'];
y = tmp';
[B,BINT,R,RINT,STATS] = regress(y,x);
yhat = x*B;
plot(days,yhat,'k','LineWidth',1)
% [bhat p wh se ci t_stat]=robust_fit((1:length(tmp))',tmp',1);
% yhat1 = x*bhat;
% plot(days,yhat1,'k','LineWidth',1)
xlim([.5 8.5])
xticks([1:8])
xticklabels({'1','1-2','1-3','1-4','1-5','1-6','1-7','1-8'})
%ylim([4 10])
set(gcf,'Color','w')
yticks(35:5:60)
ylim([35 60])
title('Accuracy in latent space')
xlabel('Days used to build latent space')
ylabel('Accuracy of day 10 data')






%% LOOKING AT WHETHER THE PRIN ANGLES OF MANIFOLDS CHANGE FROM IMAGINED TO ONLINE (7DOF)

% a few options
% FIRST is to take distribution of prin angles between imagined files per
% action to other imagined files as compared to prin angles between
% imagined files and online files
% SECOND is to compare the angles between the 7 actions and see if there is
% greater separation during online vs imagined

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\OneDrive\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
load session_data

for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    if i~=6
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_imag(folders_am==0)=0;
        folders_online(folders_am==0)=0;
    end

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);

    %%%%%%imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end


    %%%%%%online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files1=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files1 = [files1;findfiles('',folderpath)'];
    end

    % get the principal angles between the various commands
    prin_angles = get_prin_angles(files,files1);

end





%% COLLATE AE ACTIVATION DURING PNP DAYS
% INVOLVES FIRST BUILDING THE AE SPACE FOR ALL DATA FROM FIRST 11 DAYS


clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\OneDrive\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
load session_data

% option of maybe aligning all the data to the first day using procrustus
files=[];
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);


    %%%%%%imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %%%%%%online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end

    %%%%%%batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end
end

% build a massive autoencoder by loading all the data from all 11day files
condn_data = load_data_for_MLP(files);
[net,Xtrain,Ytrain] = build_mlp_AE_supervised_total(condn_data);

% save the AE
net_AE_total=net;
save net_AE_total net_AE_total



%% LOOKING AT TEMPORAL DYNAMICS OF AE LATENT SPACE ACTIVATION IN THE PATH TASK

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\OneDrive\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
foldernames={'20210825','20210827'} %20210818

% load the total AE
load net_AE_total

% get the files
files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},...
        'Robot3DPath');
    files = [files;findfiles('',folderpath,1)'];
end

% remove persistence files
files1=[];
for i=1:length(files)
    if ~isempty(regexp(files{i},'Data'))
        files1=[files1;files(i)];
    end
end
files=files1;

% get all the paths
Params.Paths{1} = [[200, -200, 0]; [0, -200, 0]; [0,0,0];  [-200, 0,0]; [-200, 200, 0]];
Params.Paths{2} = [[-200, 200, 0]; [-200, 0,0]; [0,0,0]; [0, -200, 0]; [200, -200, 0]];
Params.Paths{3} = [[200,0,-100]; [200, 0, 100]; [0, 0, 100];[0, 0, -100]; [-200, 0, -100]];
Params.Paths{4} = [ [-200, 0, -100]; [0, 0, -100]; [0, 0, 100];[200, 0, 100]; [200,0,-100]];
Params.Paths{5} = [[200, -100, 0]; [0, -100, 0]; [-141.4, 41.4, 0]; [0,180, 0]; [200, 180, 0]];
Params.Paths{6} = [[200, 180, 0];[0,180, 0]; [-141.4, 41.4, 0]; [0, -100, 0];[200, -100, 0]];

% load a specific file and look at latent activity
load(files{1})
target_id=TrialData.TargetID;
target_pos = TrialData.TargetPosition;
target_path = Params.Paths{target_id};
%target_path=flipud(target_path);
%target_path = cumsum(target_path);
start_pos = TrialData.CursorState(:,1);
kin = TrialData.CursorState;
kinax = TrialData.TaskState;
kinax = [find(kinax==3)];
kin=kin(:,kinax);

% plot the data
cmap=turbo(size(kin,2));
figure;
for i=1:size(kin,2)
    plot3(kin(1,i),kin(2,i),kin(3,i),'.','MarkerSize',30,'Color',cmap(i,:))
    hold on
end
plot3(kin(1,1),kin(2,1),kin(3,1),'xg','MarkerSize',30)
plot3(target_pos(1),target_pos(2),target_pos(3),'xr','MarkerSize',30)
for i=size(target_path,1):-1:1
    plot3(target_path(i,1),target_path(i,2),target_path(i,3),'ob','MarkerSize',30)
end

%%%%% project data into latent space
clicker_output = TrialData.FilteredClickerState;
features  = TrialData.SmoothedNeuralFeatures;
temp = cell2mat(features(kinax));
chmap = TrialData.Params.ChMap;
X = bci_pooling(temp,chmap);
%2-norm the data
for j=1:size(X,2)
    X(:,j)=X(:,j)./norm(X(:,j));
end
% feed it through the AE
X = X(1:96,:);
Z = activations(net_AE_total,X','autoencoder');
figure;hold on
view(56,35)
%cmap=parula(size(Z,2));
cmap=parula(8);
for i=1:size(Z,2)
    decode = clicker_output(i)+1;
    plot3(Z(1,i),Z(2,i),Z(3,i),'.','MarkerSize',20,'Color',cmap(decode,:))
    pause(0.05)
end

% plot video together
v = VideoWriter('PathTrial3.avi');
v.FrameRate=12;
open(v);
figure;
set(gcf,'Color','w')
subplot(2,1,1)
hold on
xlim([min(kin(1,:)) max(kin(1,:))])
ylim([min(kin(2,:)) max(kin(2,:))])
%zlim([min(kin(3,:)) max(kin(3,:))])
zlim([-1 1])
for ii=size(target_path,1):-1:1
    plot3(target_path(ii,1),target_path(ii,2),target_path(ii,3),'ob','MarkerSize',30)
end
hold on
plot3(target_pos(1),target_pos(2),target_pos(3),'xr','MarkerSize',30)
view(56,35)
axis tight
cmap=parula(7);
for i=1:size(Z,2)

    subplot(2,1,1)
    plot3(kin(1,i),kin(2,i),kin(3,i),'.k','MarkerSize',30)
    hold on

    subplot(2,1,2)
    view(56,35)
    xlim([-1 4])
    ylim([-1 3])
    zlim([-0.5 2])
    decode = clicker_output(i);
    if decode>0
        plot3(Z(1,i),Z(2,i),Z(3,i),'.','MarkerSize',20,'Color',cmap(decode,:))
        hold on
    end
    pause(0.05)

    frame=getframe(gcf);
    writeVideo(v,frame);
end
close(v)

%% LOOKING AT WHETHER THE MANIFOLDS CHANGE FROM IMAGINED TO ONLINE (HAND)



%% SESSION DATA FOR B2


clc;clear
session_data=[];
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2';
cd(root_path)

% IMAGINED DATA FOLDER IS CENTER OUT AND ONLINE DATA IS IN DISCRETE ARROW

%day1
session_data(1).Day = '20210203';
session_data(1).folders = {'145654','150544','154256'};
session_data(1).folder_type={'I','I','O'};
session_data(1).AM_PM = {'pm','pm','pm'};

%day2
session_data(2).Day = '20210210';
session_data(2).folders = {'143225','151341','151744','154405','155813'};%,'155813'154405
session_data(2).folder_type={'I','O','O','B','B'};%,'B'
session_data(2).AM_PM = {'pm','pm','pm','pm'};

session_data(3).Day = '20210217';
session_data(3).folders = {'142905','144942','145755','150256','150450','150848','151056'};
session_data(3).folder_type={'I','O','O','O','O','B','B'};
session_data(3).AM_PM = {'pm','pm','pm','pm','pm','pm','pm'};

session_data(4).Day = '20210224';
session_data(4).folders = {'143622','161431','163333','163522','163826','164805','164958'};%'165313'
session_data(4).folder_type={'I','I','O','O','O','B','B'};%'B'
session_data(4).AM_PM = {'pm','pm'};

session_data(5).Day = '20210324';
session_data(5).folders = {'140739','142834','143514','144108'};
session_data(5).folder_type={'I','O','O','B'};%'B'
session_data(5).AM_PM = {'pm','pm','pm','pm'};

% session_data(6).Day = '20210324';
% session_data(6).folders = {'144837','152426','153046'};
% session_data(6).folder_type={'I','O','O'};%'B'
% session_data(6).AM_PM = {'pm','pm','pm'};

session_data(6).Day = '20210407';
session_data(6).folders = {'153243','155425'};
session_data(6).folder_type={'I','O'};%'B'
session_data(6).AM_PM = {'pm','pm'};


save session_data_B2 session_data -v7.3

%% SESSION DATA FOR B3


clc;clear
session_data=[];
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
cd(root_path)

% IMAGINED DATA FOLDER IS CENTER OUT AND ONLINE DATA IS IN DISCRETE ARROW

% day 0 %%%% GET THE DATA AGAIN FROM THE PC AS IT HAS NOT BEEN SAVED
k=1;
session_data(k).Day = '20230223';
session_data(k).folders = {'125028','125649','130309','130627',...
    '133358','133843','134055',...
    '140223','140438'};
session_data(k).folder_type={'I','I','I','I','O','O','O',...
   'B','B'};
session_data(k).AM_PM = {'am','am','am','am','am','am','am','am','am'};


%day1
k=2;
session_data(k).Day = '20230301';
session_data(k).folders = {'113743','114316','114639','114958','120038','120246',...
    '120505','120825','121458','122238','122443','122641','122858'};
session_data(k).folder_type={'I','I','I','I','O','O','O','O','B','B','B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am','am','am','am','am','am','am','am','am'};

%day2
k=3;
session_data(k).Day = '20230302';
session_data(k).folders = {'122334','122931','123406','124341','125002',...
    '125915','130405','130751','131139','131614',...
    '132424','132824','133236','133742'};
session_data(k).folder_type={'I','I','I','I','I','O','O','O','O','O',...
   'B','B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am','am','am','am','am','am','am','am','am','am'};

%day3
k=4;
session_data(k).Day = '20230308';
session_data(k).folders = {'114109','114632','114940','115300','115621',...
    '120914','121201','121443','121702','121926',...
    '122749','123008','123237','123447','123846'};
session_data(k).folder_type={'I','I','I','I','I','O','O','O','O','O',...
   'B','B','B','B','B'};
session_data(k).AM_PM = {'am','am','am','am','am','am','am','am','am','am','am','am','am','am','am'};

%day4
k=5;
session_data(k).Day = '20230309';
session_data(k).folders = {'135628','140326','140904','141504','142051',...
    '143001','143435','143906','144336','144737',...
    '145814','150749'};
session_data(k).folder_type={'I','I','I','I','I','O','O','O','O','O',...
   'B','B'};
session_data(k).AM_PM = {'am','am','am','am','am','am','am','am','am','am','am','am'};





save session_data_B3 session_data -v7.3

%% PLASTICITY AND AE FRAMEWORK FOR B2 ARROW DATA (MAIN)
% good days: 20210324

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
cd(root_path)
load session_data_B2
dist_online_total=[];
dist_imag_total=[];
var_imag_total=[];
mean_imag_total=[];
var_online_total=[];
mean_online_total=[];
res=[];
mahab_full_online=[];
mahab_full_imagined=[];
mahab_full_batch=[];
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);

    %%%%%%imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'CenterOut',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    load('ECOG_Grid_8596-002131.mat')
    condn_data = load_data_for_MLP_B2(files,ecog_grid);

    %save the data
    filename = ['B2_condn_data_Imagined_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    % build the AE based on MLP and only for hG
    %[net,Xtrain,Ytrain] = build_mlp_AE_B2(condn_data);
    %[net,Xtrain,Ytrain] = build_mlp_AE_supervised(condn_data);

    % get the mahab distance in the full dataset
    Dimagined = mahal2_full(condn_data);
    Dimagined = triu(Dimagined);
    Dimagined = Dimagined(Dimagined>0);
    mahab_full_imagined = [mahab_full_imagined Dimagined];

    % get activations in deepest layer but averaged over a trial
    %     imag=1;
    %     [TrialZ_imag,dist_imagined,mean_imagined,var_imagined,idx_imag] = ...
    %         get_latent_regression_B2(files,net,imag,ecog_grid);
    %     dist_imag_total = [dist_imag_total;dist_imagined];
    %     mean_imag_total=[mean_imag_total;pdist(mean_imagined)];
    %     var_imag_total=[var_imag_total;var_imagined'];

    %%%%%%online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'DiscreteArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    condn_data = load_data_for_MLP_B2(files,ecog_grid);

    % save the data
    filename = ['B2_condn_data_Online_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    % get the mahab distance in the full dataset
    Donline = mahal2_full(condn_data);
    Donline = triu(Donline);
    Donline = Donline(Donline>0);
    mahab_full_online = [mahab_full_online Donline];


    % get activations in deepest layer
    %     imag=0;
    %     [TrialZ_online,dist_online,mean_online,var_online,idx_online] = ...
    %         get_latent_regression_B2(files,net,imag,ecog_grid);
    %     dist_online_total = [dist_online_total;dist_online];
    %     mean_online_total=[mean_online_total;pdist(mean_online)];
    %     var_online_total=[var_online_total;var_online'];

    %%%%%%batch data
    if ~isempty(batch_idx)
        folders = session_data(i).folders(batch_idx);
        day_date = session_data(i).Day;
        files=[];
        for ii=1:length(folders)
            folderpath = fullfile(root_path, day_date,'DiscreteArrow',folders{ii},'BCI_Fixed');
            files = [files;findfiles('',folderpath)'];
        end


        %load the data
        condn_data = load_data_for_MLP_B2(files,ecog_grid);

        % save the data
        filename = ['B2_condn_data_Batch_Day' num2str(i)];
        save(filename, 'condn_data', '-v7.3')

        % get the mahab distance in the full dataset
        Donline = mahal2_full(condn_data);
        Donline = triu(Donline);
        Donline = Donline(Donline>0);
        mahab_full_batch = [mahab_full_batch Donline];
    else
        mahab_full_batch = [mahab_full_batch zeros(6,1)];
    end

    % get activations in deepest layer
    %     imag=0;
    %     [TrialZ_batch,dist_batch,mean_batch,var_batch,idx_batch] = get_latent_regression(files,net,imag);
    %     dist_batch_total = [dist_batch_total;dist_batch];
    %     mean_batch_total=[mean_batch_total;pdist(mean_batch)];
    %     var_batch_total=[var_batch_total;var_batch'];





    % plotting imagined and online in latent space
    %     idxa = find(idx_imag==4);
    %     idxb = find(idx_online==4);
    %     idxa = idxa(randperm(length(idxa),length(idxb)));
    %     figure;hold on
    %     plot3(TrialZ_imag(1,idxa),TrialZ_imag(2,idxa),TrialZ_imag(3,idxa),'.','MarkerSize',20)
    %     plot3(TrialZ_online(1,idxb),TrialZ_online(2,idxb),TrialZ_online(3,idxb),'.','MarkerSize',20)
    %     c1 = TrialZ_imag(:,idxa);
    %     c2 = TrialZ_online(:,idxb);
    %     c1=cov(c1');
    %     c2=cov(c2');

    %      plot
    %
    %     figure;boxplot([dist_imagined' dist_online'])
    %     box off
    %     set(gcf,'Color','w')
    %     xticks(1:2)
    %     xticklabels({'Imagined Data','Online Data'})
    %     ylabel('Distance')
    %     title('Inter-class distances')
    %     set(gca,'LineWidth',1)
    %     set(gca,'FontSize',12)

    %     [h p tb st]=ttest(dist_imagined,dist_online);
    %     disp([p mean([dist_imagined' dist_online'])]);
    %     res=[res;[p mean([dist_imagined' dist_online'])]];
end

save mahab_dist_full_B2 -v7.3

figure;boxplot([mahab_full_imagined mahab_full_online mahab_full_batch])

figure;plot(1:6,mean(mahab_full_imagined));
hold on
plot(1:6,mean(mahab_full_online))
plot(2:5,mean(mahab_full_batch(:,2:5)))

figure;
plot(mean(dist_online_total'))
set(gcf,'Color','w')
title('Across Day Learning')
ylabel('Mahalanobis Dist.')
xlabel('Day')
xlim([0.5 11.5])
tmp = mean(dist_online_total');
figure;
tmp1 = tmp(1:4);
tmp2 = tmp(5:end);
tmp1(end+1:length(tmp2))=NaN;
boxplot([tmp1' tmp2'])
xticklabels({'Early Days','Late Days'})
ylabel('Mahalanobis Dist')
title('Online Bins proj. thru Imagined Manifold')
set(gcf,'Color','w')


% stats on distances, early days and late days online mahalanobis distances
early_days_online = dist_online_total(1:4,:);
early_days_online=early_days_online(:);
late_days_online = dist_online_total(5:end,:);
late_days_online=late_days_online(:);
early_days_online(end+1:length(late_days_online))=NaN;
figure;boxplot([early_days_online late_days_online ]);


tmp = mean(dist_imag_total');
figure;
tmp1 = tmp(1:4);
tmp2 = tmp(5:end);
tmp1(end+1:length(tmp2))=NaN;
boxplot([tmp1' tmp2'])
xticklabels({'Early Days','Late Days'})
ylabel('Mahalanobis Dist')
title('Imag Bins proj. thru Imagined Manifold')
set(gcf,'Color','w')

figure;
boxplot([dist_imag_total(1,:)' dist_online_total(1,:)' ])


%plotting the difference in manifold angles between mvmt and time
ang = [mean(dist_imag_total,2) mean(dist_online_total,2)];
%ang=fliplr(ang);
figure;hold on
%scatter(ones(length(ang),1)+0.05*randn(length(ang),1),ang(:,1));
%scatter(2*ones(length(ang),1)+0.05*randn(length(ang),1),ang(:,2));
idx = 0.05*randn(length(ang),2);
%idx = zeros(size(idx));
scatter(ones(length(ang),1)+idx(:,1),ang(:,1),100);
scatter(2*ones(length(ang),1)+idx(:,2),ang(:,2),100)
xlim([0.5 2.5])
col=winter(length(ang));
for i=1:length(ang)
    plot([1 2]+idx(i,:),(ang(i,:)),'Color',[.5 .5 .5 .5],'LineWidth',1)
end
ylim([5 65]) %0 to 40 for regression
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Mahalanobis distance')
set(gca,'FontSize',14)

%plotting changes in the variance of the latent distributions
figure;boxplot([var_imag_total(:) var_online_total(:)])
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Variance in latent space')
set(gca,'FontSize',14)
set(gcf,'Color','w')
set(gca,'LineWidth',2)
box off

ang=[mean(var_imag_total,2) mean(var_online_total,2)];
figure;hold on
idx = 0.05*randn(length(ang),2);
scatter(ones(length(ang),1)+idx(:,1),ang(:,1),100);
scatter(2*ones(length(ang),1)+idx(:,2),ang(:,2),100)
xlim([0.5 2.5])
col=parula(length(ang));
for i=1:length(ang)
    %plot([1 2]+idx(i,:),(ang(i,:)),'Color',[.5 .5 .5 .5],'LineWidth',1)
    plot([1 2]+idx(i,:),(ang(i,:)),'Color',col(i,:),'LineWidth',1)
end
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Variance in latent space')
set(gca,'FontSize',14)
[h p tb st]=ttest(ang(:,1),ang(:,2))

figure;
plot(ang(:,2)-ang(:,1),'.k','MarkerSize',20)
tmp=ang(:,2)-ang(:,1);

% plotting changes in the mean distance between distributions over learning
figure;boxplot([mean_imag_total(:) mean_online_total(:)])
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Mean Diff in latent space')
set(gca,'FontSize',14)
set(gcf,'Color','w')
set(gca,'LineWidth',2)
box off

ang=[mean(mean_imag_total,2) mean(mean_online_total,2)];
figure;hold on
idx = 0.05*randn(length(ang),2);
scatter(ones(length(ang),1)+idx(:,1),ang(:,1),100);
scatter(2*ones(length(ang),1)+idx(:,2),ang(:,2),100)
xlim([0.5 2.5])
col=parula(length(ang));
for i=1:length(ang)
    %plot([1 2]+idx(i,:),(ang(i,:)),'Color',[.5 .5 .5 .5],'LineWidth',1)
    plot([1 2]+idx(i,:),(ang(i,:)),'Color',col(i,:),'LineWidth',1)
end
set(gcf,'Color','w')
xticks(1:2)
xticklabels({'Imagined Data','Online Data'})
ylabel('Distance b/w means in latent space')
set(gca,'FontSize',14)
[h p tb st]=ttest(ang(:,1),ang(:,2))

figure;
plot(ang(:,2)-ang(:,1),'.k','MarkerSize',20)
tmp=ang(:,2)-ang(:,1);
[bhat p wh se ci t_stat]=robust_fit((1:length(tmp))',tmp,1);
hold on
plot([ (1:length(tmp))'],...
    [ ones(size(tmp,1),1) (1:length(tmp))']*bhat,'k','LineWidth',1);
xlim([0.5 12])
xlabel('Days')
ylabel('Delta Online vs. Imagined')
title('Mean Separation in Latent Space')
set(gca,'FontSize',14)
set(gcf,'Color','w')
box off

% boostrapped test
bhat_boot=[];
parfor iter=1:500
    x=1:length(tmp);
    x=x(randperm(length(x)));
    [bhat1 p wh se ci t_stat]=robust_fit(x',tmp,1);
    bhat_boot(iter)=bhat1(2);
end
sum(bhat_boot>bhat(2))/length(bhat_boot)

x= [ ones(size(tmp,1),1) (1:length(tmp))'];
y = tmp;
[B,BINT,R,RINT,STATS] = regress(y,x);
STATS(3)


a=randn(3)
idx = randi(3,[3,3])
a(idx)



% save res_python_B2_latent -v7.3

%%%% regressions for mahab on full as a function of day
clear tmp
tmp(:,1) = (median(mahab_full_imagined,1));
tmp(:,2) = (median(mahab_full_online,1));
tmp(:,3) = (median(mahab_full_batch,1));
tmp = tmp(2:5,:);
% for i=1:size(tmp,2)
%     tmp(:,i) = smooth(tmp(:,i));
% end

% plotting the regression for Mahab distance increases as a function of day
figure;
xlim([0 5])
hold on
x= [ ones(size(tmp(:,1),1),1) (1:length(tmp(:,1)))'];
% imag
plot(1:4,tmp(:,1),'.b','MarkerSize',20)
y = tmp(:,1);
[B,BINT,R,RINT,STATS1] = regress(y,x);
yhat = x*B;
plot(1:4,yhat,'b','LineWidth',1)
% online
plot(1:4,tmp(:,2),'.k','MarkerSize',20)
y = tmp(:,2);
[B,BINT,R,RINT,STATS2] = regress(y,x);
yhat = x*B;
plot(1:4,yhat,'k','LineWidth',1)
% batch
plot(1:4,tmp(:,3),'.r','MarkerSize',20)
y = tmp(:,3);
[B,BINT,R,RINT,STATS3] = regress(y,x);
yhat = x*B;
plot(1:4,yhat,'r','LineWidth',1)
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xlim([0.5  4.5])
xticks([1:4])
%yticks([5:5:35])
%ylim([5 35])



% plotting the robust regression for Mahab distance increases as a function of day
figure;
xlim([0 5])
hold on
x= [ ones(size(tmp(:,1),1),1) (1:length(tmp(:,1)))'];
% imag
plot(1:4,tmp(:,1),'.b','MarkerSize',20)
[bhat p1 wh se ci t_stat]=robust_fit((1:length(tmp))',tmp(:,1),2);
yhat = x*bhat;
plot(1:4,yhat,'b','LineWidth',1)
% online
plot(1:4,tmp(:,2),'.k','MarkerSize',20)
[bhat p2 wh se ci t_stat]=robust_fit((1:length(tmp))',tmp(:,2),2);
yhat = x*bhat;
plot(1:4,yhat,'k','LineWidth',1)
% batch
plot(1:4,tmp(:,3),'.r','MarkerSize',20)
[bhat p3 wh se ci t_stat]=robust_fit((1:length(tmp))',tmp(:,3),2);
yhat = x*bhat;
plot(1:4,yhat,'r','LineWidth',1)


% bootstrapped test for regression coefficient
y = tmp(:,3);
[B2,BINT,R,RINT,STATS2] = regress(y,x);
bhat_boot=[];
for i=1:2000
    ytmp = y(randperm(numel(y)));
    [B_tmp,~,Rtmp,~,STATS_tmp] = regress(ytmp,x);
    bhat_boot(i) = B_tmp(2);
end
figure;hist(bhat_boot)
vline(B2(2),'r')
sum(bhat_boot>B2(2))/length(bhat_boot)

% mixed effect model for batch and Init seed slopes
day_name=[];
mahab_dist=[];
for i=1:size(tmp,1)
    day_name = [day_name;i;i];
    mahab_dist = [mahab_dist;tmp(i,2:3)'];
end
data = table(day_name,mahab_dist);
glm = fitglme(data,'mahab_dist ~ 1 + day_name');
stat = glm.Coefficients.tStat(2);
stat_boot=[];
for i=1:1000
    day_name_tmp = day_name(randperm(numel(day_name)));
    data_tmp = table(day_name_tmp,mahab_dist);
    glm_tmp = fitglm(data_tmp,'mahab_dist ~ 1 + day_name_tmp');
    stat_boot(i) = glm_tmp.Coefficients.tStat(2);
end
figure;hist(stat_boot)
vline(stat)
sum(stat_boot>stat)/length(stat_boot)

% getting the mixed effect model stats for latent variance in  B2
%save latent_variance_B2 -v7.3
% fitting the data into a RM anova format
tmp=[tmp];
Data_RM_B2_latentVar=[];
for i=1:length(tmp)
    exp_type = [1 2 3]';
    subject = [i i i]';
    lat_var = tmp(i,:)';
    Data_RM_B2_latentVar = [Data_RM_B2_latentVar;[exp_type subject lat_dist]];
end
save Data_RM_B2_latentVar Data_RM_B2_latentVar -v7.3

% putting into a mixed effct model format
day_name=[];
latent_var=[];
exp_type=[];
subject=[];
I = 1:4:length(tmp);
for i=1:length(I)
    tmp1 = tmp(I(i):I(i)+3,:);
    for j=1:length(tmp1)
        exp_type = [exp_type;[1 2 3]'];
        latent_var = [latent_var;tmp1(j,:)'];
        day_name = [day_name;[i i i]'];
        subject = [subject;[1 1 1]'];
    end
end
data=table(exp_type,latent_var,day_name,subject);
glme = fitglme(data,'latent_var ~ 1 + exp_type +(1|day_name)+(1|subject)')

% doing paired t-tests
I = sort([find(data.exp_type==2);find(data.exp_type==3)]);
data1=data(I,:);
glme = fitglme(data1,'latent_var ~ 1 + exp_type +(1|day_name)+(1|subject)')

%% B2: looking at decoding performance from imagined -> online -> batch (MAIN)
% across days

clc;clear;
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data_B2
addpath 'C:\Users\nikic\Documents\MATLAB'
acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
iterations=5;
plot_true=false;
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);
    %disp([session_data(i).Day '  ' num2str(length(batch_idx))]);

    %%%%%% cross_val classification accuracy for imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'CenterOut',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    load('ECOG_Grid_8596-002131.mat')
    condn_data = load_data_for_MLP_TrialLevel_B2(files,ecog_grid);

    % get cross-val classification accuracy
    [acc_imagined,train_permutations] = accuracy_imagined_data_B2(condn_data, iterations);
    acc_imagined=squeeze(nanmean(acc_imagined,1));
    if plot_true
        figure;imagesc(acc_imagined)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
    end
    acc_imagined_days(:,i) = diag(acc_imagined);


    %%%%%% get classification accuracy for online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'DiscreteArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    % get the classification accuracy
    acc_online = accuracy_online_data_B2(files);
    if plot_true
        figure;imagesc(acc_online)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
    end
    acc_online_days(:,i) = diag(acc_online);


    %%%%%% cross_val classification accuracy for batch data
    if ~isempty(batch_idx)
        folders = session_data(i).folders(batch_idx);
        day_date = session_data(i).Day;
        files=[];
        for ii=1:length(folders)
            folderpath = fullfile(root_path, day_date,'DiscreteArrow',folders{ii},'BCI_Fixed');
            %cd(folderpath)
            files = [files;findfiles('',folderpath)'];
        end

        % get the classification accuracy
        acc_batch = accuracy_online_data_B2(files);
        if plot_true
            figure;imagesc(acc_batch)
            colormap bone
            clim([0 1])
            set(gcf,'color','w')
        end
        acc_batch_days(:,i) = diag(acc_batch);
    else
        acc_batch_days(:,i) = NaN*ones(4,1);
    end
end

load ('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\b1_acc_rel_imagined_prop.mat')
save hDOF_6days_accuracy_results_New_B2 -v7.3
%save hDOF_10days_accuracy_results -v7.3


%acc_online_days = (acc_online_days + acc_batch_days)/2;
figure;
ylim([0.0 0.65])
xlim([0.5 6.5])
hold on
plot(nanmean(acc_imagined_days,1))
plot(nanmean(acc_online_days,1))
plot(nanmean(acc_batch_days,1),'k')

% as regression lines
figure;plot(mean(acc_imagined_days,1),'.','MarkerSize',20)

% stats
tmp = [median(acc_imagined_days,1)' median(acc_online_days,1)' ...
    median(acc_batch_days,1)'];

figure;boxplot(acc_imagined_days)
ylim([0.2 1])
xlim([0.5 10.5])
hold on
boxplot(acc_batch_days,'Colors','k')
a = get(get(gca,'children'),'children');

figure;
boxplot([acc_imagined_days(:) acc_online_days(:) acc_batch_days(:)])

m1 = (acc_imagined_days(:));
m1b = sort(bootstrp(1000,@mean,m1));
m11 = mean(acc_imagined_days,1);
m2 = (acc_online_days(:));
m2b = sort(bootstrp(1000,@mean,m2));
m22 = mean(acc_online_days,1);
m3 = (acc_batch_days(:));
m3b = sort(bootstrp(1000,@nanmean,m3));
m33 = nanmean(acc_batch_days,1);
x=1:3;
y=[mean(m1) mean(m2) nanmean(m3)];
s1=std(m1)/sqrt(length(m1));
s2=std(m2)/sqrt(length(m2));
s3=nanstd(m3)/sqrt(sum(~isnan(m3)));
neg = [s1 s2 s3];
pos= [s1 s2 s3];
%neg = [y(1)-m1b(25) y(2)-m2b(25) y(3)-m3b(25)];
%pos = [m1b(975)-y(1) m2b(975)-y(2) m3b(975)-y(3)];
figure;
hold on
cmap = brewermap(6,'Blues');
%cmap = (turbo(7));
for i=1:3
    errorbar(x(i),y(i),neg(i),pos(i),'Color','k','LineWidth',1)
    plot(x(i),y(i),'o','MarkerSize',20,'Color','k','LineWidth',1,'MarkerFaceColor',[.5 .5 .5])
end
for i=1:size(acc_batch_days,2)
    plot(1+0.1*randn(1),m11(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',5,'Color',[cmap(end,:) .5])
    plot(2+0.1*randn(1),m22(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',5,'Color',[cmap(end,:) .5])
    plot(3+0.1*randn(1),m33(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',5,'Color',[cmap(end,:) .5])
end
xlim([.5 3.5])
ylim([0.10 0.55])
xticks(1:3)
xticklabels({'Imagined','Online','Batch'})
set(gcf,'Color','w')
set(gca,'LineWidth',1)
yticks(0:.1:1)
set(gca,'FontSize',12)
h=hline(.25,'--');
h.LineWidth=1;
xlabel('Decoder Type')
ylabel('Accuracy')

tmp = [ m11' m22' m33'];
figure;boxplot(tmp)

tmp = [ m1 m2 m3];
figure;boxplot(tmp)

% regression lines for mahab distances in latent space
imag = [0.732087
    0.471915
    0.526922
    0.187374
    0.611636
    0.501813
    ];
online=[4.09371
    1.46306
    0.93976
    2.13517
    1.05116
    1.34932
    ];
batch=[1.58127
    2.48264
    3.26464
    3.10718
    ]; % days 12 thru 5

% PLOT REGRESSION LINES
imag = tmp(:,1);
online = tmp(:,2);
batch = tmp(1:4,3);
figure;
hold on
days=1:4;
x=[ones(length(days),1) days'];
%imag
plot(days,imag(2:5),'.','MarkerSize',20)
y=imag(2:5);
[B,BINT,R,RINT,STATS1] = regress(y,x);
yhat=x*B;
plot(days,yhat,'b','LineWidth',1)
%online
plot(days,online(2:5),'.k','MarkerSize',20)
y=online(2:5);
[B,BINT,R,RINT,STATS2] = regress(y,x);
yhat=x*B;
plot(days,yhat,'k','LineWidth',1)
%batch
plot(days,batch,'.r','MarkerSize',20)
y=batch;
[B,BINT,R,RINT,STATS3] = regress(y,x);
yhat=x*B;
plot(days,yhat,'r','LineWidth',1)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlim([.5 4.5])
xticks(1:4)
xticklabels(1:4)
ylim([0 3.5])
yticks([0:3])


% PLOT REGRESSION LINES equal size in tmp
imag = tmp(:,1);
online = tmp(:,2);
batch = tmp(:,3);
figure;
hold on
days=1:4;
x=[ones(length(days),1) days'];
%imag
plot(days,imag,'.','MarkerSize',20)
y=imag;
[B,BINT,R,RINT,STATS1] = regress(y,x);
yhat=x*B;
plot(days,yhat,'b','LineWidth',1)
%online
plot(days,online,'.k','MarkerSize',20)
y=online;
[B,BINT,R,RINT,STATS2] = regress(y,x);
yhat=x*B;
plot(days,yhat,'k','LineWidth',1)
%batch
plot(days,batch,'.r','MarkerSize',20)
y=batch;
[B,BINT,R,RINT,STATS3] = regress(y,x);
yhat=x*B;
plot(days,yhat,'r','LineWidth',1)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlim([.5 4.5])
xticks(1:4)
xticklabels(1:4)



% using robust regression in matlab equal size
figure;
hold on
% imag
plot(days,imag,'.b','MarkerSize',20)
y=imag;
lm=fitlm(x(:,2:end),y,'Robust','on');
B=lm.Coefficients.Estimate;
yhat = x*B;
plot(days,yhat,'b','LineWidth',1)
% online
plot(days,online,'.k','MarkerSize',20)
y=online;
lm=fitlm(x(:,2:end),y,'Robust','on');
B=lm.Coefficients.Estimate;
yhat = x*B;
plot(days,yhat,'k','LineWidth',1)
% batch
plot(days,batch,'.r','MarkerSize',20)
y=batch;
lm=fitlm(x(:,2:end),y,'Robust','on');
B=lm.Coefficients.Estimate;
yhat=x*B;
plot(days,yhat,'r','LineWidth',1)
% beautify
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xticks([1:4])



% using robust regression in matlab
figure;
hold on
% imag
plot(days,imag(2:5),'.b','MarkerSize',20)
y=imag(2:5);
lm=fitlm(x(:,2:end),y,'Robust','on');
B=lm.Coefficients.Estimate;
yhat = x*B;
plot(days,yhat,'b','LineWidth',1)
% online
plot(days,online(2:5),'.k','MarkerSize',20)
y=online(2:5);
lm=fitlm(x(:,2:end),y,'Robust','on');
B=lm.Coefficients.Estimate;
yhat = x*B;
plot(days,yhat,'k','LineWidth',1)
% batch
plot(days,batch,'.r','MarkerSize',20)
y=batch;
lm=fitlm(x(:,2:end),y,'Robust','on');
B=lm.Coefficients.Estimate;
yhat=x*B;
plot(days,yhat,'r','LineWidth',1)
% beautify
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xticks([1:4])
yticks([5:5:35])
ylim([0 1])





% stats on the latent space variance
tmp=[0.000937406	0.0985546	0.0289443
    0.00154575	0.0315809	0.0753841
    0.00141096	0.0103554	0.20185
    0.00043903	0.0643351	0.150029
    0.00125143	0.00337939	0.112707
    0.00113499	0.00377055	0.112707
    ];
[P,ANOVATAB,STATS] = anova1(tmp);

[h p tb st] = ttest(tmp(:,1),tmp(:,2));p
[h p tb st] = ttest(tmp(:,1),tmp(:,3));p
[h p tb st] = ttest(tmp(:,2),tmp(:,3));p

% how about bootstrapped difference of means?
stat  =  mean(tmp(:,2)) - mean(tmp(:,1));
c= [tmp(:,2);tmp(:,1)];
l=size(tmp,1);
boot=[];
c=c-mean(c);
for i=1:5000
    idx=randperm(numel(c));
    a1 = c(idx(1:l));
    b1 = c(idx(l+1:end));
    boot(i) = mean(a1)-mean(b1);
end
figure;hist((boot))
vline((stat),'r')
sum(boot>stat)/length(boot)


% get the accuracies relative to imagined movement within that day
a0 = mean(acc_imagined_days(:,2:5),1);
a1 = mean(acc_online_days(:,2:5),1);
a2 = mean(acc_batch_days(:,2:5),1);
figure;
plot(a0);
hold on
plot(a1);
plot(a2)
ylim([0 1])

a1 = (a1-a0)./a0;
a2 = (a2-a0)./a0;
figure;boxplot([a1' a2'])
hline(0)

b2_acc_rel_imagined = [a1' a2'];

acc_rel_imagined = [b1_acc_rel_imagined_prop ;b2_acc_rel_imagined];
figure;
boxplot(acc_rel_imagined);
a = get(get(gca,'children'),'children');   % Get the handles of all the objects
t = get(a,'tag');   % List the names of all the objects 
box1 = a(5);   % The 7th object is the first box
set(box1, 'Color', 'k');   % Set the color of the first box to green
box1 = a(6);   % The 7th object is the first box
set(box1, 'Color', 'k');   % Set the color of the first box to green
line1 = a(3);   % The 7th object is the first box
set(line1, 'Color', 'm');   % Set the color of the first box to green
line1 = a(4);   % The 7th object is the first box
set(line1, 'Color', 'm');   % Set the color of the first box to green



% plotting scatter plot with mean and SE
m1 = acc_rel_imagined(:,1);
m1b = sort(bootstrp(1000,@mean,m1));
m2 = acc_rel_imagined(:,2);
m2b = sort(bootstrp(1000,@mean,m2));
y=[mean(acc_rel_imagined)];
s1=std(m1)/sqrt(length(m1));
s2=std(m2)/sqrt(length(m2));
neg = [s1 s2 ];
pos= [s1 s2 ];
neg = [y(1)-m1b(25) y(2)-m2b(25) ];
pos = [m1b(975)-y(1) m2b(975)-y(2) ];
figure;
hold on
% now the scatter part
%B1
hold on
scatter(ones(10,1)+0.07*randn(10,1),b1_acc_rel_imagined_prop(:,1),'or')
scatter(2*ones(10,1)+0.07*randn(10,1),b1_acc_rel_imagined_prop(:,2),'r')
%B2
scatter(ones(4,1)+0.07*randn(4,1),b2_acc_rel_imagined(:,1),'ob')
scatter(2*ones(4,1)+0.07*randn(4,1),b2_acc_rel_imagined(:,2),'ob')

cmap = brewermap(6,'Blues');
x=1:2;
for i=1:length(y)
    errorbar(x(i),y(i),neg(i),pos(i),'Color','k','LineWidth',1)
    plot(x(i),y(i),'o','MarkerSize',10,'Color','k','LineWidth',1,'MarkerFaceColor',[.5 .5 .5])
end
xlim([0.5 2.5])
hline(0,'--k')
xticks([1 2])
xticklabels({'CL1','CL2'})
ylim([-0.2 0.6])
yticks([-.2:.2:.6])
set(gcf,'Color','w')


[h p tb st] = ttest(acc_rel_imagined(:,1),acc_rel_imagined(:,2))

% saving all data 
save acc_relative_imagined_prop_B1B2 -v7.3

% using a mixed effect model
acc_improv=[];
subject=[];
exp_type=[];
%b1
for i=1:size(b1_acc_rel_imagined_prop,2)
    tmp=b1_acc_rel_imagined_prop(:,i);
    acc_improv =[acc_improv;tmp];
    exp_type = [exp_type;i*ones(size(tmp))];
    subject=[subject;1*ones(size(tmp))];
end
%b2
for i=1:size(b2_acc_rel_imagined,2)
    tmp=b2_acc_rel_imagined(:,i);
%     if i==1
%         tmp=tmp([1 3 4]);
%     end
    acc_improv =[acc_improv;tmp];
    exp_type = [exp_type;i*ones(size(tmp))];
    subject=[subject;2*ones(size(tmp))];
end

data=table(acc_improv,exp_type,subject);
glme = fitglme(data,'acc_improv ~ 1+ exp_type + (1|subject)')


%% B3: looking at decoding performance from imagined -> online -> batch (MAIN)
% across days

clc;clear;
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data_B3
addpath 'C:\Users\nikic\Documents\MATLAB'
acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
iterations=5;
plot_true=false;
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);
    %disp([session_data(i).Day '  ' num2str(length(batch_idx))]);

    %%%%%% cross_val classification accuracy for imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    load('ECOG_Grid_8596_000067_B3.mat')
    condn_data = load_data_for_MLP_TrialLevel_B3(files,ecog_grid);

    % get cross-val classification accuracy
    [acc_imagined,train_permutations] = accuracy_imagined_data(condn_data, iterations);
    acc_imagined=squeeze(nanmean(acc_imagined,1));
    if plot_true
        figure;imagesc(acc_imagined)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
    end
    acc_imagined_days(:,i) = diag(acc_imagined);


    %%%%%% get classification accuracy for online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    % get the classification accuracy
    acc_online = accuracy_online_data(files);
    if plot_true
        figure;imagesc(acc_online)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
    end
    acc_online_days(:,i) = diag(acc_online);


    %%%%%% cross_val classification accuracy for batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    % get the classification accuracy
    acc_batch = accuracy_online_data(files);
    if plot_true
        figure;imagesc(acc_batch)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
    end
    acc_batch_days(:,i) = diag(acc_batch);

end

load ('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\b1_acc_rel_imagined_prop.mat')
save hDOF_6days_accuracy_results_New_B2 -v7.3
%save hDOF_10days_accuracy_results -v7.3


%acc_online_days = (acc_online_days + acc_batch_days)/2;
figure;
ylim([0.0 1])
xlim([0.5 4.5])
hold on
plot(nanmean(acc_imagined_days,1))
plot(nanmean(acc_online_days,1),'r')
plot(nanmean(acc_batch_days,1),'k')

% as regression lines
figure;plot(mean(acc_imagined_days,1),'.','MarkerSize',20)

% stats
tmp = [median(acc_imagined_days,1)' median(acc_online_days,1)' ...
    median(acc_batch_days,1)'];

figure;boxplot(acc_imagined_days)
ylim([0.2 1])
xlim([0.5 10.5])
hold on
boxplot(acc_batch_days,'Colors','k')
a = get(get(gca,'children'),'children');

figure;
boxplot([acc_imagined_days(:) acc_online_days(:) acc_batch_days(:)])

m1 = (acc_imagined_days(:));
m1b = sort(bootstrp(1000,@mean,m1));
m11 = mean(acc_imagined_days,1);
m2 = (acc_online_days(:));
m2b = sort(bootstrp(1000,@mean,m2));
m22 = mean(acc_online_days,1);
m3 = (acc_batch_days(:));
m3b = sort(bootstrp(1000,@nanmean,m3));
m33 = nanmean(acc_batch_days,1);
x=1:3;
y=[mean(m1) mean(m2) nanmean(m3)];
s1=std(m1)/sqrt(length(m1));
s2=std(m2)/sqrt(length(m2));
s3=nanstd(m3)/sqrt(sum(~isnan(m3)));
neg = [s1 s2 s3];
pos= [s1 s2 s3];
%neg = [y(1)-m1b(25) y(2)-m2b(25) y(3)-m3b(25)];
%pos = [m1b(975)-y(1) m2b(975)-y(2) m3b(975)-y(3)];
figure;
hold on
cmap = brewermap(6,'Blues');
%cmap = (turbo(7));
for i=1:3
    errorbar(x(i),y(i),neg(i),pos(i),'Color','k','LineWidth',1)
    plot(x(i),y(i),'o','MarkerSize',20,'Color','k','LineWidth',1,'MarkerFaceColor',[.5 .5 .5])
end
for i=1:size(acc_batch_days,2)
    plot(1+0.1*randn(1),m11(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',5,'Color',[cmap(end,:) .5])
    plot(2+0.1*randn(1),m22(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',5,'Color',[cmap(end,:) .5])
    plot(3+0.1*randn(1),m33(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',5,'Color',[cmap(end,:) .5])
end
xlim([.5 3.5])
ylim([0.75 1])
xticks(1:3)
xticklabels({'Imagined','Online','Batch'})
set(gcf,'Color','w')
set(gca,'LineWidth',1)
yticks(0:.1:1)
set(gca,'FontSize',12)
h=hline(.25,'--');
h.LineWidth=1;
xlabel('Decoder Type')
ylabel('Accuracy')

tmp = [ m11' m22' m33'];
figure;boxplot(tmp)

tmp = [ m1 m2 m3];
figure;boxplot(tmp)

% regression lines for mahab distances in latent space
imag = [0.732087
    0.471915
    0.526922
    0.187374
    0.611636
    0.501813
    ];
online=[4.09371
    1.46306
    0.93976
    2.13517
    1.05116
    1.34932
    ];
batch=[1.58127
    2.48264
    3.26464
    3.10718
    ]; % days 12 thru 5

% PLOT REGRESSION LINES
imag = tmp(:,1);
online = tmp(:,2);
batch = tmp(1:4,3);
figure;
hold on
days=1:4;
x=[ones(length(days),1) days'];
%imag
plot(days,imag(2:5),'.','MarkerSize',20)
y=imag(2:5);
[B,BINT,R,RINT,STATS1] = regress(y,x);
yhat=x*B;
plot(days,yhat,'b','LineWidth',1)
%online
plot(days,online(2:5),'.k','MarkerSize',20)
y=online(2:5);
[B,BINT,R,RINT,STATS2] = regress(y,x);
yhat=x*B;
plot(days,yhat,'k','LineWidth',1)
%batch
plot(days,batch,'.r','MarkerSize',20)
y=batch;
[B,BINT,R,RINT,STATS3] = regress(y,x);
yhat=x*B;
plot(days,yhat,'r','LineWidth',1)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlim([.5 4.5])
xticks(1:4)
xticklabels(1:4)
ylim([0 3.5])
yticks([0:3])


% PLOT REGRESSION LINES equal size in tmp
imag = tmp(:,1);
online = tmp(:,2);
batch = tmp(:,3);
figure;
hold on
days=1:4;
x=[ones(length(days),1) days'];
%imag
plot(days,imag,'.','MarkerSize',20)
y=imag;
[B,BINT,R,RINT,STATS1] = regress(y,x);
yhat=x*B;
plot(days,yhat,'b','LineWidth',1)
%online
plot(days,online,'.k','MarkerSize',20)
y=online;
[B,BINT,R,RINT,STATS2] = regress(y,x);
yhat=x*B;
plot(days,yhat,'k','LineWidth',1)
%batch
plot(days,batch,'.r','MarkerSize',20)
y=batch;
[B,BINT,R,RINT,STATS3] = regress(y,x);
yhat=x*B;
plot(days,yhat,'r','LineWidth',1)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlim([.5 4.5])
xticks(1:4)
xticklabels(1:4)



% using robust regression in matlab equal size
figure;
hold on
% imag
plot(days,imag,'.b','MarkerSize',20)
y=imag;
lm=fitlm(x(:,2:end),y,'Robust','on');
B=lm.Coefficients.Estimate;
yhat = x*B;
plot(days,yhat,'b','LineWidth',1)
% online
plot(days,online,'.k','MarkerSize',20)
y=online;
lm=fitlm(x(:,2:end),y,'Robust','on');
B=lm.Coefficients.Estimate;
yhat = x*B;
plot(days,yhat,'k','LineWidth',1)
% batch
plot(days,batch,'.r','MarkerSize',20)
y=batch;
lm=fitlm(x(:,2:end),y,'Robust','on');
B=lm.Coefficients.Estimate;
yhat=x*B;
plot(days,yhat,'r','LineWidth',1)
% beautify
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xticks([1:4])



% using robust regression in matlab
figure;
hold on
% imag
plot(days,imag(2:5),'.b','MarkerSize',20)
y=imag(2:5);
lm=fitlm(x(:,2:end),y,'Robust','on');
B=lm.Coefficients.Estimate;
yhat = x*B;
plot(days,yhat,'b','LineWidth',1)
% online
plot(days,online(2:5),'.k','MarkerSize',20)
y=online(2:5);
lm=fitlm(x(:,2:end),y,'Robust','on');
B=lm.Coefficients.Estimate;
yhat = x*B;
plot(days,yhat,'k','LineWidth',1)
% batch
plot(days,batch,'.r','MarkerSize',20)
y=batch;
lm=fitlm(x(:,2:end),y,'Robust','on');
B=lm.Coefficients.Estimate;
yhat=x*B;
plot(days,yhat,'r','LineWidth',1)
% beautify
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xticks([1:4])
yticks([5:5:35])
ylim([0 1])





% stats on the latent space variance
tmp=[0.000937406	0.0985546	0.0289443
    0.00154575	0.0315809	0.0753841
    0.00141096	0.0103554	0.20185
    0.00043903	0.0643351	0.150029
    0.00125143	0.00337939	0.112707
    0.00113499	0.00377055	0.112707
    ];
[P,ANOVATAB,STATS] = anova1(tmp);

[h p tb st] = ttest(tmp(:,1),tmp(:,2));p
[h p tb st] = ttest(tmp(:,1),tmp(:,3));p
[h p tb st] = ttest(tmp(:,2),tmp(:,3));p

% how about bootstrapped difference of means?
stat  =  mean(tmp(:,2)) - mean(tmp(:,1));
c= [tmp(:,2);tmp(:,1)];
l=size(tmp,1);
boot=[];
c=c-mean(c);
for i=1:5000
    idx=randperm(numel(c));
    a1 = c(idx(1:l));
    b1 = c(idx(l+1:end));
    boot(i) = mean(a1)-mean(b1);
end
figure;hist((boot))
vline((stat),'r')
sum(boot>stat)/length(boot)


% get the accuracies relative to imagined movement within that day
a0 = mean(acc_imagined_days(:,2:5),1);
a1 = mean(acc_online_days(:,2:5),1);
a2 = mean(acc_batch_days(:,2:5),1);
figure;
plot(a0);
hold on
plot(a1);
plot(a2)
ylim([0 1])

a1 = (a1-a0)./a0;
a2 = (a2-a0)./a0;
figure;boxplot([a1' a2'])
hline(0)

b2_acc_rel_imagined = [a1' a2'];

acc_rel_imagined = [b1_acc_rel_imagined_prop ;b2_acc_rel_imagined];
figure;
boxplot(acc_rel_imagined);
a = get(get(gca,'children'),'children');   % Get the handles of all the objects
t = get(a,'tag');   % List the names of all the objects 
box1 = a(5);   % The 7th object is the first box
set(box1, 'Color', 'k');   % Set the color of the first box to green
box1 = a(6);   % The 7th object is the first box
set(box1, 'Color', 'k');   % Set the color of the first box to green
line1 = a(3);   % The 7th object is the first box
set(line1, 'Color', 'm');   % Set the color of the first box to green
line1 = a(4);   % The 7th object is the first box
set(line1, 'Color', 'm');   % Set the color of the first box to green



% plotting scatter plot with mean and SE
m1 = acc_rel_imagined(:,1);
m1b = sort(bootstrp(1000,@mean,m1));
m2 = acc_rel_imagined(:,2);
m2b = sort(bootstrp(1000,@mean,m2));
y=[mean(acc_rel_imagined)];
s1=std(m1)/sqrt(length(m1));
s2=std(m2)/sqrt(length(m2));
neg = [s1 s2 ];
pos= [s1 s2 ];
neg = [y(1)-m1b(25) y(2)-m2b(25) ];
pos = [m1b(975)-y(1) m2b(975)-y(2) ];
figure;
hold on
% now the scatter part
%B1
hold on
scatter(ones(10,1)+0.07*randn(10,1),b1_acc_rel_imagined_prop(:,1),'or')
scatter(2*ones(10,1)+0.07*randn(10,1),b1_acc_rel_imagined_prop(:,2),'r')
%B2
scatter(ones(4,1)+0.07*randn(4,1),b2_acc_rel_imagined(:,1),'ob')
scatter(2*ones(4,1)+0.07*randn(4,1),b2_acc_rel_imagined(:,2),'ob')

cmap = brewermap(6,'Blues');
x=1:2;
for i=1:length(y)
    errorbar(x(i),y(i),neg(i),pos(i),'Color','k','LineWidth',1)
    plot(x(i),y(i),'o','MarkerSize',10,'Color','k','LineWidth',1,'MarkerFaceColor',[.5 .5 .5])
end
xlim([0.5 2.5])
hline(0,'--k')
xticks([1 2])
xticklabels({'CL1','CL2'})
ylim([-0.2 0.6])
yticks([-.2:.2:.6])
set(gcf,'Color','w')


[h p tb st] = ttest(acc_rel_imagined(:,1),acc_rel_imagined(:,2))

% saving all data 
save acc_relative_imagined_prop_B1B2 -v7.3

% using a mixed effect model
acc_improv=[];
subject=[];
exp_type=[];
%b1
for i=1:size(b1_acc_rel_imagined_prop,2)
    tmp=b1_acc_rel_imagined_prop(:,i);
    acc_improv =[acc_improv;tmp];
    exp_type = [exp_type;i*ones(size(tmp))];
    subject=[subject;1*ones(size(tmp))];
end
%b2
for i=1:size(b2_acc_rel_imagined,2)
    tmp=b2_acc_rel_imagined(:,i);
%     if i==1
%         tmp=tmp([1 3 4]);
%     end
    acc_improv =[acc_improv;tmp];
    exp_type = [exp_type;i*ones(size(tmp))];
    subject=[subject;2*ones(size(tmp))];
end

data=table(acc_improv,exp_type,subject);
glme = fitglme(data,'acc_improv ~ 1+ exp_type + (1|subject)')


%% B3 PLASTICITY AND AE FRAMEWORK (MAIN MAIN)
% good days: 20210324

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
cd(root_path)
load session_data_B3
dist_online_total=[];
dist_imag_total=[];
var_imag_total=[];
mean_imag_total=[];
var_online_total=[];
mean_online_total=[];
res=[];
mahab_full_online=[];
mahab_full_imagined=[];
mahab_full_batch=[];
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);

    %%%%%%imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    load('ECOG_Grid_8596_000067_B3.mat')
    condn_data = load_data_for_MLP_B3(files,ecog_grid);
    


    %save the data
    filename = ['B3_condn_data_Imagined_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    % build the AE based on MLP and only for hG
    %[net,Xtrain,Ytrain] = build_mlp_AE_B2(condn_data);
    %[net,Xtrain,Ytrain] = build_mlp_AE_supervised(condn_data);

    % get the mahab distance in the full dataset
    Dimagined = mahal2_full(condn_data);
    Dimagined = triu(Dimagined);
    Dimagined = Dimagined(Dimagined>0);
    mahab_full_imagined = [mahab_full_imagined Dimagined];

    % get activations in deepest layer but averaged over a trial
    %     imag=1;
    %     [TrialZ_imag,dist_imagined,mean_imagined,var_imagined,idx_imag] = ...
    %         get_latent_regression_B2(files,net,imag,ecog_grid);
    %     dist_imag_total = [dist_imag_total;dist_imagined];
    %     mean_imag_total=[mean_imag_total;pdist(mean_imagined)];
    %     var_imag_total=[var_imag_total;var_imagined'];

    %%%%%%online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    condn_data = load_data_for_MLP_B3(files,ecog_grid);

    % save the data
    filename = ['B3_condn_data_Online_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    % get the mahab distance in the full dataset
    Donline = mahal2_full(condn_data);
    Donline = triu(Donline);
    Donline = Donline(Donline>0);
    mahab_full_online = [mahab_full_online Donline];


    % get activations in deepest layer
    %     imag=0;
    %     [TrialZ_online,dist_online,mean_online,var_online,idx_online] = ...
    %         get_latent_regression_B2(files,net,imag,ecog_grid);
    %     dist_online_total = [dist_online_total;dist_online];
    %     mean_online_total=[mean_online_total;pdist(mean_online)];
    %     var_online_total=[var_online_total;var_online'];

    %%%%%%batch data

    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end


    %load the data
    condn_data = load_data_for_MLP_B3(files,ecog_grid);

    % save the data
    filename = ['B3_condn_data_Batch_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    % get the mahab distance in the full dataset
    Donline = mahal2_full(condn_data);
    Donline = triu(Donline);
    Donline = Donline(Donline>0);
    mahab_full_batch = [mahab_full_batch Donline];

    % get activations in deepest layer
    %     imag=0;
    %     [TrialZ_batch,dist_batch,mean_batch,var_batch,idx_batch] = get_latent_regression(files,net,imag);
    %     dist_batch_total = [dist_batch_total;dist_batch];
    %     mean_batch_total=[mean_batch_total;pdist(mean_batch)];
    %     var_batch_total=[var_batch_total;var_batch'];





    % plotting imagined and online in latent space
    %     idxa = find(idx_imag==4);
    %     idxb = find(idx_online==4);
    %     idxa = idxa(randperm(length(idxa),length(idxb)));
    %     figure;hold on
    %     plot3(TrialZ_imag(1,idxa),TrialZ_imag(2,idxa),TrialZ_imag(3,idxa),'.','MarkerSize',20)
    %     plot3(TrialZ_online(1,idxb),TrialZ_online(2,idxb),TrialZ_online(3,idxb),'.','MarkerSize',20)
    %     c1 = TrialZ_imag(:,idxa);
    %     c2 = TrialZ_online(:,idxb);
    %     c1=cov(c1');
    %     c2=cov(c2');

    %      plot
    %
    %     figure;boxplot([dist_imagined' dist_online'])
    %     box off
    %     set(gcf,'Color','w')
    %     xticks(1:2)
    %     xticklabels({'Imagined Data','Online Data'})
    %     ylabel('Distance')
    %     title('Inter-class distances')
    %     set(gca,'LineWidth',1)
    %     set(gca,'FontSize',12)

    %     [h p tb st]=ttest(dist_imagined,dist_online);
    %     disp([p mean([dist_imagined' dist_online'])]);
    %     res=[res;[p mean([dist_imagined' dist_online'])]];
end

save mahab_dist_full_B3 -v7.3

figure;boxplot([mahab_full_imagined mahab_full_online mahab_full_batch])

figure;plot(1:4,median(mahab_full_imagined));
hold on
plot(1:4,median(mahab_full_online),'r')
plot(1:4,median(mahab_full_batch),'k')

figure;
plot(mean(dist_online_total'))
set(gcf,'Color','w')
title('Across Day Learning')
ylabel('Mahalanobis Dist.')
xlabel('Day')
xlim([0.5 11.5])
tmp = mean(dist_online_total');
figure;
tmp1 = tmp(1:4);
tmp2 = tmp(5:end);
tmp1(end+1:length(tmp2))=NaN;
boxplot([tmp1' tmp2'])
xticklabels({'Early Days','Late Days'})
ylabel('Mahalanobis Dist')
title('Online Bins proj. thru Imagined Manifold')
set(gcf,'Color','w')



%% (MAIN) Looking at the regression of projecting across day data on manifold

% get the stats from python
clc;clear

%B2
y=[0.06356695, 0.07478359, 0.09559412, 0.12340624, 0.13129963]';
x=[ones(length(y),1) [1:length(y)]'];

[B,BINT,R,RINT,STATS1] = regress(y,x);
lm = fitlm(x(:,2),y);

%B1
y=[5.95474288, 6.8286061 , 8.39052186, 8.17585576, 8.27175004,...
       9.24597975, 8.60957232, 9.03721714, 8.79579779]';
x=[ones(length(y),1) [1:length(y)]'];

[B,BINT,R,RINT,STATS2] = regress(y,x);
STATS2
lm = fitlm(x(:,2),y)

% for lm get the data from python
mahab=data(:,1);
days = data(:,2);
data=table(days,mahab);

glm = fitglme(data,'mahab ~ 1+ days');

save latent_across_day_proj_B1 -v7.3


%% ANALYSIS OF THE 3D PATH TASK AND ITS PERFORMANCE ETC.

% get all the trials that had the 3D path task with some measure of task
% performance

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath 'C:\Users\nikic\Documents\MATLAB'

task_name = 'Robot3DPath';
folders = dir(root_path);

field1 = 'name'; value1 = '';
field2 = 'folder'; value2 = '';
field3 = 'date'; value3='';
field4 = 'bytes'; value4 = [];
field5 = 'isdir';value5 =[];
field6 = 'datenum';value6=[];

folders1=struct(field1,value1,field2,value2,field3,value3,field4,value4,field5,value5,field6,value6);
k=1;
for i=1:length(folders)
    if ~isempty(str2num(folders(i).name))
        folders1(k) = folders(i);
        k=k+1;
    end
end

folders=folders1;

files=[];
for i=1:length(folders)
    folder_path = fullfile(root_path,folders(i).name,task_name);
    if exist(folder_path)
        tmp_files=findfiles('.mat',folder_path,1)';
        for j=1:length(tmp_files)
            if length(regexp(tmp_files{j},'Data'))>0
                files=[files;tmp_files(j)];
            end
        end
    end
end

figure;
plot3(TrialData.CursorState(1,:),TrialData.CursorState(2,:),TrialData.CursorState(3,:),'.')

%% plotting robot trajectories for center out (MAIN)
 
clc;clear
foldername = '20210326';
task_name = 'Robot';
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath 'C:\Users\nikic\Documents\MATLAB'
addpath(genpath('C:\Users\nikic\Documents\MATLAB\svg_plot'))

fullpath = fullfile(root_path,foldername,task_name);
files = findfiles('.mat',fullpath,1);
files1=[];
for i=1:length(files)
    if length(regexp(files{i},'Data'))>0
        files1=[files1;files(i)'];
    end
end
files=files1;

% get all the good robot 3D trial data and plot them
col = {'r','g','b','c','m','y'};
col=turbo(6);
files_suc=[];
figure;
hold on
xlim([-250,250])
ylim([-250,250])
zlim([-250,250])
recon_error = [];% wrt to how well straight line trajecotry is reconstructed
trial_error = [];% just the plain error with respect to target location
for i=1:length(files)
    load(files{i})
    if TrialData.TargetID == TrialData.SelectedTargetID
        kin = TrialData.CursorState;
        task_state = TrialData.TaskState;
        kinidx = find(task_state==3);
        kin = kin(:,kinidx);
        target = TrialData.TargetPosition;
        targetID = TrialData.TargetID;
        fs = TrialData.Params.UpdateRate;
        if size(kin,2)*(1/fs) < 12
            files_suc = [files_suc;files(i)];
            %plot3(kin(1,:),kin(2,:),kin(3,:),'LineWidth',2,'color',col{targetID});
            plot3(kin(1,:),kin(2,:),kin(3,:),'LineWidth',2,'color',col(targetID,:));
        end

        % get the errors in terms of deviation from the ideal path
        idx = find(target==0); % get the axes where errors shoudln't happen
        idx_target = find(target~=0);
        tmp_error = [];
        kin = kin(1:3,:);
        for j=1:size(kin,2)
            if sum(kin(:,j)) ~= 0
                break
            end
        end
        kin = kin(:,j:end);
        for j=1:size(kin,2)
            if (sign(target(idx_target)) * sign(kin(idx_target,j))) == -1
                f=2;
            else
                f=1;
            end
            e = f*(sum((target(idx)' - kin(idx,j)).^2));
            tmp_error = [tmp_error;e];
        end
        recon_error =[recon_error; sqrt(sum(tmp_error(1:14)))];
        trial_error = [trial_error sqrt(sum(sum((kin(:,1:10) - target').^2)))];
        %end
    end
end

set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('X-axis')
ylabel('Y-axis')
zlabel('Z-axis')
set(gca,'LineWidth',1.0)
grid on

save recon_error_IBID recon_error -v7.3

%plot2svg('3DTraj-grid.svg');


%% plotting diagonal robot trajectories (MAIN)



clc;clear
foldername = '20210115';
task_name = 'Robot';
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath 'C:\Users\nikic\Documents\MATLAB'
addpath(genpath('C:\Users\nikic\Documents\MATLAB\svg_plot'))

files={'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210115\Robot\114418\BCI_Fixed\Data0002.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210115/Robot\114613\BCI_Fixed\Data0001.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210115\Robot\114613\BCI_Fixed\Data0002.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210115\Robot\114613\BCI_Fixed\Data0004.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210115\Robot\114740\BCI_Fixed\Data0001.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0001.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0002.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0003.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0005.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0006.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0007.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0008.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0009.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0010.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0011.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0012.mat'};

% get all the good robot 3D trial data and plot them
col = {'r','g','b','c','m','y'};
col=turbo(4);
files_suc=[];
figure;
hold on
xlim([-250,250])
ylim([-250,250])
zlim([-250,250])
tid=[];
% also get the velocities in the x and y directions
vel=[];
for i=1:length(files)
    load(files{i})
    tid = [tid TrialData.TargetID];
    if TrialData.TargetID == TrialData.SelectedTargetID        
        kin = TrialData.CursorState;
        task_state = TrialData.TaskState;
        kinidx = find(task_state==3);
        kin = kin(:,kinidx);        
        target = TrialData.TargetPosition;
        targetID = TrialData.TargetID-6;
        fs = TrialData.Params.UpdateRate;
        if size(kin,2)*(1/fs) < 12
            files_suc = [files_suc;files(i)];
            %plot3(kin(1,:),kin(2,:),kin(3,:),'LineWidth',2,'color',col{targetID});
            plot3(kin(1,:),kin(2,:),kin(3,:),'LineWidth',2,'color',col(targetID,:));
            vel = [vel kin(4:6,:)];
        end
    end
end

%grid off 
set(gcf,'Color','w')
set(gca,'FontSize',12)
plot2svg('3DTraj_diag1.svg');

% pca on the velocity data
[c,s,l] = pca(vel');
figure;
stem(cumsum(l)./sum(l))
figure;stem(c(:,1))
xlim([0 4])
figure;stem(c(:,2))
xlim([0 4])
figure;stem(c(:,3))
xlim([0 4])
figure;



% pca on null velocity data, 2dim
vel_null=[];
for i=1:100
    if rand(1)>0.5
        tmp = [randn(1,10)*20;zeros(1,10)];
    else
        tmp = [zeros(1,10);randn(1,10)*20];
    end
    vel_null = [vel_null tmp];
end
[c,s,l] = pca(vel_null');
figure;stem(c(:,1),'k')
xlim([0.5 2.5])
xticks([1:2])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
box off
figure;stem(c(:,2),'k')
xlim([0.5 2.5])
xticks([1:2])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
box off

% pca on the first x and y velocity alone 
C = cov((vel(1:2,:))');
figure;imagesc(C)
[c,s,l] = pca(vel(1:2,:)');
figure;stem(c(:,1),'k')
xlim([0.5 2.5])
xticks([1:2])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
box off
figure;stem(c(:,2),'k')
xlim([0.5 2.5])
xticks([1:2])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
box off
% nullhypothesis via MP test
kindata1=vel(1:2,:)';
kindata1=kindata1-mean(kindata1);
q = size(kindata1,1)/size(kindata1,2);
q = sqrt(q);
sigma2 = var(kindata1(:)-mean(kindata1(:)));
lambda_thresh = sigma2*((1+1/q)^2);
figure;stem(l);
hline(lambda_thresh)




% get histogram of angles relative to the x and y axes
velxy = vel(1:2,:);
angles = [];
for i=1:size(velxy,2)    
    velxy(:,i) = velxy(:,i)./norm(velxy(:,i));
    tmp = abs(velxy(:,i));
    angles(i) = atan(tmp(2)/tmp(1));
end
errx = acos([1 0]*abs(velxy));
erry = acos([0 1]*abs(velxy));
err = [errx erry];
err = err(~isnan(err));
angles = angles(~isnan(angles));
%plot
[t,r]=rose(angles,20);
figure
polarplot(t,r,'LineWidth',1,'Color','k');
pax=gca;
%pax.RLim = [0 20];
thetaticks(0:30:360);
pax.ThetaAxisUnits='radians';
pax.FontSize=16;
set(gcf,'Color','w')
%pax.RTick = [5 10 15 20 ];
pax.GridAlpha = 0.25;
pax.MinorGridAlpha = 0.25;
pax.ThetaMinorGrid = 'off';
%pax.ThetaTickLabel = {'0', ' ', '\pi/2 ', ' ','\pi',' ','3\pi/2',' '};
pax.ThetaTickLabel = ''
%pax.ThetaTickLabel = {'0', ' ', ' ', ' ','\pi',' ',' ',' '};
pax.RTickLabel = {' ',' '};
pax.RAxisLocation=1;
pax.RAxis.LineWidth=1;
pax.ThetaAxis.LineWidth=1;
pax.LineWidth=1;
pax.ThetaLim = [0 pi/2];
temp = exp(1i*angles);
r1 = abs(mean(temp))*1 * max(r);
phi = angle(mean(temp));
hold on;
polarplot([phi-0.01 phi],[0 r1],'LineWidth',1.5,'Color','r')
polarplot([0 0],[0 0.25e3],'LineWidth',1.5,'Color','k')
polarplot([pi/2 pi/2 ],[0 0.25e3],'LineWidth',1.5,'Color','k')
set(gcf,'PaperPositionMode','auto')
set(gcf,'Position',[680.0,865,120.0,113.0])

% null model of what distributions should be like
err_null = [zeros(1,400) 90*pi/180*ones(1,400)];
[t,r]=rose(err_null,50);
figure
polarplot(t,r,'LineWidth',1,'Color','k');
pax=gca;
%pax.RLim = [0 20];
thetaticks(0:30:360);
pax.ThetaAxisUnits='radians';
pax.FontSize=16;
set(gcf,'Color','w')
%pax.RTick = [5 10 15 20 ];
pax.GridAlpha = 0.25;
pax.MinorGridAlpha = 0.25;
pax.ThetaMinorGrid = 'off';
%pax.ThetaTickLabel = {'0', ' ', '\pi/2 ', ' ','\pi',' ','3\pi/2',' '};
pax.ThetaTickLabel = ''
%pax.ThetaTickLabel = {'0', ' ', ' ', ' ','\pi',' ',' ',' '};
pax.RTickLabel = {' ',' '};
pax.RAxisLocation=1;
pax.RAxis.LineWidth=1;
pax.ThetaAxis.LineWidth=1;
pax.LineWidth=1;
pax.ThetaLim = [0 90*pi/180];
hold on;
polarplot([0 0],[0 0.4e3],'LineWidth',1.5,'Color','k')
polarplot([pi/2 pi/2 ],[0 0.4e3],'LineWidth',1.5,'Color','k')
set(gcf,'PaperPositionMode','auto')
set(gcf,'Position',[680.0,865,120.0,113.0])


% now plotting a few example trials along with position, user input and
% velocity profile

idx=7;
load(files{idx})
kin = TrialData.CursorState;
task_state = TrialData.TaskState;
kinidx = find(task_state==3);
kin = kin(:,kinidx);
decodes = TrialData.FilteredClickerState;
fs=TrialData.Params.UpdateRate;
tt = [0:length(decodes)-1]*(1/fs);
col = turbo(8);

% plot the trajectory 
figure
hold on
for i=1:length(decodes)
    c = col(decodes(i)+1,:);
    plot3(kin(1,i),-kin(2,i),kin(3,i),'.','MarkerSize',40,'Color',c);
end
xlim([-300 300])
ylim([-300 300])
zlim([-300 300])
target = TrialData.TargetPosition;
plot3(target(1),-target(2),target(3),'ok','MarkerSize',50)
%plot(-150,150,'o','MarkerSize',50,'Color','k')
set(gcf,'Color','w')
xticks([-200:200:200])
yticks([-200:200:200])
zticks([-200:200:200])

% plot as a straight line
figure;
hold on
plot3(kin(1,:),-kin(2,:),kin(3,:),'LineWidth',2,'Color','b');
xlim([-300 300])
ylim([-300 300])
zlim([-300 300])
target = TrialData.TargetPosition;
plot3(target(1),-target(2),target(3),'ok','MarkerSize',50)
%plot(-150,150,'o','MarkerSize',50,'Color','k')
set(gcf,'Color','w')
xticks([-200:200:200])
yticks([-200:200:200])
zticks([-200:200:200])
xlabel('X axis')
ylabel('Y axis')
zlabel('Z axis')
set(gcf,'Color','w')
set(gca,'FontSize',14)
%view(40,40)



% plot the decodes
figure;
set(gcf,'Color','w')
subplot(2,1,1)
hold on
for i=0:7
    h=barh(i,length(decodes),1);
    h.FaceColor = col(i+1,:);
    h.FaceAlpha = 0.8;
end
stem(decodes,'filled','LineWidth',1,'Color','k')
ii = [9 17 25 33 41 49];
xticks(ii)
xticklabels(tt(ii))
yticks ''
axis tight
% plot the velocity profile 
subplot(2,1,2)
hold on
plot(kin(4,:),'r','LineWidth',1)
plot(kin(5,:),'b','LineWidth',1)
plot(kin(6,:),'g','LineWidth',1)
xticks(ii)
xticklabels(tt(ii))
axis tight

%%%%% plotting the dynamics and decodes for the virtual r2g lateral task
filename = fullfile(root_path,'20211013\RobotLateralR2G\135901\BCI_Fixed\Data0001.mat');
load(filename)
grip_mode = TrialData.OpMode;
decodes = TrialData.FilteredClickerState;
translation_decodes = decodes(grip_mode==0);
gripper_decodes = decodes(grip_mode==1);

% plot the decodes
fs=TrialData.Params.UpdateRate;
tt = [0:length(decodes)-1]*(1/fs);
col = turbo(8);
figure;
set(gcf,'Color','w')
hold on
for i=0:7
    h=barh(i,length(decodes),1);
    h.FaceColor = col(i+1,:);
    h.FaceAlpha = 0.5;
end
stem(decodes,'filled','LineWidth',1,'Color','k')
ii = [26 51 76 101 126 151 176 201 226 ];
xticks(ii)
xticklabels(tt(ii))
yticks ''
axis tight

%% (MAIN) B1 ROBOT PATH TASK STATS
% get the path efficiency i.e., distance from ideal path as compared to
% random walks of same length

clc;clear;close all




%% (MAIN) B3 looking at decoding performance from imagined -> online -> batch
% across days

clc;clear;
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
load session_data
addpath 'C:\Users\nikic\Documents\MATLAB'
session_data = session_data([1:9 11]); % removing bad days
acc_imagined_days=[];
acc_online_days=[];
acc_batch_days=[];
iterations=50;
plot_true=false;
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    if i~=6
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_imag(folders_am==0)=0;
        folders_online(folders_am==0)=0;
    end

    if i==3 || i==6 || i==8
        folders_pm = strcmp(session_data(i).AM_PM,'pm');
        folders_batch(folders_pm==0)=0;
        if i==8
            idx = find(folders_batch==1);
            folders_batch(idx(3:end))=0;
        end
    else
        folders_am = strcmp(session_data(i).AM_PM,'am');
        folders_batch(folders_am==0) = 0;
    end

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);
    %disp([session_data(i).Day '  ' num2str(length(batch_idx))]);

    %%%%%% cross_val classification accuracy for imagined data
    folders = session_data(i).folders(imag_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    condn_data = load_data_for_MLP_TrialLevel(files);
    % save the data
    filename = ['condn_data_ImaginedTrials_Day' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    % get cross-val classification accuracy
    [acc_imagined,train_permutations] = accuracy_imagined_data(condn_data, iterations);
    acc_imagined=squeeze(nanmean(acc_imagined,1));
    if plot_true
        figure;imagesc(acc_imagined)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
    end
    acc_imagined_days(:,i) = diag(acc_imagined);


    %%%%%% get classification accuracy for online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    % get the classification accuracy
    acc_online = accuracy_online_data(files);
    if plot_true
        figure;imagesc(acc_online)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
    end
    acc_online_days(:,i) = diag(acc_online);


    %%%%%% cross_val classification accuracy for batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    % get the classification accuracy
    acc_batch = accuracy_online_data(files);
    if plot_true
        figure;imagesc(acc_batch)
        colormap bone
        clim([0 1])
        set(gcf,'color','w')
    end
    acc_batch_days(:,i) = diag(acc_batch);
end

save hDOF_10days_accuracy_results_New -v7.3
%save hDOF_10days_accuracy_results -v7.3


%acc_online_days = (acc_online_days + acc_batch_days)/2;
figure;
ylim([0.2 1])
xlim([0.5 10.5])
hold on
plot(mean(acc_imagined_days,1))
plot(median(acc_online_days,1))
plot(median(acc_batch_days,1),'k')

% linear model for time to see if improvement in decoding accuracy
days=1:10;
y=mean(acc_imagined_days,1)';
figure;hold on
plot(days,y,'.k','MarkerSize',20)
x = [ones(length(days),1) days'];
[B,BINT,R,RINT,STATS] = regress(y,x);
yhat = x*B;
plot(days,yhat,'k','LineWidth',1)
% [bhat p wh se ci t_stat]=robust_fit((1:length(tmp))',tmp',1);
% yhat1 = x*bhat;
% plot(days,yhat1,'k','LineWidth',1)
xlim([.5 10.5])
xticks([1:10])
set(gcf,'Color','w')
yticks(0:.2:1)
ylim([0 1])
STATS(3)
lm = fitlm(x(:,2),y)


% as regression lines
figure;plot(mean(acc_imagined_days,1),'.','MarkerSize',20)

% stats
tmp = [median(acc_imagined_days,1)' median(acc_online_days,1)' ...
    median(acc_batch_days,1)'];

figure;boxplot(acc_imagined_days)
ylim([0.2 1])
xlim([0.5 10.5])
hold on
boxplot(acc_batch_days,'Colors','k')
a = get(get(gca,'children'),'children');

figure;
boxplot([acc_imagined_days(:) acc_online_days(:) acc_batch_days(:)])

m1 = (acc_imagined_days(:));
m1b = sort(bootstrp(1000,@mean,m1));
m11 = mean(acc_imagined_days,1);
m2 = (acc_online_days(:));
m2b = sort(bootstrp(1000,@mean,m2));
m22 = mean(acc_online_days,1);
m3 = (acc_batch_days(:));
m3b = sort(bootstrp(1000,@mean,m3));
m33 = mean(acc_batch_days,1);
x=1:3;
y=[mean(m1) mean(m2) mean(m3)];
neg = [y(1)-m1b(25) y(2)-m2b(25) y(3)-m3b(25)];
pos = [m1b(975)-y(1) m2b(975)-y(2) m3b(975)-y(3)];
figure;
hold on
cmap = brewermap(10,'Blues');
%cmap = (turbo(7));
for i=1:10
    plot(1+0.1*randn(1),m11(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',3,'Color',[cmap(end,:) .5])
    plot(2+0.1*randn(1),m22(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',3,'Color',[cmap(end,:) .5])
    plot(3+0.1*randn(1),m33(i),'o','MarkerFaceColor',cmap(end,:),'MarkerSize',3,'Color',[cmap(end,:) .5])
end
for i=1:3
    errorbar(x(i),y(i),neg(i),pos(i),'Color','k','LineWidth',1)
    plot(x(i),y(i),'o','MarkerSize',20,'Color','k','LineWidth',1,'MarkerFaceColor',[.5 .5 .5])
end
xlim([.5 3.5])
ylim([0.5 1])
xticks(1:3)
xticklabels({'Imagined','Online','Batch'})
set(gcf,'Color','w')
set(gca,'LineWidth',1)
yticks(0:.1:1)
set(gca,'FontSize',12)

tmp = [ m11' m22' m33'];
figure;boxplot(tmp)

% fit a general linear regression model across days 
acc=[];
days_acc=[];
experiment=[];
tmp = mean(acc_imagined_days,1);
acc = [acc;tmp'];
experiment =[experiment;ones(length(tmp),1)];
tmp1 = mean(acc_online_days,1);
acc = [acc;tmp1'];
experiment =[experiment;2*ones(length(tmp),1)];
tmp2 = mean(acc_batch_days,1);
acc = [acc;tmp2'];
experiment =[experiment;3*ones(length(tmp),1)];

data = table(experiment,acc);
glm = fitglm(data,'acc ~ 1 + experiment');

% stats
[h p tb st] = ttest(tmp,tmp1)
[h p tb st] = ttest(tmp,tmp2)
[h p tb st] = ttest(tmp1,tmp2)

% using a GLM for each action across days 
acc=[];
days_acc=[];
experiment=[];
for i=1:size(acc_imagined_days,2)
    a = acc_imagined_days(:,i);
    acc = [acc;a];
    days_acc = [days_acc;i*ones(size(a))];
    experiment = [experiment;1*ones(size(a))];
end
for i=1:size(acc_online_days,2)
    a = acc_online_days(:,i);
    acc = [acc;a];
    days_acc = [days_acc;i*ones(size(a))];
    experiment = [experiment;2*ones(size(a))];
end
for i=1:size(acc_batch_days,2)
    a = acc_batch_days(:,i);
    acc = [acc;a];
    days_acc = [days_acc;i*ones(size(a))];
    experiment = [experiment;3*ones(size(a))];
end

data = table(days_acc,experiment,acc);
glme = fitglme(data,'acc ~ 1 + experiment +(1|days_acc)')

%test of medians between cl1 and cl2
a0=data.acc(data.experiment==1);
a1=data.acc(data.experiment==2);
a2=data.acc(data.experiment==3);

a1 = (median(acc_online_days,1))';
a2 = (median(acc_batch_days,1))';
stat = median(a2)-median(a1);
boot=[];
a=[a1;a2];
for i=1:1000
    idx = randperm(length(a));
    a11 = a(idx(1:10));
    a22 = a(idx(11:end));
    boot(i) = median(a11) - median(a22);
end
figure;hist((boot))
vline(stat)
sum((boot) > stat)/length(boot)

% X = [ones(10,1) (1:10)'];
% Y =  mean(acc_batch_days,1)';
% [B,BINT,R,RINT,STATS] = regress(Y,X)

figure;
boxplot([a0 a1 a2])

% get the accuracies relative to imagined movement within that day
a0 = mean(acc_imagined_days,1);
a1 = mean(acc_online_days,1);
a2 = mean(acc_batch_days,1);
figure;
plot(a0);
hold on
plot(a1);
plot(a2)
ylim([0 1])

a1 = (a1-a0)./a0;
a2 = (a2-a0)./a0;
figure;boxplot([a1' a2'])
hline(0)

b1_acc_rel_imagined_prop = [a1' a2'];
save b1_acc_rel_imagined_prop b1_acc_rel_imagined_prop


%% PLOTTING STATS FOR REAL ROBOT R2G PERFORMANCE

clc;clear







