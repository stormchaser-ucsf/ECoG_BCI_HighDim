

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


%% analzying session specific data



%% using an MLP-AE to look at differences between imagined and online control
% projecting to the latent space here works best for 4 of the 7 actions and
% not all simultaneously due to noise 

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';

% Imagined movement data
%folders={'110604','111123','111649'};
folders={'134638','135318','135829'};
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
%folders = {'113524','113909','114318','114537'};
%folders = {'140842','141045'};
folders={'135435','135630','135830'};%20210623
%folders={'113645','114239'};
day_date = '20210623';
files=[];
for i=1:length(folders)
    folderpath = fullfile(root_path, day_date,'Robot3DArrow',folders{i},'BCI_Fixed');
    files = [files;findfiles('',folderpath)'];
end

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

% get softmax activations from the latent space
TrialZ=[];
idx=[];
imag=0;
batch=1;
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
        X = X(1:96,:);
        Z = activations(net,X','autoencoder');
        % pass it next through softmax layer
        if batch==0            
            Z = activations(net1,Z','Classif');
        else
            Z = activations(net2,Z','Classif');
        end
        
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
[c,s,l]=pca(Z');
Z=s';
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








