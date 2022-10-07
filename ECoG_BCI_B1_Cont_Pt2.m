%% TRAINING 9DOF DECODER



clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames = {'20220126','20220202','20220204','20220209','20220211'};
cd(root_path)

files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    if i==1
        D = D([1 2 7:13]);
    elseif i==2
        D = D([1 2 5:9 11:17]);
    elseif i==3
         D = D([1 2 3:6 8:end]);    
    end
        
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        if ~exist(filepath)
            filepath=fullfile(folderpath,D(j).name,'Imagined');
        end
        files = [files;findfiles('',filepath)'];
        disp([length(files)])
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
D8=[];
D9=[];
for i=1:length(files)
    disp(i)
    load(files{i});
    
    
    features  = TrialData.SmoothedNeuralFeatures;
    kinax = TrialData.TaskState;
    kinax = find(kinax==3);
    temp = cell2mat(features(kinax));
    
    % get smoothed delta hg and beta features
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
    elseif TrialData.TargetID == 7
        D7 = [D7 temp];
    elseif TrialData.TargetID == 8
        D8 = [D8 temp];
    elseif TrialData.TargetID == 9
        D9 = [D9 temp];    
    end
    
    
end

clear condn_data
idx = [1:96];
condn_data{1}=[D1(idx,:) ]';
condn_data{2}= [D2(idx,:)]';
condn_data{3}=[D3(idx,:)]';
condn_data{4}=[D4(idx,:)]';
condn_data{5}=[D5(idx,:)]';
condn_data{6}=[D6(idx,:)]';
condn_data{7}=[D7(idx,:)]';
condn_data{8}=[D8(idx,:)]';
condn_data{9}=[D9(idx,:)]';

% data augmentation
for i=1:length(condn_data)
   tmp =  condn_data{i};
   tmp_old = tmp;
   % random draws of 10 trials, average and add small noise, 200 times
   for iter=1:250
       idx = randperm(size(tmp,1),10);
       tmp1 =  tmp(idx,:);
       tmp1 = mean(tmp1,1);
       tmp1 = tmp1 + 0.1*randn(size(tmp1));
       tmp_old = [tmp_old;tmp1];
   end
   condn_data{i} = tmp_old;
end

A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};
E = condn_data{5};
F = condn_data{6};
G = condn_data{7};
H = condn_data{8};
I = condn_data{9};



clear N
N = [A' B' C' D' E' F' G' H' I' ];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1);8*ones(size(H,1),1)...
    ;9*ones(size(I,1),1);];

T = zeros(size(T1,1),9);
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
[aa bb]=find(T1==8);[aa(1) aa(end)]
T(aa(1):aa(end),8)=1;
[aa bb]=find(T1==9);[aa(1) aa(end)]
T(aa(1):aa(end),9)=1;



% code to train a neural network
clear net
net = patternnet([64 64 64]) ;
net.performParam.regularization=0.2;
net.divideParam.trainRatio=0.85;
net.divideParam.valRatio=0.15;
net.divideParam.testRatio=0;
net = train(net,N,T','UseParallel','yes');

net_9DoF = net;
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
save net_9DoF net_9DoF
genFunction(net_9DoF,'MLP_9DoF_Days1to5')

% 
% cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
% genFunction(net,'MLP_9DoF_Till_02102022')

%% LOOKING AT BR DATA AND STREAMING IN THE ANIN TO PULL OUT TRIAL STRUCTURE

clc;clear
cd('C:\Users\nikic\Documents\GitHub\BR_Ecog_Visualization')
blackrock2mat

cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220929\BlackRock\20220929-103233')
load('20220929-103233-001.mat')

anin=-anin(:,2);
tt=(1/Fs)*(0:length(anin)-1);
figure;plot(tt,anin)
xlabel('Time in sec')







