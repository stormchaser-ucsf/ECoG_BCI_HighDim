
%% GOAL
% goal here is to compare continuous kinematic decoding (using GRU and KF
% vs discrete input based model).

%% ANALYSIS 1: USING ORIGINAL 6DOF BUT IN A CONTINUOUS VEL MAPPING FRAMEWORK

clc;clear
close all



folderpath = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220520\Robot';
folders={'141735','142458','143022','144223','144607','145327','145327'};

files=[];
for i=1:length(folders)
    filepath = fullfile(folderpath,folders{i},'Imagined');
    files=[files;findfiles('',filepath)'];
end

% set aside 10% of files for testing and 90% for training
idx = randperm(length(files),round(0.8*length(files)));
files_train = files(idx);
I=ones(length(files),1);
I(idx)=0;
test_files = files(logical(I));

% get the training data
neural=[];
kinematics=[];
D1=[];D2=[];D3=[];D4=[];D5=[];D6=[];
for i=1:length(files_train)
    disp(i/length(files_train)*100)
    load(files_train{i})
    idx=find(TrialData.TaskState==3);
    kin = TrialData.CursorState;
    kin = kin(:,idx);
    neural_features = TrialData.SmoothedNeuralFeatures;
    neural_features = cell2mat(neural_features(idx));
    temp=neural_features;
    fidx = [129:256 513:640 769:896];
    neural_features = neural_features(fidx,:);
    %neural = [neural neural_features];
    kinematics = [kinematics kin];

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
    neural = [neural temp];

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

    % getting data for GRU -> hG and Low Freq Oscill. (<20Hz) at 10Hz
    raw_data = TrialData.BroadbandData;
    raw_data = cell2mat(raw_data(idx)');

end


%%%%%%% KALMAN FILTER %%%
% equations for KF
% model: Xhat(t+1) = A*Xhat(t) + K[y(t)-C*A*Xhat(t)]
% state: X(t+1) = A*X(t) + w(t) ; need a W noise cov matrix
% measurements: y(t) = C*X(t) + q(t); need a Q noise cov matrix
% C and Q estimated using least squares
% W is set a priori
% A is also set a priori


% state data in KF
X=kinematics;
X(7,:)=1;

% measurement data in KF
Y=neural;

% A matrix
A=zeros(7);
A(1:6,1:6) = TrialData.Params.dA;
A(7,7)=1;

% velocity W matrix
W = zeros(7);
for u=4:6
    W(u,u)=150;
end

% least squares estimates of velocity C
C = Y/(X(4:end,:));
C = [zeros(size(C,1),3) C];

% get estimates of Q
q = Y-C*X;
Q= cov(q');

% recursively estimate Kalman Gain
P = eye(7);
chk=1;
counter=0;
norm_val=[];
while chk>1e-10
    temp_norm = norm(P(:));
    P = A*P*A' + W;
    P(1:3,:) = zeros(3,size(P,2));
    P(:,1:3) = zeros(size(P,1),3);
    P(end,end) = 0;
    K = P*C'*pinv(C*P*C' + Q);
    P = P - K*C*P;
    chk = abs(temp_norm - norm(P(:)));
    counter=counter+1;
    norm_val = [norm_val chk];
end


%%%%% build a classifier using simple MLP
idx = [1:96];
condn_data{1}=[D1(idx,:) ]';
condn_data{2}= [D2(idx,:)]';
condn_data{3}=[D3(idx,:)]';
condn_data{4}=[D4(idx,:)]';
condn_data{5}=[D5(idx,:)]';
condn_data{6}=[D6(idx,:)]';

A1 = condn_data{1};
B = condn_data{2};
C1 = condn_data{3};
D = condn_data{4};
E = condn_data{5};
F = condn_data{6};


clear N
N = [A1' B' C1' D' E' F' ];
T1 = [ones(size(A1,1),1);2*ones(size(B,1),1);3*ones(size(C1,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1)];

T = zeros(size(T1,1),6);
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
net = patternnet([64 64 ]) ;
net.performParam.regularization=0.2;
net = train(net,N,T','UseGPU','yes');


% test the model out on held out trials
KF_dev=[];
MLP_dev=[];
for i=1:length(test_files)
    load(test_files{i})
    idx=find(TrialData.TaskState==3);
    kin = TrialData.CursorState;
    kin = kin(:,idx);
    neural_features = TrialData.SmoothedNeuralFeatures;
    neural_features = cell2mat(neural_features(idx));
    temp=neural_features;
    fidx = [129:256 513:640 769:896];
    neural_features = neural_features(fidx,:);

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
    neural_features = [temp];



    % run it through KF
    Y = neural_features;
    X = kin;
    X(end+1,:)=1;

    Xhat=1e-6*zeros(size(X));
    Xhat(:,1)=X(:,1);
    for j=2:size(X,2)
        x = A*Xhat(:,j-1);
        x = x + K*(Y(:,j) - C*x);
        Xhat(:,j)=x;
    end

    % run it through MLP
    B =  TrialData.Params.dB;
    out  = net(neural_features);
    Xhat1=1e-6*zeros(size(X));
    Xhat1 = Xhat1(1:end-1,:);
    Xhat1(:,1)=X(1:end-1,1);
    dt=1/TrialData.Params.UpdateRate;
    for j=2:size(X,2)
        xt  = Xhat1(:,j);
        xtm1 = Xhat1(:,j-1);
        tmp = out(:,j);
        [aa bb]=max(tmp);
        if bb==1
            v=[1;0;0];
        elseif bb==2
            v=[0;1;0];
        elseif bb==3
            v=[-1;0;0];
        elseif bb==4
            v=[0;-1;0];
        elseif bb==5
            v=[0;0;1];
        elseif bb==6
            v=[0;0;-1];
        end
        u=[60*v*dt ;zeros(3,1)];
        xt = xtm1 + u;
        Xhat1(:,j)=xt;
    end

    %if TrialData.TargetID==6
    pos=TrialData.TargetPosition;
    figure;plot3(X(1,:),X(2,:),X(3,:),'LineWidth',1)
    hold on
    plot3(Xhat(1,:),Xhat(2,:),Xhat(3,:),'LineWidth',1)
    plot3(Xhat1(1,:),Xhat1(2,:),Xhat1(3,:),'--k','LineWidth',1)
    title(['Target ID ' num2str(TrialData.TargetID)])
    plot3(X(1,1),X(2,1),X(3,1),'.g','MarkerSize',50)
    plot3(pos(1),pos(2),pos(3),'sb','MarkerSize',50,'MarkerEdgeColor','b',...
        'MarkerFaceColor',[0.2,0.2,0.2],'LineWidth',2)
    legend({'Ground Truth','ReFit KF','IBID','',''})
    set(gcf,'Color','w')
    set(gca,'LineWidth',1)
    xlabel('X- axis')
    ylabel('Y- axis')
    zlabel('Z- axis')
    set(gca,'FontSize',14)
    %end

    % calculate deviations from robot's trajectory
    %     a = X(1:3,:) - Xhat(1:3,:);
    %     a = mean(sqrt(sum(a.^2)));
    %     KF_dev = [KF_dev (a)];
    %
    %
    %     a = X(1:3,:) - Xhat1(1:3,:);
    %     a = mean(sqrt(sum(a.^2)));
    %     MLP_dev = [MLP_dev (a)];

    % calculate deviations based on ground truth position
    ax = find(pos==0);
    a=sqrt(sum(sum(Xhat(ax,:).^2)))/size(Xhat,2);
    %a=mean(sqrt(sum(Xhat(ax,:).^2)));
    KF_dev = [KF_dev (a)];

    a=sqrt(sum(sum(Xhat1(ax,:).^2)))/size(Xhat1,2);
    %a=mean(sqrt(sum(Xhat1(ax,:).^2)));
    MLP_dev = [MLP_dev (a)];


end

%figure;boxplot([MLP_dev' KF_dev']/length(MLP_dev))
figure;boxplot([MLP_dev' KF_dev']/1)
xticks(1:2)
xticklabels({'IBID','Re-Fit KF'})
ylabel('Average Deviation')
set(gcf,'Color','w')
set(gca,'FontSize',14)
set(gca,'LineWidth',1)
box off

[h p tb st]=ttest(MLP_dev,KF_dev)


%%  SAME AS ABOVE BUT USING A GRU
% using a sequence to sequence classifcation sceheme where o/p is class and
% i/p are neural features.

%Xtrain - cell array of sequences, features, time
%Ytrain - cell array of sequences and outputs
% can concatenate to make it one sequence of model each trial as a sequence

% also use in a regression model

clc;clear
close all


folderpath = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220520\Robot';
folders={'141735','142458','143022','144223','144607','145327','145327'};

files=[];
for i=1:length(folders)
    filepath = fullfile(folderpath,folders{i},'Imagined');
    files=[files;findfiles('',filepath)'];
end

% set aside 10% of files for testing and 90% for training
idx = randperm(length(files),round(0.8*length(files)));
files_train = files(idx);
I=ones(length(files),1);
I(idx)=0;
test_files = files(logical(I));

% get the training data
% time-pts X channels X samples is data format
neural={};
kinematics=[];
kinematics_tid=[];
for ii=1:length(files_train)
    disp(ii/length(files_train)*100)
    load(files_train{ii})
    idx0=find(TrialData.TaskState==2) ;
    idx=[find(TrialData.TaskState==3)];
    idx1=[find(TrialData.TaskState==4)];
    idx1=idx1(1:1);
    idx=[idx idx1];

    kin = TrialData.CursorState;
    kin = kin(:,idx);

    neural_features = TrialData.NeuralFeatures;
    neural_features = cell2mat(neural_features(idx));

    temp=neural_features;
    fidx = [ 769:896];
    neural_features = neural_features(fidx,:);
    %neural = [neural neural_features];
    %kinematics = [kinematics kin];

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

    %     %2-norm
    %     for j=1:size(temp,2)
    %         temp(:,j) = temp(:,j)./norm(temp(:,j));
    %     end


    % get all the data at once
    %neural = cat(3,neural,temp);
    %kinematics_tid = cat(2,kinematics_tid,repmat(TrialData.TargetID,size(temp,2),1));

    % get data as a cell array
    neural{ii} = temp;
    %kinematics_tid(ii)=TrialData.TargetID;
    %kinematics_tid{ii} = categorical(((repmat(TrialData.TargetID,1,size(temp,2)))));
    %     if TrialData.TargetID ==1
    %         tmp = categorical(repmat(TrialData.TargetID,1,size(temp,2)));
    %     elseif TrialData.TargetID ==2
    %         tmp = repmat(categorical(cellstr('Two')),1,1);
    %     elseif TrialData.TargetID ==3
    %         tmp = repmat(categorical(cellstr('Three')),1,1);
    %     elseif TrialData.TargetID ==4
    %         tmp = repmat(categorical(cellstr('Four')),1,1);
    %     elseif TrialData.TargetID ==5
    %         tmp = repmat(categorical(cellstr('Five')),1,1);
    %     elseif TrialData.TargetID ==6
    %         tmp = repmat(categorical(cellstr('Six')),1,1);
    %     end
    tmp = categorical(repmat(TrialData.TargetID,1,size(temp,2)));
    kinematics_tid{ii}=tmp;
    kinematics{ii} = kin(4:6,:);



    %     % get the samples for the lstm
    %     for j=1:2:size(temp,2)
    %         if (j+4)<size(temp,2)
    %             neural=cat(3,neural,temp(:,j:j+4)');
    %             kinematics=cat(3,kinematics,kin(:,j:j+4)');
    %             kinematics_tid=[kinematics_tid;TrialData.TargetID];
    %         end
    %     end
end

condn_data_new = neural';
Y = kinematics_tid';
Y1 = kinematics';



%%%%% train a discrete GRU
% validation split
%idx = randperm(size(condn_data_new,1),round(0.8*size(condn_data_new,1)));
%I = ones(size(condn_data_new,1),1);
%I(idx)=0;

XTrain = condn_data_new;
%XTest = condn_data_new(logical(I));
YTrain = Y;
%YTest = categorical(Y(logical(I)));

% specify lstm structure
inputSize = 96;
numHiddenUnits = [64];
drop = [0.5];
numClasses = 6;
layers = [sequenceInputLayer(inputSize)
    gruLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer(drop)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% training options
options = trainingOptions('adam', ...
    'MaxEpochs',20, ...
    'MiniBatchSize',8, ...
    'GradientThreshold',2, ...
    'Verbose',true, ...
    'ValidationFrequency',4,...
    'Shuffle','every-epoch', ...
    'ValidationPatience',5,...
    'Plots','training-progress');

%'ValidationData',{XTest,YTest},...

% train the model
net = trainNetwork(XTrain,YTrain,layers,options);

%%%% train a kinematics decoding GRU
%validation split
idx = randperm(size(condn_data_new,1),round(0.8*size(condn_data_new,1)));
I = ones(size(condn_data_new,1),1);
I(idx)=0;

XTrain = condn_data_new;
XTest = condn_data_new(logical(I));
YTrain = Y1;
YTest = (Y1(logical(I)));

% specify lstm structure
inputSize = 96;
numHiddenUnits = [64];
drop = [0.5];
kinDimension = 3;
layers = [sequenceInputLayer(inputSize)
    gruLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer(drop)
    fullyConnectedLayer(kinDimension)
    regressionLayer];

% training options
options = trainingOptions('adam', ...
    'MaxEpochs',200, ...
    'MiniBatchSize',16, ...
    'GradientThreshold',2, ...
    'Verbose',true, ...
    'ValidationFrequency',4,...
    'Shuffle','every-epoch', ...
    'ValidationPatience',5,...
    'ValidationData',{XTest,YTest},...
    'Plots','training-progress');

%

% train the model
net1 = trainNetwork(XTrain,YTrain,layers,options);



% test the model out on held out trials
gru_dev=[];
gru_kin_dev=[];
acc=zeros(6);
for ii=1:length(test_files)
    load(test_files{ii})

    idx=find(TrialData.TaskState==3);
    idx1=[find(TrialData.TaskState==4)];
    idx1=idx1(1:1);
    idx=[idx idx1];

    kin = TrialData.CursorState;
    kin = kin(:,idx);
    neural_features = TrialData.NeuralFeatures;
    neural_features = cell2mat(neural_features(idx));
    temp=neural_features;
    fidx = [ 769:896];
    neural_features = neural_features(fidx,:);

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

    % get predictions for the entire sequence
    act=predict(net,temp);

    % classify and generate trajectory
    X=kin;
    Xhat1=zeros(size(X));
    Xhat1(:,1)=X(:,1);
    dt=1/TrialData.Params.UpdateRate;
    for j=2:size(X,2)
        xt  = Xhat1(:,j);
        xtm1 = Xhat1(:,j-1);
        tmp = act(:,j);
        [aa bb]=max(tmp);
        if bb==1
            v=[1;0;0];
        elseif bb==2
            v=[0;1;0];
        elseif bb==3
            v=[-1;0;0];
        elseif bb==4
            v=[0;-1;0];
        elseif bb==5
            v=[0;0;1];
        elseif bb==6
            v=[0;0;-1];
        end
        u=[60*v*dt ;zeros(3,1)];
        xt = xtm1 + u;
        Xhat1(:,j)=xt;
    end

    % use GRU for kinematics decoding
    act=predict(net1,temp); % these are all the velocities at individdual time-points
    Xhat = zeros(size(X));
    Xhat(:,1) = X(:,1);
    A=TrialData.Params.dA;
    for j=2:size(act,2)
        Xhat(4:6,j-1) = act(:,j-1);
        Xhat(:,j) = A*Xhat(:,j-1);
    end


    % plotting
    figure;plot3(X(1,:),X(2,:),X(3,:),'LineWidth',1)
    hold on
    plot3(Xhat(1,:),Xhat(2,:),Xhat(3,:),'LineWidth',1)
    plot3(Xhat1(1,:),Xhat1(2,:),Xhat1(3,:),'--k','LineWidth',1)
    title(['Target ID ' num2str(TrialData.TargetID)])
    plot3(X(1,1),X(2,1),X(3,1),'.g','MarkerSize',50)
    legend({'Ground Truth','GRU velocity decoding','Input-based Discrete',''})
    set(gcf,'Color','w')
    set(gca,'LineWidth',1)
    xlabel('X- axis')
    ylabel('Y- axis')
    zlabel('Z- axis')
    set(gca,'FontSize',14)


    % position deviation
    pos=TrialData.TargetPosition;
    ax = find(pos==0);

    % disrete gru
    a=sqrt(sum(sum(Xhat1(ax,:).^2)))/size(Xhat1,2);
    gru_dev = [gru_dev (a)];

    % kin gru
    a=sqrt(sum(sum(Xhat(ax,:).^2)))/size(Xhat,2);
    gru_kin_dev = [gru_kin_dev (a)];




end


figure;boxplot([gru_dev' gru_kin_dev'])


%% (MAIN: KF) ANALYSIS 2: USING IMAGINED END POINT CONTROL OF THE ROBOT HAND (MAIN)
% SMOOTH BATCH KALMAN FILTER (MAIN)

% STEP 1: Take the first 2sec of online data and the first 3s of Imagined
% data for the iAE analyses

% STEP 2:
% take the first 2 sec of online data and look at trajectories towards the
% target

clc;clear;
close all
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
cd(root_path)

%day1
session_data(1).Day = '20230220';
session_data(1).folders = {'102528','103151','103544','104517','105110','110103',...
    '110650','111006','112144','112608','113541','114821'};
session_data(1).folder_type={'I','I','I','I','I','O','O','O','B','B','B','B'};

% day 2
session_data(2).Day = '20230224';
session_data(2).folders = {'103522','103942','104105','104228','104354',...
    '104954','105421','105712',...
    '110944','111424','111723','112021',...
    };
session_data(2).folder_type={'I','I','I','I','I','O','O','O','B','B','B','B'};

% day 3
session_data(3).Day = '20230301';
session_data(3).folders = {'101030','101443','101604','101716','101834','101947',...
    '102646','103040','103331','103632',...
    '104548','104845','105142','105437','105728',...
    '110536','110917','111210','111509'};
session_data(3).folder_type={'I','I','I','I','I','I',...
    'O','O','O','O',...
    'B','B','B','B','B',...
    'B','B','B','B'};

% day 4
session_data(4).Day = '20230315';
session_data(4).folders = {'103701','104112','104246','104421','104548',...
    '105260','105948','110323','110631'...
    '111229','111639','112018','112331',...
    '113004','113259','113554','113904'
    };
session_data(4).folder_type={'I','I','I','I','I',...
    'O','O','O','O',...
    'B','B','B','B',...
    'B','B','B','B'};

% day 5
session_data(5).Day = '20230322';
session_data(5).folders = {'103916','104251','104414','104531','104647',...
    '105507','105812','110106','110345'...
    '111337','111656','111953','112247',...
    '113330','113651','113946','114241'
    };
session_data(5).folder_type={'I','I','I','I','I',...
    'O','O','O','O',...
    'B','B','B','B',...
    'B','B','B','B'};

% also make sure to load the data i.e., the neural activity corresponding
% to an intended decode that is within 30degrees towards target.
%FIRST is however, on just the imagined movement data itself


mahab_full_online=[];
mahab_full_imagined=[];
mahab_full_batch=[];
kin_data = [];
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
        folderpath = fullfile(root_path, day_date,'RobotKF',folders{ii},'Imagined');
        %cd(folderpath)
        files = [files;findfiles('',folderpath)'];
    end

    %load the data
    %condn_data = load_data_for_MLP_CKD(files);
    condn_data = load_data_for_MLP(files,1);
    %
    % save the data
    filename = ['Biomimetic_CenterOut_condn_data_Imagined_Day_45deg' num2str(i)];
    save(filename, 'condn_data', '-v7.3')
    %
    % get the mahab distance in the full dataset
    Dimagined = mahal2_full(condn_data);
    Dimagined = triu(Dimagined);
    Dimagined = Dimagined(Dimagined>0);
    mahab_full_imagined = [mahab_full_imagined Dimagined];


    %%%%%%online data
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'RobotKF',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end


    %load the data
    %condn_data = load_data_for_MLP_CKD(files);
    condn_data = load_data_for_MLP(files,1);

    % save the data
    filename = ['Biomimetic_CenterOut_condn_data_Online_Day_45deg' num2str(i)];
    save(filename, 'condn_data', '-v7.3')

    % get the mahab distance in the full dataset
    Donline = mahal2_full(condn_data);
    Donline = triu(Donline);
    Donline = Donline(Donline>0);
    mahab_full_online = [mahab_full_online Donline];


    % get the kinematics
    kin_data_tmp = get_kinematics(files);
    kin_data = cat(1,kin_data,kin_data_tmp');



    %%%%%%batch data
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'RobotKF',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end


    %load the data
    %condn_data = load_data_for_MLP_CKD(files);
    condn_data = load_data_for_MLP(files,1);

    %save the data
    filename = ['Biomimetic_CenterOut_condn_data_Batch_Day_45deg' num2str(i)];
    save(filename, 'condn_data', '-v7.3')


    % get the mahab distance in the full dataset
    Donline = mahal2_full(condn_data);
    Donline = triu(Donline);
    Donline = Donline(Donline>0);
    mahab_full_batch = [mahab_full_batch Donline];

    % get the kinematics
    kin_data_tmp = get_kinematics(files);
    kin_data = cat(1,kin_data,kin_data_tmp');

end
%

close all

figure;
boxplot([mahab_full_imagined mahab_full_online mahab_full_batch])

figure;
boxplot([mahab_full_imagined(:) mahab_full_online(:) mahab_full_batch(:)])




% plot the trajectories with the lowest errors
figure;
hold on
view(40,40)
xlim([-200,200]*0.5)
ylim([-200,200]*0.5)
zlim([-200,200]*0.5)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('X-axis')
ylabel('Y-axis')
zlabel('Z-axis')
cmap = turbo(6);
recon_error=[];
for i=1:6
    idx = find([kin_data(:).TargetID]==i);
    errors = [kin_data(idx).error ];
    [aa bb] = sort(errors);
    bb = idx(bb(1:2));
    col = cmap(i,:);

    for j=1:length(bb)
        filename = kin_data(bb(j)).filename;
        disp(filename)
        file_loaded=1;
        try
            load(filename);
        catch
            file_loaded=0;
        end
        if file_loaded
            kinax = TrialData.TaskState;
            kinax = [find(kinax==3)];
            % get the kinematics data and the target data
            kindata = TrialData.CursorState;
            kindata = kindata(1:3,kinax);
            kindata = kindata(:,1:20) - kindata(:,1);
            plot3(kindata(1,:),kindata(2,:),kindata(3,:),'LineWidth',2,'Color',col)

            % get the errors in terms of deviation from the ideal path
            target = TrialData.TargetPosition;
            kin = kindata;
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
            recon_error =[recon_error; sqrt(sum(tmp_error))];
        end
    end
    %waitforbuttonpress
end
grid on

recon_error1 = load('recon_error_IBID.mat');
recon_error1 = recon_error1.recon_error;

recon_error(end+1:length(recon_error1)) = NaN;
figure;boxplot(([recon_error recon_error1]/14),'whisker',2.5)
set(gcf,'Color','w')
set(gca,'FontSize',14)
xticks(1:2)
xticklabels({'CKD','CIDC'})
ylabel('Trajectory Error')
box off
set(gca,'LineWidth',1)


%% KF 3D end point control: getting batch sizes and half lives



clc;clear;
close all
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')
cd(root_path)

%day1
session_data(1).Day = '20230220';
session_data(1).folders = {'102528','103151','103544','104517','105110','110103',...
    '110650','111006','112144','112608','113541','114821'};
session_data(1).folder_type={'I','I','I','I','I','O','O','O','B','B','B','B'};

% day 2
session_data(2).Day = '20230224';
session_data(2).folders = {'103522','103942','104105','104228','104354',...
    '104954','105421','105712',...
    '110944','111424','111723','112021',...
    };
session_data(2).folder_type={'I','I','I','I','I','O','O','O','B','B','B','B'};

% day 3
session_data(3).Day = '20230301';
session_data(3).folders = {'101030','101443','101604','101716','101834','101947',...
    '102646','103040','103331','103632',...
    '104548','104845','105142','105437','105728',...
    '110536','110917','111210','111509'};
session_data(3).folder_type={'I','I','I','I','I','I',...
    'O','O','O','O',...
    'B','B','B','B','B',...
    'B','B','B','B'};

% day 4
session_data(4).Day = '20230315';
session_data(4).folders = {'103701','104112','104246','104421','104548',...
    '105260','105948','110323','110631'...
    '111229','111639','112018','112331',...
    '113004','113259','113554','113904'
    };
session_data(4).folder_type={'I','I','I','I','I',...
    'O','O','O','O',...
    'B','B','B','B',...
    'B','B','B','B'};

% day 5
session_data(5).Day = '20230322';
session_data(5).folders = {'103916','104251','104414','104531','104647',...
    '105507','105812','110106','110345'...
    '111337','111656','111953','112247',...
    '113330','113651','113946','114241'
    };
session_data(5).folder_type={'I','I','I','I','I',...
    'O','O','O','O',...
    'B','B','B','B',...
    'B','B','B','B'};

half_lives=[];
for i=1:length(session_data)
    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    batch_idx = find(folders_batch==1);

    %%%%%%online data    
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'RobotKF',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end

    [neural,kinematics]=get_neural_kinematics_KF(files,1);
    half_lives = [half_lives (size(neural,2)*(1/5)) * log(0.5)/log(0.75)];

     %%%%%%batch data    
    folders = session_data(i).folders(batch_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'RobotKF',folders{ii},'BCI_Fixed');
        files = [files;findfiles('',folderpath)'];
    end

    [neural,kinematics]=get_neural_kinematics_KF(files,1);
    half_lives = [half_lives (size(neural,2)*(1/5)) * log(0.5)/log(0.75)];
end

figure;hist(half_lives)
vline(mean(half_lives))
vline(median(half_lives))




%% PLAYING AROUND WITH NORMALIZING DATA

% rows are observations, columns are features
a=randn(20,50);
b=randn(20,50)*3;
figure;
stem(std(a))
hold on
stem(std(b))

a=a./std(a);
b=b./std(b);

figure;
stem(std(a))
hold on
stem(std(b))

% norm each column
for i=1:size(a,2)
    a(:,i) = a(:,i)./norm(a(:,i));
end
for i=1:size(b,2)
    b(:,i) = b(:,i)./norm(b(:,i));
end

figure;
stem(std(a))
hold on
stem(std(b))



