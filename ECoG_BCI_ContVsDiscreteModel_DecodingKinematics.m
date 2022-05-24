
%% GOAL
% goal here is to compare continuous kinematic decoding (using GRU and KF
% vs discrete input based model).

%% ANALYSIS 1: USING ORIGINAL 6DOF

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
    figure;plot3(X(1,:),X(2,:),X(3,:),'LineWidth',1)
    hold on    
    plot3(Xhat(1,:),Xhat(2,:),Xhat(3,:),'LineWidth',1)
    plot3(Xhat1(1,:),Xhat1(2,:),Xhat1(3,:),'--k','LineWidth',1)
    title(['Target ID ' num2str(TrialData.TargetID)])
    plot3(X(1,1),X(2,1),X(3,1),'.g','MarkerSize',50)
    legend({'Ground Truth','ReFit KF','Input-based Discrete',''})    
    set(gcf,'Color','w')
    set(gca,'LineWidth',1)
    xlabel('X- axis')
    ylabel('Y- axis')
    zlabel('Z- axis')
    set(gca,'FontSize',14)
    %end
    
    % calculate deviations
    a = X(1:3,:) - Xhat(1:3,:);
    a = mean(sqrt(sum(a.^2)));
    KF_dev = [KF_dev (a)];
    
    
    a = X(1:3,:) - Xhat1(1:3,:);
    a = mean(sqrt(sum(a.^2)));
    MLP_dev = [MLP_dev (a)];
end

figure;boxplot([MLP_dev' KF_dev']/length(MLP_dev))
xticks(1:2)
xticklabels({'Input Driven','Re-Fit KF'})
ylabel('Average Position Deviation')
set(gcf,'Color','w')
set(gca,'FontSize',14)
set(gca,'LineWidth',1)
box off

[h p tb st]=ttest(MLP_dev,KF_dev)


% using a GRU with multiple bins of history (10samples at 10Hz - 1s) 


%% ANALYSIS 2: USING IMAGINED END POINT CONTROL OF THE ROBOT HAND