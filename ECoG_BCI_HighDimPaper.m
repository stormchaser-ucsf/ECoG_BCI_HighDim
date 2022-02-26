

% classifying hand actions

clc;clear
filepath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
dates = {'20220128','20220204'};

files=[];
for i=1:length(dates)
    full_path = fullfile(filepath,dates{i},'Hand') ;
    tmp = findfiles('.mat',full_path,1)';
    for j=1:length(tmp)
        if isempty(regexp(tmp{j},'params'))
            files=[files;tmp(j)];
        end
    end
end


% load the files
D1=[];D2=[];D3=[];D4=[];D5=[];D6=[];D7=[];D8=[];D9=[];D10=[];
for i=1:length(files)
    disp(i)
    load(files{i})
    
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
    elseif TrialData.TargetID == 10
        D10 = [D10 temp];
    end
end


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
condn_data{10}=[D10(idx,:)]';

A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};
E = condn_data{5};
F = condn_data{6};
G = condn_data{7};
H = condn_data{8};
I = condn_data{9};
J = condn_data{10};


clear N
N = [A' B' C' D' E' F' G' H' I' J'];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1);8*ones(size(H,1),1)...
    ;9*ones(size(I,1),1);10*ones(size(J,1),1)];

T = zeros(size(T1,1),10);
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
[aa bb]=find(T1==10);[aa(1) aa(end)]
T(aa(1):aa(end),10)=1;


% code to train a neural network
clear net
net = patternnet([96 96 96]) ;
net.performParam.regularization=0.2;
net = train(net,N,T','UseGPU','yes');


%% classifying at trial level on held out trials


clc;clear
filepath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
dates = {'20220128','20220204','20220209','20220218'};

files=[];
for i=1:length(dates)
    full_path = fullfile(filepath,dates{i},'Hand') ;
    tmp = findfiles('.mat',full_path,1)';
    for j=1:length(tmp)
        if isempty(regexp(tmp{j},'params'))
            files=[files;tmp(j)];
        end
    end
end

for iter=1:5
    
    train_files_idx = round(0.8*length(files));
    idx = randperm(length(files),train_files_idx);
    train_files = files(idx);
    I = ones(length(files),1);
    I(idx)=0;
    test_files = files(logical(I));
    
    % train the model on the training data
    D1=[];D2=[];D3=[];D4=[];D5=[];D6=[];D7=[];D8=[];D9=[];D10=[];
    for i=1:length(train_files)
        disp(i)
        load(train_files{i})
        
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
        elseif TrialData.TargetID == 10
            D10 = [D10 temp];
        end
    end
    
    
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
    condn_data{10}=[D10(idx,:)]';
    
    A = condn_data{1};
    B = condn_data{2};
    C = condn_data{3};
    D = condn_data{4};
    E = condn_data{5};
    F = condn_data{6};
    G = condn_data{7};
    H = condn_data{8};
    I = condn_data{9};
    J = condn_data{10};
    
    
    clear N
    N = [A' B' C' D' E' F' G' H' I' J'];
    T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
        5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1);8*ones(size(H,1),1)...
        ;9*ones(size(I,1),1);10*ones(size(J,1),1)];
    
    T = zeros(size(T1,1),10);
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
    [aa bb]=find(T1==10);[aa(1) aa(end)]
    T(aa(1):aa(end),10)=1;
    
    
    % code to train a neural network
    clear net
    net = patternnet([64 64 64]) ;
    net.performParam.regularization=0.2;
    net.divideParam.trainRatio = 0.8;
    net.divideParam.valRatio = 0.2;
    net.divideParam.testRatio = 0.0;
    net = train(net,N,T','UseGPU','yes');
    
    
    % test on held out trials
    acc = zeros(10);
    for i=1:length(test_files)
        disp(i)
        load(test_files{i})
        features  = TrialData.SmoothedNeuralFeatures;
        kinax = TrialData.TaskState;
        kinax = find(kinax==3);
        clear temp
        temp = cell2mat(features(kinax));
        temp = temp(:,1:20);
        
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
        
        % run it thru the model
        out = net(temp);
        decodes=[];
        for j=1:size(out,2)
            [aa bb]=max(out(:,j));
            decodes(j) = bb;
        end
        
        decodes = mode(decodes);
        acc(TrialData.TargetID,decodes)=acc(TrialData.TargetID,decodes)+1;
    end
    
    for i=1:size(acc,1)
        acc(i,:)=acc(i,:)./sum(acc(i,:));
    end
    
    acc_overall(iter,:,:) = acc;
    
end

acc=squeeze(nanmean(acc_overall,1));
diag(acc)
mean(ans)
figure;imagesc(acc)
colormap bone
xticks(1:10)
yticks(1:10)
caxis([0 .55])
yticklabels({'Thumb','Index','Middle','Ring','Pinky','Pinch','Tripod','Power','Add','Abd'})
xticklabels({'Thumb','Index','Middle','Ring','Pinky','Pinch','Tripod','Power','Add','Abd'})
set(gcf,'Color','w')
title('Accuracy on held out trials')
set(gca,'FontSize',12)

figure;stem(diag(acc))
xlim([0 11])
hline(0.1,'r')
xticks(1:10)
xticklabels({'Thumb','Index','Middle','Ring','Pinky','Pinch','Tripod','Power','Add','Abd'})
box off
ylim([0 0.6])
set(gcf,'Color','w')

%% ERPS OF THE HAND ACTIONS

clc;clear

root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
foldernames = {'20220128'};
cd(root_path)

files=[];
for i=1%length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Hand')
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
D8=[];
D9=[];
D10=[];

for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.NeuralFeatures;
    features = cell2mat(features);
    %features = features(129:256,:); % delta
    features = features(769:end,:); % hG
    %features = features(513:640,:); %beta
    fs = TrialData.Params.UpdateRate;
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
    
    % get first 25 samples via interpolation
    tb = (1/fs)*[1:size(tmp_data,2)];
    t=(1/fs)*[1:25];
    tb = tb*t(end)/tb(end);
    tmp_data1 = interp1(tb,tmp_data',t,'spline')';
    idx3 = interp1(tb,idx3,t,'spline');
    
    % now stick all the data together
    data = [features(:,[state1 state2]) tmp_data1 features(:,[state4])];
    
    
    % now get the ERPs
    if TrialData.TargetID == 1
        D1 = cat(3,D1,data);
    elseif TrialData.TargetID == 2
        D2 = cat(3,D2,data);
    elseif TrialData.TargetID == 3
        D3 = cat(3,D3,data);
    elseif TrialData.TargetID == 4
        D4 = cat(3,D4,data);
    elseif TrialData.TargetID == 5
        D5 = cat(3,D5,data);
    elseif TrialData.TargetID == 6
        D6 = cat(3,D6,data);
    elseif TrialData.TargetID == 7
        D7 = cat(3,D7,data);
    elseif TrialData.TargetID == 8
        D8 = cat(3,D8,data);
    elseif TrialData.TargetID == 9
        D9 = cat(3,D9,data);
    elseif TrialData.TargetID == 10
        D10 = cat(3,D10,data);
    end
end

%
% % plot the ERPs now
% chMap=TrialData.Params.ChMap;
% figure
% ha=tight_subplot(8,16);
% d = 1;
% set(gcf,'Color','w')
% tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);
% for i = 1:size(D1,1)
%     [x y] = find(chMap==d);
%     if x == 1
%         axes(ha(y));
%         %subplot(8, 16, y)
%     else
%         s = 16*(x-1) + y;
%         axes(ha(s));
%         %subplot(8, 16, s)
%     end
%     hold on
%     erps =  squeeze(D6(i,:,:));
%     plot(erps, 'color', [0.4 0.4 0.4 ]);
%     ylabel (num2str(i))
%     axis tight
%     ylim([-2 2])
%     plot(mean(erps,2),'r','LineWidth',1.5)
%     %set(gca,'LineWidth',1)
%     %vline([time(2:4)])
%     h=vline(tim);
%     %set(h,'LineWidth',1)
%     set(h,'Color','b')
%     h=hline(0);
%     set(h,'LineWidth',1.5)
%     if i~=102
%         yticklabels ''
%         xticklabels ''
%     else
%         %xticks([tim])
%         %xticklabels({'S1','S2','S3','S4'})
%     end
%     d = d+1;
% end
% %
%
% bootstrapped confidnce intervals for a particular channel
% chdata = ch100;
%
% zscore the data to the first 8 time-bins
% tmp_data=chdata(1:8,:);
% m = mean(tmp_data(:));
% s = std(tmp_data(:));
% chdata = (chdata -m)./s;
%
%
% m = mean(chdata,2);
% mb = sort(bootstrp(1000,@mean,chdata'));
% figure;
% hold on
% plot(m,'b')
% plot(mb(25,:),'--b')
% plot(mb(975,:),'--b')
% hline(0)
%
% % shuffle the data and see the results
% tmp_mean=[];
% for i=1:1000
%     %tmp = circshift(chdata,randperm(size(chdata,1),1));
%     tmp = chdata;
%     tmp(randperm(numel(chdata))) = tmp;
%     tmp_data=tmp(1:8,:);
%     m = mean(tmp_data(:));
%     s = std(tmp_data(:));
%     tmp = (tmp -m)./s;
%     tmp_mean(i,:) = mean(tmp,2);
% end
%
% tmp_mean = sort(tmp_mean);
% plot(tmp_mean(25,:),'--r')
% plot(tmp_mean(975,:),'--r')

% plot the ERPs with bootstrapped C.I. shading
chMap=TrialData.Params.ChMap;
figure
ha=tight_subplot(8,16);
d = 1;
set(gcf,'Color','w')
tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);
for i = 1:size(D5,1)
    [x y] = find(chMap==i);
    if x == 1
        axes(ha(y));
        %subplot(8, 16, y)
    else
        s = 16*(x-1) + y;
        axes(ha(s));
        %subplot(8, 16, s)
    end
    hold on
    erps =  squeeze(D7(i,:,:));
    
    chdata = erps;
    % zscore the data to the first 8 time-bins
    tmp_data=chdata(1:6,:);
    m = mean(tmp_data(:));
    s = std(tmp_data(:));
    chdata = (chdata -m)./s;
    
    % get the confidence intervals
    m = mean(chdata,2);
    mb = sort(bootstrp(1000,@mean,chdata'));
    tt=1:size(erps,1);
    [fillhandle,msg]=jbfill(tt,mb(25,:),mb(975,:)...
        ,[0.3 0.3 0.7],[0.3 0.3 0.7],1,.2);
    hold on
    plot(m,'b')
    %plot(mb(25,:),'--b')
    %plot(mb(975,:),'--b')
    %hline(0)
    
    % shuffle the data for null confidence intervals
    tmp_mean=[];
    for j=1:1000
        %tmp = circshift(chdata,randperm(size(chdata,1),1));
        tmp = chdata;
        tmp(randperm(numel(chdata))) = tmp;
        tmp_data=tmp(1:6,:);
        m = mean(tmp_data(:));
        s = std(tmp_data(:));
        tmp = (tmp -m)./s;
        tmp_mean(j,:) = mean(tmp,2);
    end
    
    tmp_mean = sort(tmp_mean);
    %plot(tmp_mean(25,:),'--r')
    %plot(tmp_mean(975,:),'--r')
    [fillhandle,msg]=jbfill(tt,tmp_mean(25,:),tmp_mean(975,:)...
        ,[0.7 0.3 0.3],[0.7 0.3 0.3],1,.2);
    
    
    % statistical test
    % if the mean is outside confidence intervals in state 3
    m = mean(chdata,2);
    idx=10:30;
    mstat = m((idx));
    pval=[];
    for j=1:length(idx)
        pval(j) = (sum((mstat(j)) >= (tmp_mean(:,idx(j)))))./(size(tmp_mean,1));
    end
    
    res=sum((1-pval)<=0.025);
    if res>=floor(length(idx)/2)
        suc=1;
    else
        suc=0;
    end
    
    % beautify
    ylabel (num2str(i))
    axis tight
    ylim([-2 2])
    %set(gca,'LineWidth',1)
    %vline([time(2:4)])
    h=vline(tim);
    %set(h,'LineWidth',1)
    set(h,'Color','k')
    h=hline(0);
    set(h,'LineWidth',1.5)
    if i~=102
        yticklabels ''
        xticklabels ''
    else
        %xticks([tim])
        %xticklabels({'S1','S2','S3','S4'})
    end
    
    if suc==1
        box on
        set(gca,'LineWidth',2)
        set(gca,'XColor','g')
        set(gca,'YColor','g')
    end
    d = d+1;
end
sgtitle('Tripod Grasp')

%% Mahalanobis distance of imagined actions


clc;clear

root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
foldernames = {'20211201','20211203','20211206','20211208','20211215','20211217',...
'20220126','20220223'};
cd(root_path)

files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'ImaginedMvmtDAQ')
    D=dir(folderpath);
    if i==3
        D = D([1:3 5:7 9:end]);
    elseif i==4
        D = D([1:3 5:end]);
    elseif i==6
        D = D([1:5 7:end]);
    end
    
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'Imagined');
        tmp=dir(filepath);
        files = [files;findfiles('',filepath)'];
    end
end

ImaginedMvmt = {'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
    'Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp',...
    'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
    'Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
    'Squeeze Both Hands',...
    'Rotate Right Wrist','Rotate Left Wrist',...
    'Imagined Head Movement',...
    'Right Shoulder Shrug',...
    'Left Shoulder Shrug',...
    'Right Tricep','Left Tricep',...
    'Right Bicep','Left Bicep',...
    'Right Leg','Left Leg',...
    'Lips','Tongue'};

% %no bicep or tricep
% ImaginedMvmt = {'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
%     'Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp',...
%     'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
%     'Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
%     'Squeeze Both Hands',...
%     'Rotate Right Wrist','Rotate Left Wrist',...
%     'Imagined Head Movement',...
%     'Right Leg','Left Leg',...
%     'Lips','Tongue'};



Data={};
for i=1:length(ImaginedMvmt)
    Data{i}=zeros(0,0);
end


for i=1:length(files)
    disp(i/length(files)*100)
    load(files{i})
    
    features  = TrialData.NeuralFeatures;
    kinax = TrialData.TaskState;
    kinax = find(kinax==3);
    temp = cell2mat(features(kinax));
    
    % only hG
    %temp = temp([769:end],:);
    
    
    % hg and delta and beta
    temp = temp([1:128  769:end],:);
    
    % %    get smoothed delta hg and beta features
    %     new_temp=[];
    %     [xx yy] = size(TrialData.Params.ChMap);
    %     for k=1:size(temp,2)
    %         tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
    %         tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
    %         tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
    %         pooled_data=[];
    %         for i=1:2:xx
    %             for j=1:2:yy
    %                 delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
    %                 beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
    %                 hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
    %                 pooled_data = [pooled_data; delta; beta ;hg];
    %             end
    %         end
    %         new_temp= [new_temp pooled_data];
    %     end
    %      temp=new_temp;
    %      temp = temp([ 1:96 ],:);
    
    % save the data
    for j=1:length(ImaginedMvmt)
        if strcmp(ImaginedMvmt{j},TrialData.ImaginedAction)
            tmp=Data{j};
            tmp = [tmp temp];
            Data{j} = tmp;
            break
        end
    end
    
    
end

% get the mahalanobis distance between the imagined actions
D=zeros(length(ImaginedMvmt));
for i=1:length(Data)
    A = Data{i}';
    for j=i+1:length(Data)
        B = Data{j}';
        d = mahal2(A,B,1);
        D(i,j)=d;
        D(j,i)=d;
    end
end
figure;imagesc(D);
xticks(1:size(D,1))
yticks(1:size(D,1))
xticklabels(ImaginedMvmt)
yticklabels(ImaginedMvmt)
set(gcf,'Color','w')
%caxis([1 50])

Z = linkage(D,'ward');
figure;dendrogram(Z,0)
x = string(get(gca,'xticklabels'));
x1=[];
for i=1:length(x)
    tmp = str2num(x{i});
    x1 = [x1 ImaginedMvmt(tmp)];
end
xticklabels(x1)
set(gcf,'Color','w')


%title('pooled hg complete')


% subsample to get an unbiased estimate of the distance matrix
D1=[];
for iter=1:10
    disp(iter)
    D=zeros(length(ImaginedMvmt));
    for i=1:length(Data)
        A = Data{i}';
        idx = randperm(size(A,1),400);
        A=A(idx,:);
        for j=i+1:length(Data)
            B = Data{j}';
            idx = randperm(size(B,1),400);
            B=B(idx,:);
            d = mahal2(A,B,2);
            D(i,j)=d;
            D(j,i)=d;
        end
    end
    D1(iter,:,:)=D;
end

D=squeeze(nanmean(D1,1));
figure;imagesc(D);
xticks(1:size(D,1))
yticks(1:size(D,1))
xticklabels(ImaginedMvmt)
yticklabels(ImaginedMvmt)
set(gcf,'Color','w')
%caxis([1 50])

Z = linkage(D,'ward');
figure;dendrogram(Z,0)
x = string(get(gca,'xticklabels'));
x1=[];
for i=1:length(x)
    tmp = str2num(x{i});
    x1 = [x1 ImaginedMvmt(tmp)];
end
xticklabels(x1)
set(gcf,'Color','w')




%% MAHALANOBIS AFTER NORMAIZING DATA WITHIN A SESSION

clc;clear

root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
foldernames = {'20211201','20211203','20211206','20211208','20211215','20211217',...
'20220126','20220223'};
cd(root_path)

files=[]; % these are the foldernanes within each day
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'ImaginedMvmtDAQ')
    D=dir(folderpath);
    if i==3
        D = D([1:3 5:7 9:end]);
    elseif i==4
        D = D([1:3 5:end]);
    elseif i==6
        D = D([1:5 7:end]);
    end
    
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'Imagined');
        tmp=dir(filepath);
        files = [files;string(filepath)];
    end
end

ImaginedMvmt = {'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
    'Rotate Right Wrist','Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp',...
    'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
    'Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
    'Rotate Left Wrist','Squeeze Both Hands',...
    'Imagined Head Movement',...
    'Right Shoulder Shrug',...
    'Left Shoulder Shrug',...
    'Right Tricep','Left Tricep',...
    'Right Bicep','Left Bicep',...
    'Right Leg','Left Leg',...
    'Lips','Tongue'};

%no bicep or tricep
% ImaginedMvmt = {'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
%     'Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp','Rotate Right Wrist',...
%     'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
%     'Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
%     'Rotate Left Wrist',...
%     'Squeeze Both Hands',...
%     'Right Shoulder Shrug',...
%     'Left Shoulder Shrug',...
%     'Imagined Head Movement',...
%     'Right Leg','Left Leg',...
%     'Lips','Tongue'};



Data={};
for i=1:length(ImaginedMvmt)
    Data{i}=zeros(0,0);
end


for i=1:length(files)
    
    disp(i/length(files)*100)
    d=dir(files{i});
    len = length(d)-2;
    d=d(3:end);
    
    Data_tmp={};
    for ii=1:length(ImaginedMvmt)
        Data_tmp{ii}=zeros(0,0);
    end
    data_overall=[];
    cd(files{i})
    for jj=1:len
        load(d(jj).name)
        features  = TrialData.SmoothedNeuralFeatures;
        kinax = TrialData.TaskState;
        kinax = find(kinax==3);
        temp = cell2mat(features(kinax));
        temp=temp(:,4:end);
        
        % hg and delta and beta
        temp = temp([129:256 513:640 769:end],:);
        %temp = temp([ 769:end],:);
        
        %get smoothed delta hg and beta features
        %         new_temp=[];
        %         [xx yy] = size(TrialData.Params.ChMap);
        %         for k=1:size(temp,2)
        %             tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
        %             tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
        %             tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
        %             pooled_data=[];
        %             for i=1:2:xx
        %                 for j=1:2:yy
        %                     delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
        %                     beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
        %                     hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
        %                     pooled_data = [pooled_data; delta; beta ;hg];
        %                 end
        %             end
        %             new_temp= [new_temp pooled_data];
        %         end
        %         temp=new_temp;
        %         temp = temp([ 1:32 65:96 ],:);
        
        
        data_overall = [data_overall;temp'];
        
        
        
        for j=1:length(ImaginedMvmt)
            if strcmp(ImaginedMvmt{j},TrialData.ImaginedAction)
                tmp=Data_tmp{j};
                tmp = [tmp temp];
                Data_tmp{j} = tmp;
                break
            end
        end
    end
    
    m=mean(data_overall,1);
    s=std(data_overall,1);
    
    for j=1:length(Data_tmp)
        tmp=Data_tmp{j};
        tmp = (tmp-m')./s';
        Data_tmp{j}=tmp;
        
        % transfer to main file
        tmp=Data{j};
        tmp = [tmp Data_tmp{j}];
        Data{j} = tmp;
    end
end


% get the mahalanobis distance between the imagined actions
D=zeros(length(ImaginedMvmt));
for i=1:length(Data)
    A = Data{i}';
    for j=i+1:length(Data)
        B = Data{j}';
        d = mahal2(A,B,2);
        D(i,j)=d;
        D(j,i)=d;
    end
end
figure;imagesc(D);
xticks(1:size(D,1))
yticks(1:size(D,1))
xticklabels(ImaginedMvmt)
yticklabels(ImaginedMvmt)
set(gcf,'Color','w')
%colormap bone
%caxis([1 50])

Z = linkage(D,'ward');
figure;dendrogram(Z,0)
x = string(get(gca,'xticklabels'));
x1=[];
for i=1:length(x)
    tmp = str2num(x{i});
    x1 = [x1 ImaginedMvmt(tmp)];
end
xticklabels(x1)
set(gcf,'Color','w')


%title('pooled hg complete')

%
% % subsample to get an unbiased estimate of the distance matrix
% D1=[];
% for iter=1:20
%     disp(iter)
%     D=zeros(length(ImaginedMvmt));
%     for i=1:length(Data)
%         A = Data{i}';
%         idx = randperm(size(A,1),400);
%         A=A(idx,:);
%         for j=i+1:length(Data)
%             B = Data{j}';
%             idx = randperm(size(B,1),400);
%             B=B(idx,:);
%             d = mahal2(A,B,2);
%             D(i,j)=d;
%             D(j,i)=d;
%         end
%     end
%     D1(iter,:,:)=D;
% end
%
% D=squeeze(nanmean(D1,1));
% figure;imagesc(D);
% xticks(1:size(D,1))
% yticks(1:size(D,1))
% xticklabels(ImaginedMvmt)
% yticklabels(ImaginedMvmt)
% set(gcf,'Color','w')
% %caxis([1 50])
%
% Z = linkage(D,'complete');
% figure;dendrogram(Z,0)
% x = string(get(gca,'xticklabels'));
% x1=[];
% for i=1:length(x)
%    tmp = str2num(x{i});
%    x1 = [x1 ImaginedMvmt(tmp)];
% end
% xticklabels(x1)
% set(gcf,'Color','w')


% getting D from a multiclass SVM
condn_data={};
for i=1:length(Data)
    tmp=Data{i};
    condn_data{i}=tmp(1:end,:)';
end

% do a bunch of linear SVMs
tic
D_overall=[];
D_overall_1=[];
for iter=1:1
    D=[];svm_model={};D1=[];
    for i=1:length(condn_data)
        %disp(i)
        A = condn_data{i}';
        A = A(257:end,:);
        model_wts=[];
        for j=i:length(condn_data)
            B = condn_data{j}';
            B = B(257:end,:);
            if i==j
                D(i,j)=0;
                svm_model{i,j}=0;
            else
                [res_acc, model,pval,model_runs] = ...
                    svm_linear(A,B,4,0.7);
                svm_dist=[];
                for k=1:length(model_runs)
                    svm_dist_tmp=(model_runs(k).d);
                    svm_class_op=model_runs(k).grp_test;
                    svm_dist(k)=sqrt(sum((svm_dist_tmp.*svm_class_op).^2));
                end
                D(i,j) = mean(res_acc);
                D(j,i) = D(i,j);
                D1(i,j) = mean(svm_dist);
                D1(j,i) = D1(i,j);
                %model_wts = [model_wts;mean(model,1)];
                svm_model{i,j} = mean(model,1);
                svm_model{j,i} = -mean(model,1);
            end
        end
    end
    D_overall(iter,:,:) = D;
    D_overall_1(iter,:,:)=D1;
end
toc

D=squeeze(mean(D_overall_1,1));

figure;imagesc(D)
caxis([.55 1])
xticks(1:size(D,1))
yticks(1:size(D,1))
xticklabels(ImaginedMvmt)
yticklabels(ImaginedMvmt)
set(gcf,'Color','w')
colormap hot

Z = linkage(D,'ward');
figure;dendrogram(Z,0)
x = string(get(gca,'xticklabels'));
x1=[];
for i=1:length(x)
    tmp = str2num(x{i});
    x1 = [x1 ImaginedMvmt(tmp)];
end
xticklabels(x1)
set(gcf,'Color','w')


%
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
%save highDimensional_Analyses_SVM_Distances -v7.3
%save highDimensional_Analyses_ignoringFirst600ms -v7.3
%save highDimensional_Analyses2 -v7.3
save highDimensional_Analyses -v7.3
%
% % plotting svm model
% wt=[];
% for i=1:30
%     tmp  = svm_model{13,i};
%     if length(tmp)>1
%         wt = [wt;tmp];
%     end
% end
% tmp=mean(wt,1);
% chmap=TrialData.Params.ChMap;
% figure;imagesc(abs(tmp(chmap)))
% colormap bone


%
%
% % swap the 18th and 6th columns and rows
% ImaginedMvmt1 = {'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
%     'Rotate Right Wrist','Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp',...
%     'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
%     'Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
%     'Squeeze Both Hands',...
%     'Rotate Left Wrist',...
%     'Imagined Head Movement',...
%     'Right Shoulder Shrug',...
%     'Left Shoulder Shrug',...
%     'Right Tricep','Left Tricep',...
%     'Right Bicep','Left Bicep',...
%     'Right Leg','Left Leg',...
%     'Lips','Tongue'};
% D1=D;
% idx=[1:5 7:17 19:30];
% r1=6;r2=18;
% for i=1:length(idx)
%    % swapping the columns
%    D1(idx(i),r1)=D(idx(i),r2);
%    D1(idx(i),r2)=D(idx(i),r1);
%
%    % swapping the rows
%    D1(r1,idx(i)) = D(r2,idx(i));
%    D1(r2,idx(i)) = D(r1,idx(i));
% end
% D1(r1,r2)=D(r2,r1);
% D1(r2,r1)=D(r1,r2);
% figure;imagesc(D1)
%
%
%
%
% caxis([.55 1])
% xticks(1:size(D1,1))
% yticks(1:size(D1,1))
% xticklabels(ImaginedMvmt1)
% yticklabels(ImaginedMvmt1)
% set(gcf,'Color','w')
% colormap hot


% training a simple linear autoencoder

tmp=Data{1};
X=tmp(257:end,:);

autoenc = trainAutoencoder(X,16,'MaxEpochs',400,...
    'DecoderTransferFunction','satlin');

Xhat = predict(autoenc,X);
chmap = TrialData.Params.ChMap;

i=6;
figure;
subplot(2,1,1)
aa=X(:,6);
imagesc(aa(chmap))
subplot(2,1,2)
aa=Xhat(:,6);
imagesc(aa(chmap))

figure;
subplot(2,1,1)
aa=mean(X,2);
imagesc(aa(chmap))
subplot(2,1,2)
aa=mean(Xhat,2);
imagesc(aa(chmap))


% plot the PC activations
tmp=Data{5};
X=tmp(257:end,:);
[c,s,l]=pca(X');
aa=c(:,2);
figure
imagesc(aa(chmap))


% plot the PC or mean activations
for i=1:length(Data)
    tmp=Data{i};
    X=tmp(257:end,:);
    % artifact correction
    for j=1:size(X,1)
        idx=find(abs(zscore(X(j,:)))>3);
        if ~isempty(idx)
            X(j,idx) = median(X(j,:));
        end
    end
    % [c,s,l]=pca(X');
    %     %X=tmp(129:256,:);
    X1=mean(X,2);
    %X1=c(:,1);
    figure
    imagesc(X1(chmap));
    title(ImaginedMvmt{i})
    axis off
    set(gcf,'Color','w')
    colorbar
    caxis([-0.75 0.75])
end

% cutting outliers from the data
Data1={};
for i=1:length(Data)
    close
    tmp=Data{i};
    X=tmp(257:end,:);
    figure;imagesc(X);
    title(num2str(i))
    [aa bb]=ginput;
    aa=round(aa);
    I=ones(size(X,2),1);
    for j=1:2:length(aa)
        I(aa(j):aa(j+1))=0;
    end
    I=logical(I);
    X=X(:,I);
    Data1{i}=X;
end


% get the mahalanobis distance between the imagined actions after artifact
% correction
D=zeros(length(ImaginedMvmt));
for i=1:length(Data1)
    A = Data1{i}';
    for j=i+1:length(Data1)
        B = Data1{j}';
        d = mahal2(A,B,2);
        D(i,j)=d;
        D(j,i)=d;
    end
end
figure;imagesc(D);
xticks(1:size(D,1))
yticks(1:size(D,1))
xticklabels(ImaginedMvmt)
yticklabels(ImaginedMvmt)
set(gcf,'Color','w')
colormap parula
caxis([0 15])

Z = linkage(D,'complete');
figure;dendrogram(Z,0)
x = string(get(gca,'xticklabels'));
x1=[];
for i=1:length(x)
    tmp = str2num(x{i});
    x1 = [x1 ImaginedMvmt(tmp)];
end
xticklabels(x1)
set(gcf,'Color','w')

chmap = TrialData.Params.ChMap;
% plotting the average maps
for i=1:length(Data1)
    tmp=Data1{i};
    X=tmp(1:end,:);
    %     % artifact correction
    %     for j=1:size(X,1)
    %         idx=find(abs(zscore(X(j,:)))>3);
    %         if ~isempty(idx)
    %             X(j,idx) = median(X(j,:));
    %         end
    %     end
    % [c,s,l]=pca(X');
    %     %X=tmp(129:256,:);
    X1=mean(X,2);
    %X1=c(:,1);
    figure
    imagesc(X1(chmap));
    title(ImaginedMvmt{i})
    axis off
    set(gcf,'Color','w')
    colorbar
    caxis([-0.75 0.75])
end

% averaging all right hand finger movements
rt_fingers=[];
for i=1:5
    tmp=Data1{i};
    X=tmp(1:end,:);
    rt_fingers= [rt_fingers X];
end
X1=mean(rt_fingers,2);
figure
imagesc(X1(chmap));
title('All Right Fingers')
axis off
set(gcf,'Color','w')
colorbar
caxis([-0.75 0.75])

% averaging all left hand finger movements
lt_fingers=[];
for i=29:30
    tmp=Data{i};
    X=tmp(257:end,:);
    lt_fingers= [lt_fingers X];
end
X1=mean(lt_fingers,2);
figure
imagesc(X1(chmap));
title('All Left Fingers')
axis off
set(gcf,'Color','w')
colorbar
caxis([-0.2 0.2])


% averaging all lips and tongue movements
lips_tong=[];
for i=28:28
    tmp=Data{i};
    X=tmp(257:end,:);
    lips_tong= [lips_tong X];
end
X1=mean(lips_tong,2);
figure
imagesc(X1(chmap));
title('Lips and Tongue')
axis off
set(gcf,'Color','w')
colorbar

[c,s,l]=pca(rt_fingers');
X1=c(:,2);
figure;
imagesc(X1(chmap));
title('Rt fingers PC1')

[c,s,l]=pca(lt_fingers');
X1=c(:,2);
figure;
imagesc(X1(chmap));
title('Lt fingers PC1')


[c,s,l]=pca(lips_tong');
X1=c(:,3);
figure;
imagesc(X1(chmap));
title('Lips and Tongue PC1')
axis off
set(gcf,'Color','w')

[c1,s,l]=pca(rt_fingers');
[c2,s,l]=pca(lt_fingers');
[c3,s,l]=pca(lips_tong');
a=subspacea(c1(:,2:10),c2(:,2:10))*180/pi;
b=subspacea(c1(:,2:10),c3(:,2:10))*180/pi;
c=subspacea(c2(:,2:10),c3(:,2:10))*180/pi;
figure;hold on
plot(a)
plot(b)
plot(c)

%% ERPs of imagined actions using smoothed neural data
% and maybe with higher sampling rate


clc;clear

root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
foldernames = {'20211201','20211203','20211206','20211208','20211215','20211217',...
    '20220126'};

cd(root_path)

files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'ImaginedMvmtDAQ')
    D=dir(folderpath);
    if i==3
        D = D([1:3 5:7 9:end]);
    elseif i==4
        D = D([1:3 5:end]);
    elseif i==6
        D = D([1:5 7:end]);
    end
    
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'Imagined');
        tmp=dir(filepath);
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
D8=[];
D9=[];
D10=[];
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.NeuralFeatures;
    features = cell2mat(features);
    %features = features(129:256,:); % delta
    features = features(769:end,:); % hG
    %features = features(513:640,:); %beta
    fs = TrialData.Params.UpdateRate;
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
    tmp_data1=tmp_data;
    
    %     % get first 25 samples via interpolation
    %     tb = (1/fs)*[1:size(tmp_data,2)];
    %     t=(1/fs)*[1:25];
    %     tb = tb*t(end)/tb(end);
    %     tmp_data1 = interp1(tb,tmp_data',t,'spline')';
    %     idx3 = interp1(tb,idx3,t,'spline');
    
    % now stick all the data together
    data = [features(:,[state1 state2]) tmp_data1 features(:,[state4])];
    
    
    % now get the ERPs
    if TrialData.TargetID == 1
        D1 = cat(3,D1,data);
    elseif TrialData.TargetID == 2
        D2 = cat(3,D2,data);
    elseif TrialData.TargetID == 3
        D3 = cat(3,D3,data);
    elseif TrialData.TargetID == 4
        D4 = cat(3,D4,data);
    elseif TrialData.TargetID == 5
        D5 = cat(3,D5,data);
    elseif TrialData.TargetID == 6
        D6 = cat(3,D6,data);
    elseif TrialData.TargetID == 7
        D7 = cat(3,D7,data);
    elseif TrialData.TargetID == 8
        D8 = cat(3,D8,data);
    elseif TrialData.TargetID == 9
        D9 = cat(3,D9,data);
    elseif TrialData.TargetID == 10
        D10 = cat(3,D10,data);
    end
end


%% get all robot3D data
% especially useful for Jensen

clear;clc
filepath ='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
d=dir(filepath);
idx=[];
for j=1:length(d)
    idx=[idx d(j).isdir];
end
d=d(logical(idx));

Trials={};
% store the neural features (96 dimensional)
% store the target location
% store the deocoded outputs
% store the kinematics of the robot
iter=1;
for i=3:length(d)
    foldername=fullfile(d(i).folder,d(i).name);
    d1 = dir(foldername);
    for j=3:length(d1)
        if strcmp(d1(j).name,'Robot')
            filename=fullfile(foldername,d1(j).name);
            files=findfiles('.mat',filename,1)';
            for k=1:length(files)
                if isempty(regexp(files{k},'kf_params'))
                    disp(iter)
                    load(files{k})
                    feat=TrialData.SmoothedNeuralFeatures;
                    idx=find(TrialData.TaskState==3);
                    feat = cell2mat(feat(idx));
                    % pooling
                    neural_feat = bci_pooling(feat,TrialData.Params.ChMap);
                    % getting kin variables
                    kin=TrialData.CursorState;
                    decoded_labels = TrialData.ClickerState;
                    try
                        (TrialData.FilteredClickerState);
                    catch ME
                        TrialData.FilteredClickerState=[];
                    end
                    mode_filter_decoded_labels = TrialData.FilteredClickerState;
                    target_position = TrialData.TargetPosition;
                    % saving
                    Trials(iter).kin=kin;
                    Trials(iter).neural_feat=neural_feat;
                    Trials(iter).target_position=target_position;
                    Trials(iter).mode_filter_decoded_labels=mode_filter_decoded_labels;
                    Trials(iter).decoded_labels=decoded_labels;
                    Trials(iter).Date = files{k}(61:68);
                    Trials(iter).Time = files{k}(76:80);
                    Trials(iter).TrialNum = files{k}(93:100);
                    iter=iter+1;
                end
            end
        end
    end
end

cd(filepath)
Robot3DTrials = Trials;
save Robot3DTrials Robot3DTrials -v7.3

%% LSTMS for the hand task
% extract 800ms snippets with 400ms overlap, train LSTMS
% use the fine tuning approach: fine tune on subset of trials, test on held
% out trials 


clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
foldernames = {'20211201','20211203','20211206','20211208','20211215','20211217',...
'20220126'};
cd(root_path)

imagined_files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'ImaginedMvmtDAQ')
    D=dir(folderpath);
    if i==3
        D = D([1:3 5:7 9:end]);
    elseif i==4
        D = D([1:3 5:end]);
    elseif i==6
        D = D([1:5 7:end]);
    end
    
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'Imagined');
        tmp=dir(filepath);
        imagined_files = [imagined_files;findfiles('',filepath)'];
    end
end


% load the data for the imagined files, if they belong to right thumb,
% index, middle, ring, pinky, pinch, tripod, power
D1i={};
D2i={};
D3i={};
D4i={};
D5i={};
D6i={};
D7i={};
D8i={};
for i=1:length(imagined_files)
    disp(i/length(imagined_files)*100)
    try
        load(imagined_files{i})
        file_loaded = true;
    catch
        file_loaded=false;
        disp(['Could not load ' files{j}]);
    end
    
    if file_loaded
        action = TrialData.ImaginedAction;
        idx = find(TrialData.TaskState==3) ;
        raw_data = cell2mat(TrialData.BroadbandData(idx)');
        idx1 = find(TrialData.TaskState==4) ;
        raw_data4 = cell2mat(TrialData.BroadbandData(idx1)');
        s = size(raw_data,1);
        data_seg={};
        bins =1:400:s;
        raw_data = [raw_data;raw_data4];
        for k=1:length(bins)-1
            tmp = raw_data(bins(k)+[0:799],:);
            data_seg = cat(2,data_seg,tmp);
        end
        
        if strcmp('Right Thumb',action)
            D1i = cat(2,D1i,data_seg);
            %D1f = cat(2,D1f,feat_stats1);
        elseif strcmp('Right Index',action)
            D2i = cat(2,D2i,data_seg);
            %D2f = cat(2,D2f,feat_stats1);
        elseif strcmp('Right Middle',action)
            D3i = cat(2,D3i,data_seg);
            %D3f = cat(2,D3f,feat_stats1);
        elseif strcmp('Right Ring',action)
            D4i = cat(2,D4i,data_seg);
            %D4f = cat(2,D4f,feat_stats1);
        elseif strcmp('Right Pinky',action)
            D5i = cat(2,D5i,data_seg);
            %D5f = cat(2,D5f,feat_stats1);
        elseif strcmp('Right Pinch Grasp',action)
            D6i = cat(2,D6i,data_seg);
            %D6f = cat(2,D6f,feat_stats1);
        elseif strcmp('Right Tripod Grasp',action)
            D7i = cat(2,D7i,data_seg);
            %D7f = cat(2,D7f,feat_stats1);
        elseif strcmp('Right Power Grasp',action)
            D8i = cat(2,D8i,data_seg);
            %D7f = cat(2,D7f,feat_stats1);
        end
    end    
end



% GETTING DATA FROM THE HAND TASK, all but the last day's data 
foldernames = {'20220128','20220204','20220209','20220223'}; 
cd(root_path)

hand_files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Hand')
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        tmp=dir(filepath);
        hand_files = [hand_files;findfiles('',filepath)'];
    end
end


% load the data for the imagined files, if they belong to right thumb,
% index, middle, ring, pinky, pinch, tripod, power
D1={};
D2={};
D3={};
D4={};
D5={};
D6={};
D7={};
D8={};
for i=1:length(hand_files)
    disp(i/length(hand_files)*100)
    try
        load(hand_files{i})
        file_loaded = true;
    catch
        file_loaded=false;
        disp(['Could not load ' hand_files{j}]);
    end
    
    if file_loaded
        action = TrialData.TargetID;
        idx = find(TrialData.TaskState==3) ;
        raw_data = cell2mat(TrialData.BroadbandData(idx)');
        idx1 = find(TrialData.TaskState==4) ;
        raw_data4 = cell2mat(TrialData.BroadbandData(idx1)');
        s = size(raw_data,1);
        data_seg={};
        bins =1:400:s;
        raw_data = [raw_data;raw_data4];
        for k=1:length(bins)-1
            tmp = raw_data(bins(k)+[0:799],:);
            data_seg = cat(2,data_seg,tmp);
        end
        
        if action==1
            D1 = cat(2,D1,data_seg);
            %D1f = cat(2,D1f,feat_stats1);
        elseif action==2
            D2 = cat(2,D2,data_seg);
            %D2f = cat(2,D2f,feat_stats1);
        elseif action==3
            D3 = cat(2,D3,data_seg);
            %D3f = cat(2,D3f,feat_stats1);
        elseif action==4
            D4 = cat(2,D4,data_seg);
            %D4f = cat(2,D4f,feat_stats1);
        elseif action==5
            D5 = cat(2,D5,data_seg);
            %D5f = cat(2,D5f,feat_stats1);
        elseif action==6
            D6 = cat(2,D6,data_seg);
            %D6f = cat(2,D6f,feat_stats1);
        elseif action==7
            D7 = cat(2,D7,data_seg);
            %D7f = cat(2,D7f,feat_stats1);
        elseif action==8
            D8 = cat(2,D8,data_seg);
            %D7f = cat(2,D7f,feat_stats1);
        end
    end    
end


cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
save lstm_hand_data D1 D2 D3 D4 D5 D6 D7 D8 D1i D2i D3i D4i D5i D6i D7i D8i -v7.3






%% BUILDIN THE DECODER USIGN IMAGINED PLUS ONLINE DATA

clear;clc

Y=[];

condn_data_new=[];jj=1;

load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20211001\Robot3DArrow\103931\BCI_Fixed\Data0001.mat')
chmap = TrialData.Params.ChMap;


% low pass filter of raw
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
   'PassbandFrequency',30,'PassbandRipple',0.2, ...
   'SampleRate',1e3);

% log spaced hg filters
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
Params.FilterBank(end+1).fpass = [0.5,4]; % low pass
Params.FilterBank(end+1).fpass = [13,19]; % beta1
Params.FilterBank(end+1).fpass = [19,30]; % beta2

% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end

%hg1
hg1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',77, ...
    'SampleRate',1e3);
hg2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',77,'HalfPowerFrequency2',85, ...
    'SampleRate',1e3);
hg3 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',85,'HalfPowerFrequency2',93, ...
    'SampleRate',1e3);
hg4 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',93,'HalfPowerFrequency2',102, ...
    'SampleRate',1e3);
hg5 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',102,'HalfPowerFrequency2',113, ...
    'SampleRate',1e3);
hg6 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',113,'HalfPowerFrequency2',124, ...
    'SampleRate',1e3);
hg7 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',124,'HalfPowerFrequency2',136, ...
    'SampleRate',1e3);
hg8 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',136,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);




    
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
load('lstm_hand_data','D1','D1i')
condn_data1 = zeros(800,128,length(D1)+length(D1i));
k=1;
for i=1:length(D1)
    disp(k)
    tmp = D1{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data1(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;1];
end
for i=1:length(D1i)
    disp(k)
    tmp = D1i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data1(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;1];
end
clear D1 D1i


for ii=1:size(condn_data1,3)
    disp(ii)
    
    tmp = squeeze(condn_data1(:,:,ii));
    
    % hg filtering
    %tmp_hg = abs(hilbert(filtfilt(hgFilt,tmp)));
    %tmp_hg = filtfilt(lpFilt1,tmp_hg);
    
    
    %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));    
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);
    
 
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800); 
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
        
    % make new data structure 
    %tmp = [tmp_hg tmp_beta tmp_delta]; 
    tmp = [tmp_hg tmp_lp]; 
   
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end

load('lstm_hand_data','D2','D2i')
condn_data2 = zeros(800,128,length(D2)+length(D2i));
k=1;
for i=1:length(D2)
    disp(k)
    tmp = D2{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data2(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;2];
end
for i=1:length(D2i)
    disp(k)
    tmp = D2i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data2(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;2];
end
clear D2 D2i condn_data1


for ii=1:size(condn_data2,3)
    disp(ii)
    
    tmp = squeeze(condn_data2(:,:,ii));
    
    %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));    
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);    
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);    
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure 
    %tmp = [tmp_hg tmp_beta tmp_delta]; 
    tmp = [tmp_hg tmp_lp];    
   
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end





load('lstm_hand_data','D3','D3i')
condn_data3 = zeros(800,128,length(D3)+length(D3i));
k=1;
for i=1:length(D3)
    disp(k)
    tmp = D3{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data3(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;3];
end
for i=1:length(D3i)
    disp(k)
    tmp = D3i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data3(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;3];
end
clear D3 D3i condn_data2


for ii=1:size(condn_data3,3)
    disp(ii)
    
    tmp = squeeze(condn_data3(:,:,ii));
  
    %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));    
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);    
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);    
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure 
    %tmp = [tmp_hg tmp_beta tmp_delta]; 
    tmp = [tmp_hg tmp_lp];    
   
   
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end


load('lstm_hand_data','D4','D4i')
condn_data4 = zeros(800,128,length(D4)+length(D4i));
k=1;
for i=1:length(D4)
    disp(k)
    tmp = D4{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data4(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;4];
end
for i=1:length(D4i)
    disp(k)
    tmp = D4i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data4(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;4];
end
clear D4 D4i condn_data3


for ii=1:size(condn_data4,3)
    disp(ii)
    
    tmp = squeeze(condn_data4(:,:,ii));
   
     %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));    
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);    
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);    
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure 
    %tmp = [tmp_hg tmp_beta tmp_delta]; 
    tmp = [tmp_hg tmp_lp]; 
        
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end


load('lstm_hand_data','D5','D5i')
condn_data5 = zeros(800,128,length(D5)+length(D5i));
k=1;
for i=1:length(D5)
    disp(k)
    tmp = D5{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data5(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;5];
end
for i=1:length(D5i)
    disp(k)
    tmp = D5i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data5(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;5];
end
clear D5 D5i condn_data4


for ii=1:size(condn_data5,3)
    disp(ii)
    
    tmp = squeeze(condn_data5(:,:,ii));
   
    %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));    
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);    
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);    
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure 
    %tmp = [tmp_hg tmp_beta tmp_delta]; 
    tmp = [tmp_hg tmp_lp];    
    
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end



load('lstm_hand_data','D6','D6i')
condn_data6 = zeros(800,128,length(D6)+length(D6i));
k=1;
for i=1:length(D6)
    disp(k)
    tmp = D6{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data6(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;6];
end
for i=1:length(D6i)
    disp(k)
    tmp = D6i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data6(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;6];
end
clear D6 D6i condn_data5


for ii=1:size(condn_data6,3)
    disp(ii)
    
    tmp = squeeze(condn_data6(:,:,ii));

     %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));    
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);    
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);    
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure 
    %tmp = [tmp_hg tmp_beta tmp_delta]; 
    tmp = [tmp_hg tmp_lp]; 
   
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end



load('lstm_hand_data','D7','D7i')
condn_data7 = zeros(800,128,length(D7)+length(D7i));
k=1;
for i=1:length(D7)
    disp(k)
    tmp = D7{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data7(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;7];
end
for i=1:length(D7i)
    disp(k)
    tmp = D7i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data7(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;7];
end
clear D7 D7i condn_data6


for ii=1:size(condn_data7,3)
    disp(ii)
    
    tmp = squeeze(condn_data7(:,:,ii));
    
     %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));    
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);    
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);    
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure 
    %tmp = [tmp_hg tmp_beta tmp_delta]; 
    tmp = [tmp_hg tmp_lp];    
   
   
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end



load('lstm_hand_data','D8','D8i')
condn_data8 = zeros(800,128,length(D8)+length(D8i));
k=1;
for i=1:length(D8)
    disp(k)
    tmp = D8{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data8(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;8];
end
for i=1:length(D8i)
    disp(k)
    tmp = D8i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data8(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;8];
end
clear D8 D8i condn_data7


for ii=1:size(condn_data8,3)
    disp(ii)
    
    tmp = squeeze(condn_data8(:,:,ii));
    
     %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));    
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);    
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);    
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure 
    %tmp = [tmp_hg tmp_beta tmp_delta]; 
    tmp = [tmp_hg tmp_lp];    
   
   
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end


save downsampled_lstm_hand_data condn_data_new Y -v7.3
clear condn_data  condn_data8

% set aside training and testing data in a cell format
idx = randperm(size(condn_data_new,3),round(0.85*size(condn_data_new,3)));
I = zeros(size(condn_data_new,3),1);
I(idx)=1;

XTrain={};
XTest={};
YTrain=[];
YTest=[];
for i=1:size(condn_data_new,3)    
    tmp = squeeze(condn_data_new(:,:,i));
    if I(i)==1
        XTrain = cat(1,XTrain,tmp');
        YTrain = [YTrain Y(i)];
    else
        XTest = cat(1,XTest,tmp');
        YTest = [YTest Y(i)];
    end    
end

% shuffle
idx  = randperm(length(YTrain));
XTrain = XTrain(idx);
YTrain = YTrain(idx);

YTrain = categorical(YTrain');
YTest = categorical(YTest');


% implement label smoothing to see how that does
%save training_data_bilstm_pooled3Feat condn_data_new Y -v7.3

% 
% tmp=str2num(cell2mat(tmp));
% a=0.01;
% tmp1 = (1-a).*tmp + (a)*(1/7);
% clear YTrain
% YTrain = tmp1;
% YTrain =categorical(YTrain);
clear condn_data_new

% specify lstm structure
inputSize = 256;
numHiddenUnits1 = [  90 128 150 200 325];
drop1 = [ 0.2 0.3 0.3  0.4 0.4];
numClasses = 8;
for i=3%1:length(drop1)
    numHiddenUnits=numHiddenUnits1(i);
    drop=drop1(i);
    layers = [ ...
        sequenceInputLayer(inputSize)
        bilstmLayer(numHiddenUnits,'OutputMode','sequence')
        dropoutLayer(drop)        
        gruLayer(numHiddenUnits,'OutputMode','last')
        dropoutLayer(drop)
        batchNormalizationLayer
        fullyConnectedLayer(40)
        reluLayer
        dropoutLayer(.2)
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
    
    
    
    % options
    options = trainingOptions('adam', ...
        'MaxEpochs',15, ...        
        'MiniBatchSize',128, ...
        'GradientThreshold',2, ...
        'Verbose',true, ...
        'ValidationFrequency',30,...
        'Shuffle','every-epoch', ...
        'ValidationData',{XTest,YTest},...
        'ValidationPatience',6,...
        'Plots','training-progress');
    
    % train the model
    net = trainNetwork(XTrain,YTrain,layers,options);
end
% 
% net_800 =net;
% save net_800 net_800

net_bilstm_hand = net;
save net_bilstm_hand net_bilstm_hand


%% TESTING THE DATA ON ONLINE DATA
clc;clear
load net_bilstm_hand
filepath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220218\Hand';

% get the name of the files
files = findfiles('mat',filepath,1)';
files1=[];
for i=1:length(files)
   if regexp(files{i},'Data')
       files1=[files1;files(i)];
   end    
end
files=files1;
clear files1


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
% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end

% low pass filters
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
   'PassbandFrequency',30,'PassbandRipple',0.2, ...
   'SampleRate',1e3);


% load the data, and run it through the classifier
decodes_overall=[];k=1;
for i=1:length(files)
    disp(i)
    
    % load
    load(files{i})
    
    if TrialData.TargetID<=8
        
        % create buffer
        data_buffer = randn(800,128)*0.25;
        
        %get data
        raw_data = TrialData.BroadbandData;
        raw_data1=cell2mat(raw_data');
        
        % state of trial
        state_idx = TrialData.TaskState;
        decodes=[];
        for j=1:length(raw_data)
            tmp = raw_data{j};
            s=size(tmp,1);
            if s<800
                data_buffer = circshift(data_buffer,-s);
                data_buffer(end-s+1:end,:)=tmp;
            else
                data_buffer(1:end,:)=tmp(s-800+1:end,:);
            end
            
            %hg features
            filtered_data=zeros(size(data_buffer,1),size(data_buffer,2),8);
            for ii=1:length(Params.FilterBank)
                filtered_data(:,:,ii) =  ((filter(...
                    Params.FilterBank(ii).b, ...
                    Params.FilterBank(ii).a, ...
                    data_buffer)));
            end
            tmp_hg = squeeze(mean(filtered_data.^2,3));
            tmp_hg = resample(tmp_hg,200,800)*5e2;
            
            
            
            % low-pass features
            tmp_lp = filter(lpFilt,data_buffer);
            tmp_lp = resample(tmp_lp,200,800);
            
            % concatenate
            neural_features = [tmp_hg tmp_lp];
            
            % classifier output
            out=predict(net_bilstm_hand,neural_features');
            [aa bb]=max(out);
            class_predict = bb;
            
            % store results
            if state_idx(j)==3
                decodes=[decodes class_predict];
            end
        end
        decodes_overall(k).decodes = decodes;
        decodes_overall(k).tid = TrialData.TargetID;
        k=k+1;
    end
end

% looking at the accuracy of the bilstm decoder overall
acc=zeros(8,8);
for i=1:length(decodes_overall)
    tmp = decodes_overall(i).decodes;
    tid=decodes_overall(i).tid;
    for j=1:length(tmp)
       acc(tid,tmp(j)) =  acc(tid,tmp(j))+1;
    end
end
for i=1:length(acc)
    acc(i,:) = acc(i,:)/sum(acc(i,:));
end

% looking at accuracy in terms of max decodes
acc_trial=zeros(8,8);
for i=1:length(decodes_overall)
    tmp = decodes_overall(i).decodes;
    tid=decodes_overall(i).tid;
    acc1=zeros(8,8);
    for j=1:length(tmp)
       acc1(tid,tmp(j)) =  acc1(tid,tmp(j))+1;
    end
    acc1=sum(acc1);
    [aa bb]=max(acc1);
    acc_trial(tid,bb)=acc_trial(tid,bb)+1;
end
for i=1:length(acc_trial)
    acc_trial(i,:) = acc_trial(i,:)/sum(acc_trial(i,:));
end


%% LSTM FINE TUNING APPROACH
% update the model weights on 90% of trials and test on held out 10% of
% trials using CV split 


clc;clear
load net_bilstm_hand


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
% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end

% low pass filters
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
   'PassbandFrequency',30,'PassbandRipple',0.2, ...
   'SampleRate',1e3);


% get the trials
filepath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220218\Hand';

% get the name of the files
files = findfiles('mat',filepath,1)';
files1=[];
for i=1:length(files)
   if ~isempty(regexp(files{i},'Data0001')) || ~isempty(regexp(files{i},'Data0002')) ||...
          ~isempty(regexp(files{i},'Data0003')) || ~isempty(regexp(files{i},'Data0004'))||...
          ~isempty(regexp(files{i},'Data0005')) || ~isempty(regexp(files{i},'Data0006'))||...
           ~isempty(regexp(files{i},'Data0007')) || ~isempty(regexp(files{i},'Data0008'))
       files1=[files1;files(i)];
   end    
end
files=files1;
clear files1

% set aside 90% of trials for training and 10% for testing
idx = randperm(length(files),round(0.8*length(files)));
I=zeros(length(files),1);
I(idx)=1;
train_idx = find(I==1);
test_idx = find(I==0);
train_files=files(train_idx);
test_files=files(test_idx);

% get neural features for training trials
D1={};
D2={};
D3={};
D4={};
D5={};
D6={};
D7={};
D8={};
for i=1:length(train_files)
    disp(i/length(train_files)*100)
    try
        load(train_files{i})
        file_loaded = true;
    catch
        file_loaded=false;
        disp(['Could not load ' hand_files{j}]);
    end
    
    if file_loaded
        action = TrialData.TargetID;
        idx = find(TrialData.TaskState==3) ;
        raw_data = cell2mat(TrialData.BroadbandData(idx)');
        idx1 = find(TrialData.TaskState==4) ;
        raw_data4 = cell2mat(TrialData.BroadbandData(idx1)');
        s = size(raw_data,1);
        data_seg={};
        bins =1:400:s;
        raw_data = [raw_data;raw_data4];
        for k=1:length(bins)-1
            tmp = raw_data(bins(k)+[0:799],:);
            data_seg = cat(2,data_seg,tmp);
        end
        
        if action==1
            D1 = cat(2,D1,data_seg);
            %D1f = cat(2,D1f,feat_stats1);
        elseif action==2
            D2 = cat(2,D2,data_seg);
            %D2f = cat(2,D2f,feat_stats1);
        elseif action==3
            D3 = cat(2,D3,data_seg);
            %D3f = cat(2,D3f,feat_stats1);
        elseif action==4
            D4 = cat(2,D4,data_seg);
            %D4f = cat(2,D4f,feat_stats1);
        elseif action==5
            D5 = cat(2,D5,data_seg);
            %D5f = cat(2,D5f,feat_stats1);
        elseif action==6
            D6 = cat(2,D6,data_seg);
            %D6f = cat(2,D6f,feat_stats1);
        elseif action==7
            D7 = cat(2,D7,data_seg);
            %D7f = cat(2,D7f,feat_stats1);
        elseif action==8
            D8 = cat(2,D8,data_seg);
            %D7f = cat(2,D7f,feat_stats1);
        end
    end    
end
D1i={};
D2i={};
D3i={};
D4i={};
D5i={};
D6i={};
D7i={};
D8i={};
Y=[];
condn_data_new=[];jj=1;

condn_data1 = zeros(800,128,length(D1)+length(D1i));
k=1;
for i=1:length(D1)
    disp(k)
    tmp = D1{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data1(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;1];
end
for i=1:length(D1i)
    disp(k)
    tmp = D1i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data1(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;1];
end


for ii=1:size(condn_data1,3)
    disp(ii)
    
    tmp = squeeze(condn_data1(:,:,ii));
    
    % hg filtering
    %tmp_hg = abs(hilbert(filtfilt(hgFilt,tmp)));
    %tmp_hg = filtfilt(lpFilt1,tmp_hg);
    
    
    %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));    
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);
    
 
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800); 
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
        
    % make new data structure 
    %tmp = [tmp_hg tmp_beta tmp_delta]; 
    tmp = [tmp_hg tmp_lp]; 
   
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end

condn_data2 = zeros(800,128,length(D2)+length(D2i));
k=1;
for i=1:length(D2)
    disp(k)
    tmp = D2{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data2(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;2];
end
for i=1:length(D2i)
    disp(k)
    tmp = D2i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data2(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;2];
end


for ii=1:size(condn_data2,3)
    disp(ii)
    
    tmp = squeeze(condn_data2(:,:,ii));
    
    %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));    
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);    
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);    
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure 
    %tmp = [tmp_hg tmp_beta tmp_delta]; 
    tmp = [tmp_hg tmp_lp];    
   
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end

condn_data3 = zeros(800,128,length(D3)+length(D3i));
k=1;
for i=1:length(D3)
    disp(k)
    tmp = D3{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data3(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;3];
end
for i=1:length(D3i)
    disp(k)
    tmp = D3i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data3(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;3];
end


for ii=1:size(condn_data3,3)
    disp(ii)
    
    tmp = squeeze(condn_data3(:,:,ii));
  
    %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));    
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);    
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);    
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure 
    %tmp = [tmp_hg tmp_beta tmp_delta]; 
    tmp = [tmp_hg tmp_lp];    
   
   
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end

condn_data4 = zeros(800,128,length(D4)+length(D4i));
k=1;
for i=1:length(D4)
    disp(k)
    tmp = D4{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data4(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;4];
end
for i=1:length(D4i)
    disp(k)
    tmp = D4i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data4(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;4];
end


for ii=1:size(condn_data4,3)
    disp(ii)
    
    tmp = squeeze(condn_data4(:,:,ii));
   
     %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));    
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);    
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);    
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure 
    %tmp = [tmp_hg tmp_beta tmp_delta]; 
    tmp = [tmp_hg tmp_lp]; 
        
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end

condn_data5 = zeros(800,128,length(D5)+length(D5i));
k=1;
for i=1:length(D5)
    disp(k)
    tmp = D5{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data5(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;5];
end
for i=1:length(D5i)
    disp(k)
    tmp = D5i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data5(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;5];
end

for ii=1:size(condn_data5,3)
    disp(ii)
    
    tmp = squeeze(condn_data5(:,:,ii));
   
    %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));    
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);    
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);    
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure 
    %tmp = [tmp_hg tmp_beta tmp_delta]; 
    tmp = [tmp_hg tmp_lp];    
    
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end


condn_data6 = zeros(800,128,length(D6)+length(D6i));
k=1;
for i=1:length(D6)
    disp(k)
    tmp = D6{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data6(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;6];
end
for i=1:length(D6i)
    disp(k)
    tmp = D6i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data6(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;6];
end

for ii=1:size(condn_data6,3)
    disp(ii)
    
    tmp = squeeze(condn_data6(:,:,ii));

     %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));    
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);    
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);    
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure 
    %tmp = [tmp_hg tmp_beta tmp_delta]; 
    tmp = [tmp_hg tmp_lp]; 
   
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end

condn_data7 = zeros(800,128,length(D7)+length(D7i));
k=1;
for i=1:length(D7)
    disp(k)
    tmp = D7{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data7(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;7];
end
for i=1:length(D7i)
    disp(k)
    tmp = D7i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data7(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;7];
end

for ii=1:size(condn_data7,3)
    disp(ii)
    
    tmp = squeeze(condn_data7(:,:,ii));
    
     %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));    
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);    
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);    
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure 
    %tmp = [tmp_hg tmp_beta tmp_delta]; 
    tmp = [tmp_hg tmp_lp];    
   
   
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end

condn_data8 = zeros(800,128,length(D8)+length(D8i));
k=1;
for i=1:length(D8)
    disp(k)
    tmp = D8{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data8(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;8];
end
for i=1:length(D8i)
    disp(k)
    tmp = D8i{i};
    %tmp1(:,1,:)=tmp;
    tmp1=tmp;
    condn_data8(:,:,k) = tmp1;
    k=k+1;
    Y = [Y ;8];
end

for ii=1:size(condn_data8,3)
    disp(ii)
    
    tmp = squeeze(condn_data8(:,:,ii));
    
     %get hG through filter bank approach
    filtered_data=zeros(size(tmp,1),size(tmp,2),8);
    for i=1:8%length(Params.FilterBank)
        filtered_data(:,:,i) =  ((filter(...
            Params.FilterBank(i).b, ...
            Params.FilterBank(i).a, ...
            tmp)));
    end
    tmp_hg = squeeze(mean(filtered_data.^2,3));    
    
    % LFO low pass filtering
    tmp_lp = filter(lpFilt,tmp);    
    
    % downsample the data
    tmp_lp = resample(tmp_lp,200,800);    
    tmp_hg = resample(tmp_hg,200,800)*5e2;
    
    % spatial pool
    %tmp_lp = spatial_pool(tmp_lp,TrialData);
    %tmp_hg = spatial_pool(tmp_hg,TrialData);
    %tmp_beta = spatial_pool(tmp_beta,TrialData);
    %tmp_delta = spatial_pool(tmp_delta,TrialData);
    
    
    % make new data structure 
    %tmp = [tmp_hg tmp_beta tmp_delta]; 
    tmp = [tmp_hg tmp_lp];    
   
   
    % store
    condn_data_new(:,:,jj) = tmp;
    jj=jj+1;
end

%%%%%% update LSTM weights

% set aside training and testing data in a cell format
idx = randperm(size(condn_data_new,3),round(0.85*size(condn_data_new,3)));
I = zeros(size(condn_data_new,3),1);
I(idx)=1;
XTrain={};
XTest={};
YTrain=[];
YTest=[];
for i=1:size(condn_data_new,3)    
    tmp = squeeze(condn_data_new(:,:,i));
    if I(i)==1
        XTrain = cat(1,XTrain,tmp');
        YTrain = [YTrain Y(i)];
    else
        XTest = cat(1,XTest,tmp');
        YTest = [YTest Y(i)];
    end    
end
% shuffle
idx  = randperm(length(YTrain));
XTrain = XTrain(idx);
YTrain = YTrain(idx);

YTrain = categorical(YTrain');
YTest = categorical(YTest');
    
% options
options = trainingOptions('adam', ...
    'MaxEpochs',20, ...
    'MiniBatchSize',64, ...
    'GradientThreshold',2, ...
    'Verbose',true, ...
    'ValidationFrequency',10,...
    'Shuffle','every-epoch', ...
    'ValidationData',{XTest,YTest},...
    'ValidationPatience',5,...
    'Plots','training-progress');

% train the model
layers = net_bilstm_hand.Layers;
net_bilstm_hand_FT = trainNetwork(XTrain,YTrain,layers,options);



%%%% get neural features for testing and test LSTM

% load the data, and run it through the classifier
decodes_overall=[];k=1;
for i=1:length(test_files)
    disp(i)
    
    % load
    load(test_files{i})
    
    if TrialData.TargetID<=8
        
        % create buffer
        data_buffer = randn(800,128)*0.25;
        
        %get data
        raw_data = TrialData.BroadbandData;
        raw_data1=cell2mat(raw_data');
        
        % state of trial
        state_idx = TrialData.TaskState;
        decodes=[];
        for j=1:length(raw_data)
            tmp = raw_data{j};
            s=size(tmp,1);
            if s<800
                data_buffer = circshift(data_buffer,-s);
                data_buffer(end-s+1:end,:)=tmp;
            else
                data_buffer(1:end,:)=tmp(s-800+1:end,:);
            end
            
            %hg features
            filtered_data=zeros(size(data_buffer,1),size(data_buffer,2),8);
            for ii=1:length(Params.FilterBank)
                filtered_data(:,:,ii) =  ((filter(...
                    Params.FilterBank(ii).b, ...
                    Params.FilterBank(ii).a, ...
                    data_buffer)));
            end
            tmp_hg = squeeze(mean(filtered_data.^2,3));
            tmp_hg = resample(tmp_hg,200,800)*5e2;
            
            
            
            % low-pass features
            tmp_lp = filter(lpFilt,data_buffer);
            tmp_lp = resample(tmp_lp,200,800);
            
            % concatenate
            neural_features = [tmp_hg tmp_lp];
            
            % classifier output
            out=predict(net_bilstm_hand_FT,neural_features');
            [aa bb]=max(out);
            class_predict = bb;
            
            % store results
            if state_idx(j)==3
                decodes=[decodes class_predict];
            end
        end
        decodes_overall(k).decodes = decodes;
        decodes_overall(k).tid = TrialData.TargetID;
        k=k+1;
    end
end

% looking at the accuracy of the bilstm decoder overall
acc=zeros(8,8);
for i=1:length(decodes_overall)
    tmp = decodes_overall(i).decodes;
    tid=decodes_overall(i).tid;
    for j=1:length(tmp)
       acc(tid,tmp(j)) =  acc(tid,tmp(j))+1;
    end
end
for i=1:length(acc)
    acc(i,:) = acc(i,:)/sum(acc(i,:));
end

% looking at accuracy in terms of max decodes
acc_trial=zeros(8,8);
for i=1:length(decodes_overall)
    tmp = decodes_overall(i).decodes;
    tid=decodes_overall(i).tid;
    acc1=zeros(8,8);
    for j=1:length(tmp)
       acc1(tid,tmp(j)) =  acc1(tid,tmp(j))+1;
    end
    acc1=sum(acc1);
    [aa bb]=max(acc1);
    acc_trial(tid,bb)=acc_trial(tid,bb)+1;
end
for i=1:length(acc_trial)
    acc_trial(i,:) = acc_trial(i,:)/sum(acc_trial(i,:));
end




%% BUILDING MLP FOR THE HAND MODEL 

clc;clear
close all

root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';

% GETTING DATA FROM THE HAND TASK, all but the last day's data 
foldernames = {'20220128','20220204','20220209','20220218','20220223'}; 
cd(root_path)

hand_files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Hand')
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        tmp=dir(filepath);
        hand_files = [hand_files;findfiles('',filepath)'];
    end
end


% load the data for the imagined files, if they belong to right thumb,
% index, middle, ring, pinky, pinch, tripod, power
D1=[];%thumb
D2=[];%index
D3=[];%middle
D4=[];%ring
D5=[];%pinky
D6=[];%power
D7=[];%pinch
D8=[];%tripod
D9=[];%wrist out
D10=[];%wrist in



for i=1:length(hand_files)
    disp(i/length(hand_files)*100)
    load(hand_files{i})
    
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
    elseif TrialData.TargetID == 10
        D10 = [D10 temp];
    end
end


idx = [1:96];
condn_data{1}=[D1(idx,:) ]';
condn_data{2}= [D2(idx,:)]';
condn_data{3}=[D3(idx,:)]';
condn_data{4}=[D4(idx,:)]';
condn_data{5}=[D5(idx,:)]';
condn_data{6}=[D6(idx,:)]';
condn_data{7}=[D9(idx,:)]';
condn_data{8}=[D10(idx,:)]';

A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};
E = condn_data{5};
F = condn_data{6};
G = condn_data{7};
H = condn_data{8};
% I = condn_data{9};
% J = condn_data{10};


clear N
N = [A' B' C' D' E' F' G' H' ];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1);8*ones(size(H,1),1)];

T = zeros(size(T1,1),8);
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



% code to train a neural network
clear net_hand_mlp
net_hand_mlp = patternnet([64 64 64]) ;
net_hand_mlp.performParam.regularization=0.2;
net_hand_mlp = train(net_hand_mlp,N,T','UseParallel','yes');
genFunction(net_hand_mlp,'MLP_Hand')
save net_hand_mlp net_hand_mlp

% using custom layers
layers = [ ...
    featureInputLayer(96)
    fullyConnectedLayer(64)
    layerNormalizationLayer
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(64)
    layerNormalizationLayer
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(64)
    layerNormalizationLayer
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(8)
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
    'MaxEpochs',25, ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'MiniBatchSize',256,...
    'ValidationFrequency',20,...
    'ValidationPatience',5,...
    'ExecutionEnvironment','GPU',...
    'ValidationData',{Xtest',Ytest});

% build the classifier
net_mlp_hand = trainNetwork(Xtrain',Ytrain,layers,options);
net_mlp_hand_adam_64=net_mlp_hand;
save net_mlp_hand_adam_64 net_mlp_hand_adam_64








