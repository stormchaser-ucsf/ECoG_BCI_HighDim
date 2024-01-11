%% EVERYTHING HERE IS FOR B3

%%
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
        temp = temp(:,5:end);

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
        %temp = temp(:,1:20);
        temp = temp(:,5:end);

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

acc=squeeze(nanmedian(acc_overall,1));
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
ylim([0 1])
set(gcf,'Color','w')



%% Mahalanobis distance of imagined actions


clc;clear

root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
foldernames = {'20211201','20211203','20211206','20211208','20211215','20211217',...
    '20220126','20220223'};
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB')

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


    %hg and delta and beta
    temp = temp([129:256 513:640 769:end],:);

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




%% MAHALANOBIS AFTER NORMAIZING DATA WITHIN A SESSION (MAIN)



clc;clear
addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'
addpath('C:\Users\nikic\Documents\MATLAB')
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')

root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
foldernames = {'20230111','20230118','20230119','20230125','20230126','20230201','20230203'};
cd(root_path)


files=[]; % these are the foldernanes within each day
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'ImaginedMvmtDAQ')
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'Imagined');
        tmp=dir(filepath);
        files = [files;string(filepath)];
    end
end

ImaginedMvmt = {'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
    'Rotate Right Wrist','Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp',...
    'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
    'Rotate Left Wrist','Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
    'Squeeze Both Hands',...
    'Imagined Head Movement',...
    'Right Shoulder Shrug',...
    'Left Shoulder Shrug',...
    'Right Tricep','Left Tricep',...
    'Right Bicep','Left Bicep',...
    'Right Knee','Left Knee',...
    'Right Ankle','Left Ankle',...
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
        %disp(d(jj).name)
        file_loaded=true;
        try
            load(d(jj).name)
        catch
            file_loaded=false;
            %disp(['file not loaded  ' d(jj).name])
        end
        if file_loaded
            features  = TrialData.SmoothedNeuralFeatures;
            kinax = TrialData.TaskState;
            kinax2 = find(kinax==2);
            kinax = find(kinax==3);
            temp = cell2mat(features(kinax));
            temp2 = cell2mat(features(kinax2));
            %temp=temp(:,3:end); % ignore the first 600ms

            % baseline the data to state 2
            %m = mean(temp2,2);
            %s = std(temp2')';
            %temp = (temp-m)./s;

            % take from 400 to 2000ms
            temp = temp(:,3:end);
            %temp=temp(:,3:end); % ignore the first 600ms

            if size(temp,1)==1792

                % hg and delta and beta
                temp = temp([257:512 1025:1280 1537:1792],:);


                % remove the bad channels 108, 113 118
                bad_ch = [108 113 118];
                good_ch = ones(size(temp,1),1);
                for ii=1:length(bad_ch)
                    bad_ch_tmp = bad_ch(ii)*[1 2 3];
                    good_ch(bad_ch_tmp)=0;
                end

                temp = temp(logical(good_ch),:);
                %                 figure;hist(temp(:))
                %                 title(d(jj).name)

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


                if max(abs(temp(:))) < 10
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
            end
        end
    end

    m=mean(data_overall,1);
    s=std(data_overall,1);

    for j=1:length(Data_tmp)
        tmp=Data_tmp{j};
        if ~isempty(tmp)
            tmp = (tmp-m')./s';
            Data_tmp{j}=tmp;

            % transfer to main file
            tmp=Data{j};
            tmp = [tmp Data_tmp{j}];
            Data{j} = tmp;
        end
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
%caxis([20 200])

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
        A = A(1:end,:);
        model_wts=[];
        for j=i:length(condn_data)
            B = condn_data{j}';
            B = B(1:end,:);
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

D=squeeze(mean(D_overall,1));

figure;imagesc(D)
caxis([.96 1])
xticks(1:size(D,1))
yticks(1:size(D,1))
xticklabels(ImaginedMvmt)
yticklabels(ImaginedMvmt)
set(gcf,'Color','w')
colormap hot

Z = linkage(D,'ward');
figure;H=dendrogram(Z,0);
for i=1:length(H)
    H(i).LineWidth=1;
end
x = string(get(gca,'xticklabels'));
x1=[];
for i=1:length(x)
    tmp = str2num(x{i});
    x1 = [x1 ImaginedMvmt(tmp)];
end
xticklabels(x1)
set(gcf,'Color','w')
set(gca,'LineWidth',1)
ylabel('Distances')


%
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3')
%save highDimensional_Analyses_SVM_Distances -v7.3
%save highDimensional_Analyses_ignoringFirst600ms -v7.3
%save highDimensional_Analyses2 -v7.3
save highDimensional_Analyses -v7.3  % MAIN RESULTS
%
% plotting svm model
wt=[];
for i=1:30
    tmp  = svm_model{13,i};
    if length(tmp)>1
        wt = [wt;tmp];
    end
end
tmp=mean(wt,1);
chmap=TrialData.Params.ChMap;
figure;imagesc(abs(tmp(chmap)))
colormap bone




% swap the 18th and 6th columns and rows
ImaginedMvmt1 = {'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
    'Rotate Right Wrist','Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp',...
    'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
    'Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
    'Squeeze Both Hands',...
    'Rotate Left Wrist',...
    'Imagined Head Movement',...
    'Right Shoulder Shrug',...
    'Left Shoulder Shrug',...
    'Right Tricep','Left Tricep',...
    'Right Bicep','Left Bicep',...
    'Right Leg','Left Leg',...
    'Lips','Tongue'};
D1=D;
idx=[1:5 7:17 19:30];
r1=6;r2=18;
for i=1:length(idx)
    % swapping the columns
    D1(idx(i),r1)=D(idx(i),r2);
    D1(idx(i),r2)=D(idx(i),r1);

    % swapping the rows
    D1(r1,idx(i)) = D(r2,idx(i));
    D1(r2,idx(i)) = D(r1,idx(i));
end
D1(r1,r2)=D(r2,r1);
D1(r2,r1)=D(r1,r2);
figure;imagesc(D1)




caxis([.55 1])
xticks(1:size(D1,1))
yticks(1:size(D1,1))
xticklabels(ImaginedMvmt1)
yticklabels(ImaginedMvmt1)
set(gcf,'Color','w')
colormap hot


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
for i=1:9
    tmp=Data{i};
    X=tmp(257:end,:);
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
idx=[10:12 16:18];
for i=1:length(idx)
    tmp=Data{idx(i)};
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


% averaging all left ring pinky wrist
left_ring_pinky_wrist=[];
for i=13:15
    tmp=Data{i};
    X=tmp(257:end,:);
    left_ring_pinky_wrist= [left_ring_pinky_wrist X];
end
X1=mean(left_ring_pinky_wrist,2);
figure
imagesc(X1(chmap));
title('Lips and Tongue')
axis off
set(gcf,'Color','w')
colorbar


% right leg and left leg
rt_lt_leg=[];
for i=27:28
    tmp=Data{i};
    X=tmp(257:end,:);
    rt_lt_leg= [rt_lt_leg X];
end
X1=mean(rt_lt_leg,2);
figure
imagesc(X1(chmap));
title('Rt and lt leg')
axis off
set(gcf,'Color','w')
colorbar



% averaging all lips and tongue
tong_lips=[];
for i=29:30
    tmp=Data{i};
    X=tmp(257:end,:);
    tong_lips= [tong_lips X];
end
X1=mean(tong_lips,2);
figure
imagesc(X1(chmap));
title('Lips and Tongue')
axis off
set(gcf,'Color','w')
colorbar



[c,s,l]=pca(rt_lt_leg');
X1=c(:,2);
figure;
imagesc(X1(chmap));
title('Right and Left Leg')
set(gcf,'Color','w')
axis off


[c,s,l]=pca(tong_lips');
X1=c(:,2);
figure;
imagesc(X1(chmap));
title('Tongue and Lips')
set(gcf,'Color','w')
axis off

[c,s,l]=pca(left_ring_pinky_wrist');
X1=c(:,2);
figure;
imagesc(X1(chmap));
title('Left Ring Pinky Wrist')
axis off
set(gcf,'Color','w')





[c,s,l]=pca(rt_fingers');
X1=c(:,2);
figure;
imagesc(X1(chmap));
title('Right hand ')
axis off
set(gcf,'Color','w')
caxis([-.2 .2])

[c,s,l]=pca(lt_fingers');
X1=c(:,3);
figure;
imagesc(X1(chmap));
caxis([-0.15 0.15])
title('Left Hand')
set(gcf,'Color','w')
axis off

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


% plotting



%% ERPs of imagined actions higher sampling rate (MAIN)
% using hG and LMP, beta etc.

clc;clear
addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'
addpath('C:\Users\nikic\Documents\MATLAB')
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')

root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
foldernames = {'20230111','20230118','20230119','20230125','20230126','20230201','20230203'};
cd(root_path)




files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'ImaginedMvmtDAQ')
    D=dir(folderpath);

    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'Imagined');
        tmp=dir(filepath);
        files = [files;findfiles('',filepath)'];
    end
end

ImaginedMvmt = {'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
    'Rotate Right Wrist','Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp',...
    'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
    'Rotate Left Wrist','Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
    'Squeeze Both Hands',...
    'Imagined Head Movement',...
    'Right Shoulder Shrug',...
    'Left Shoulder Shrug',...
    'Right Tricep','Left Tricep',...
    'Right Bicep','Left Bicep',...
    'Right Knee','Left Knee',...
    'Right Ankle','Left Ankle',...
    'Lips','Tongue'};



% load the ERP data for each target
ERP_Data={};
for i=1:length(ImaginedMvmt)
    ERP_Data{i}=[];
end

% TIMING INFORMATION FOR THE TRIALS
Params.InterTrialInterval = 2; % rest period between trials
Params.InstructedDelayTime = 2; % text appears telling subject which action to imagine
Params.CueTime = 2; % A red square; subject has to get ready
Params.ImaginedMvmtTime = 4; % A green square, subject has actively imagine the action

% low pass filter of raw
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',5,'PassbandRipple',0.2, ...
    'SampleRate',1e3);


%
% % log spaced hg filters
% Params.Fs = 1000;
% Params.FilterBank(1).fpass = [70,77];   % high gamma1
% Params.FilterBank(end+1).fpass = [77,85];   % high gamma2
% Params.FilterBank(end+1).fpass = [85,93];   % high gamma3
% Params.FilterBank(end+1).fpass = [93,102];  % high gamma4
% Params.FilterBank(end+1).fpass = [102,113]; % high gamma5
% Params.FilterBank(end+1).fpass = [113,124]; % high gamma6
% Params.FilterBank(end+1).fpass = [124,136]; % high gamma7
% Params.FilterBank(end+1).fpass = [136,150]; % high gamma8
% Params.FilterBank(end+1).fpass = [0.5,4]; % delta
% Params.FilterBank(end+1).fpass = [13,19]; % beta1
% Params.FilterBank(end+1).fpass = [19,30]; % beta2
%
% % compute filter coefficients
% for i=1:length(Params.FilterBank),
%     [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
%     Params.FilterBank(i).b = b;
%     Params.FilterBank(i).a = a;
% end

for i=1:length(files)
    disp(i/length(files)*100)
    file_loaded=1;
    try
        load(files{i});
    catch
        file_loaded=0;
    end
    if file_loaded
        features  = TrialData.BroadbandData;
        features = cell2mat(features');
        Params = TrialData.Params;


        % artifact correction is there is too much noise in raw signal
        if sum(abs(features(:))>5)/length(features(:))*100 < 5




            %get hG through filter bank approach
            filtered_data=zeros(size(features,1),size(features,2),2);
            k=1;
            for ii=9:16 %9:16 is hG, 4:5 is beta
                filtered_data(:,:,k) =  abs(hilbert(filtfilt(...
                    Params.FilterBank(ii).b, ...
                    Params.FilterBank(ii).a, ...
                    features)));
                k=k+1;
            end
            %tmp_hg = squeeze(mean(filtered_data.^2,3));
            tmp_hg = squeeze(mean(filtered_data,3));

            % low pass filter the data
            %     features1 = [randn(4000,128);features;randn(4000,128)];
            %     tmp_hg = abs(hilbert(filtfilt(lpFilt,features1)));
            %     tmp_hg = tmp_hg(4001:end-4000,:);

            task_state = TrialData.TaskState;
            idx=[];
            for ii=1:length(task_state)
                tmp = TrialData.BroadbandData{ii};
                idx = [idx;task_state(ii)*ones(size(tmp,1),1)];
            end


            % z-score to 1s before the get ready symbol
            fidx2 =  find(idx==2);fidx2=fidx2(1);
            fidx2 = fidx2+[-1000:0];
            m = mean(tmp_hg(fidx2,:));
            s = std(tmp_hg(fidx2,:));
            fidx = [fidx2 fidx2(end)+[1:7000]];

            tmp_hg_epoch  = tmp_hg(fidx,:);
            tmp_hg_epoch = (tmp_hg_epoch-m)./s;

            % downsample to 200Hz
            %tmp_lp = resample(tmp_lp,200,800);
            %     tmp_hg_epoch1=[];
            %     for j=1:size(tmp_hg_epoch,2);
            %         tmp_hg_epoch1(:,j) = decimate(tmp_hg_epoch(:,j),5);
            %     end

            features = tmp_hg_epoch;
            var_exists=true;
            try
                tmp_action_name = TrialData.ImaginedAction;
            catch
                var_exists=false;
            end

            if var_exists && size(features,2)==256
                for j=1:length(ImaginedMvmt)
                    if strcmp(ImaginedMvmt{j},TrialData.ImaginedAction)
                        tmp = ERP_Data{j};
                        tmp = cat(3,tmp,features);
                        ERP_Data{j}=tmp;
                        break
                    end
                end
            end
        end
    end
end

%save high_res_erp_beta_imagined_data -v7.3
%save high_res_erp_LMP_imagined_data -v7.3
save B3_high_res_erp_imagined_data -v7.3

% get the number of epochs used
ep=[];
for i=1:length(ERP_Data)
    tmp = ERP_Data{i};
    ep(i) = size(tmp,3);
end
figure;stem(ep)
xticks(1:32)
xticklabels(ImaginedMvmt)

% plot ERPs and see individual channels, with stats
data = ERP_Data{21};
figure;
for ch=1:size(data,2)

    clf
    set(gcf,'Color','w')
    set(gcf,'WindowState','maximized')
    title(num2str(ch))

    chdata = squeeze((data(:,ch,:)));

    % bad trial removal
    tmp_bad=zscore(chdata')';
    artifact_check=((tmp_bad)>=5) + (tmp_bad<=-4);
    good_idx = find(sum(artifact_check)==0);
    chdata = chdata(:,good_idx);


    % confidence intervals around the mean
    m=mean(chdata,2);
    opt=statset('UseParallel',true);
    mb = sort(bootstrp(1000,@mean,chdata','Options',opt));
    tt=linspace(-1,7,size(data,1));
    %figure;
    [fillhandle,msg]=jbfill(tt,(mb(25,:)),(mb(975,:))...
        ,[0.5 0.5 0.5],[0.5 0.5 0.5],1,.4);
    hold on
    plot(tt,(m),'k','LineWidth',1)
    axis tight
    %plot(tt,mb(25,:),'--k','LineWidth',.25)
    %plot(tt,mb(975,:),'--k','LineWidth',.25)
    % beautify
    ylabel(num2str(ch))
    ylim([-1.5 2])
    %yticks ''
    %xticks ''
    vline([0 2 6],'r')
    hline(0)
    axis tight

    % channel significance: if mean is outside the 95% boostrapped C.I. for
    % any duration of time
    tmp = mb(:,1:1000);
    tmp = tmp(:);
    pval=[];sign_time=[];
    for j=3001:6000
        if m(j)>0
            ptest = (sum(m(j) >= tmp(:)))/length(tmp);
            sign_time=[sign_time 1];
        else
            ptest = (sum(m(j) <= tmp(:)))/length(tmp);
            sign_time=[sign_time -1];
        end
        ptest = 1-ptest;
        pval = [pval ptest];

    end
    [pfdr, pval1]=fdr(pval,0.05);pfdr;
    %pfdr=0.0005;
    pval(pval<=pfdr) = 1;
    pval(pval~=1)=0;
    m1=m(3001:6000);
    tt1=tt(3001:6000);
    idx1=find(pval==1);
    plot(tt1(idx1),m1(idx1),'b')

    %sum(pval/3000)


    %
    if sum(pval)>300 % how many sig. samples do you want for it to be 'sig'
        if sum(pval.*sign_time)>0
            box_col = 'g';
        else
            box_col = 'b';
        end
        box on
        set(gca,'LineWidth',2)
        set(gca,'XColor',box_col)
        set(gca,'YColor',box_col)
    end

    waitforbuttonpress



end




% plot ERPs at all channels with tests for significance
idx = [9,10,20,21,27,29,31,32 ];
bad_ch = [108,113,118];
load('ECOG_Grid_8596_000063_B3.mat')
chMap=ecog_grid;
sig_ch=zeros(32,256);
for i=1:length(idx)
    figure
    ha=tight_subplot(size(chMap,1),size(chMap,2));
    d = 1;
    set(gcf,'Color','w')
    set(gcf,'WindowState','maximized')
    data = ERP_Data{idx(i)};
    for ch=1:size(data,2)

        if sum(ch==bad_ch) == 0

            disp(['movement ' num2str(i) ' channel ' num2str(ch)])

            [x y] = find(chMap==ch);
            if x == 1
                axes(ha(y));
                %subplot(8, 16, y)
            else
                s = 23*(x-1) + y;
                axes(ha(s));
                %subplot(8, 16, s)
            end
            hold on

            chdata = squeeze((data(:,ch,:)));

            % bad trial removal
            tmp_bad=zscore(chdata')';
            artifact_check=((tmp_bad)>=5) + (tmp_bad<=-4);
            good_idx = find(sum(artifact_check)==0);
            chdata = chdata(:,good_idx);


            % confidence intervals around the mean
            m=mean(chdata,2);
            opt=statset('UseParallel',true);
            mb = sort(bootstrp(1000,@mean,chdata','Options',opt));
            tt=linspace(-1,7,size(data,1));
            %figure;
            [fillhandle,msg]=jbfill(tt,(mb(25,:)),(mb(975,:))...
                ,[0.5 0.5 0.5],[0.5 0.5 0.5],1,.4);
            hold on
            plot(tt,(m),'k','LineWidth',1)
            axis tight
            %plot(tt,mb(25,:),'--k','LineWidth',.25)
            %plot(tt,mb(975,:),'--k','LineWidth',.25)
            % beautify
            ylabel(num2str(ch))
            ylim([-1.5 2])
            yticks ''
            xticks ''
            vline([0 2 6],'r')
            hline(0)
            %hline([0.5, -0.5])
            axis tight


            % channel significance: if mean is outside the 95% boostrapped C.I. for
            % any duration of time
            tmp = mb(:,1:1000);
            tmp = tmp(:);
            pval=[];sign_time=[];
            for j=3001:6000
                if m(j)>0
                    ptest = (sum(m(j) >= tmp(:)))/length(tmp);
                    sign_time=[sign_time 1];
                else
                    ptest = (sum(m(j) <= tmp(:)))/length(tmp);
                    sign_time=[sign_time -1];
                end
                ptest = 1-ptest;
                pval = [pval ptest];

            end
            [pfdr, pval1]=fdr(pval,0.05);pfdr;
            pfdr=0.0005;
            pval(pval<=pfdr) = 1;
            pval(pval~=1)=0;
            m1=m(3001:6000);
            tt1=tt(3001:6000);
            idx1=find(pval==1);
            plot(tt1(idx1),m1(idx1),'b')

            %sum(pval/3000)


            %
            if sum(pval)>300
                if sum(pval.*sign_time)>0
                    box_col = 'g';
                    sig_ch(i,ch)=1;
                else
                    box_col = 'b';
                    sig_ch(i,ch)=-1;
                end
                box on
                set(gca,'LineWidth',2)
                set(gca,'XColor',box_col)
                set(gca,'YColor',box_col)
            end
        end
    end
    sgtitle(ImaginedMvmt(idx(i)))
    %     filename = fullfile('F:\DATA\ecog data\ECoG BCI\Results\ERPs Imagined Actions\delta',ImaginedMvmt{idx(i)});
    %     saveas(gcf,filename)
    %     set(gcf,'PaperPositionMode','auto')
    %     print(gcf,filename,'-dpng','-r500')
end
%save ERPs_sig_ch_beta -v7.3
save ERPs_sig_ch_LMP -v7.3
%save ERPs_sig_ch_hg -v7.3

% plotting sig channels on by one
for i=1:length(ImaginedMvmt)
    tmp = sig_ch(i,:);
    figure;imagesc(abs(tmp(chMap)));
    title(ImaginedMvmt{i})
end

%plot brain plots of sig. ch for hg in tongue, rt bicep, rt thumb, lt
%thumb,head and beta for left leg
imaging_B1;close all
idx =  [1,10,30,25,20,27,28,21];
for i=1:length(idx)
    tmp = sig_ch(idx(i),:);
    figure;imagesc(tmp(chMap))
    sgtitle(ImaginedMvmt{idx(i)})
    plot_sig_channels_binary_B1(tmp,cortex,elecmatrix,chMap);
    sgtitle(ImaginedMvmt{idx(i)})
end

figure
c_h = ctmr_gauss_plot(cortex,elecmatrix(ch,:),...
    Wa1(1:286),'lh');
temp=Wa1;
temp=temp./(max(abs(temp)));
chI = find(temp~=0);
for j=1:length(chI)
    ms = abs(temp(chI(j))) * (12-2) + 2;
    e_h = el_add(elecmatrix(chI(j),:), 'color', 'b','msize',abs(ms));
end
set(gcf,'Color','w')
set(gca,'FontSize',20)
view(-94,30)


%%%% ROI SPECIFIC ERPS FOR SELECTION ACTIONS %%%%%%
%idx = [3,8,10,19,20,25,28,30];
%idx = [1,10,20,25,30]; % have to play around with this
%idx= [ 20 29 30]; % face
%idx = [1 3  19];%hand
%idx= [ 25 28]% limbs
idx=[ 10 12 16 ];% left hand
%cmap = turbo(length(idx));
cmap = brewermap(length(idx),'Set1')
s1 = [106 97 103 100 116 115];
m1 = [25 31 3 9 27];
rol = [42 15 59 36 32 63];
pmd = [79 67 34 91 90 51];
pmv= [53 2 13 ];
channels = {s1,m1,rol,pmd,pmv};
% plotting
channels=1:128;
for i=1:length(channels)
    figure;
    hold on
    ch = channels(i);
    % first plot the ERPs
    for j=1:length(idx)
        tmp_erp = ERP_Data{idx(j)};
        tmp = tmp_erp(:,ch,:);
        tmp = squeeze(mean(tmp,2))';
        tmp=detrend(tmp')';
        tmp = tmp+0.1;
        %         if j==4
        %             tmp = tmp-0.4;
        %         end
        m = smooth(mean(tmp,1),300);
        %mb = sort(bootstrp(1000,@mean,tmp)); % bootstrap
        s = std(tmp,1)/sqrt(size(tmp,1)); % standard error
        s = smooth(s',300);
        tt=(0:size(tmp,2)-1) + 3000;
        plot(tt,m,'Color',cmap(j,:),'LineWidth',2)
        %  [fillhandle,msg]=jbfill(tt,(m-s)',(m+s)'...
        %      ,cmap(j,:),cmap(j,:),1,.2);
        hold on
    end
    set(gcf,'Color','w')
    % now plot the C.I.
    for j=1:length(idx)
        tmp_erp = ERP_Data{idx(j)};
        tmp = tmp_erp(:,ch,:);
        tmp = squeeze(mean(tmp,2))';
        tmp=detrend(tmp')';
        tmp = tmp+0.1;
        %         if j==4
        %             tmp = tmp-0.4;
        %         end
        m = smooth(mean(tmp,1),300);
        %mb = sort(bootstrp(1000,@mean,tmp)); % bootstrap
        s = std(tmp,1)/sqrt(size(tmp,1)); % standard error
        s = smooth(s',300);
        tt=(0:size(tmp,2)-1) + 3000;
        %plot(tt,m,'Color',cmap(j,:),'LineWidth',2)
        [fillhandle,msg]=jbfill(tt,(m-s)',(m+s)'...
            ,cmap(j,:),cmap(j,:),1,.1);
        hold on
    end
    axis tight
    legend(ImaginedMvmt(idx))
    vline([1000 3000 7000]+3000,'r')
    hline(0,'k')
    xlim([2000 7200]+3000)
    title(num2str(i))
    xticks(4000:2000:11000)
    %yticks(-.6:.4:1.4)
    %ylim([-.6 1.3])
    %waitforbuttonpress
    %close
end

%good channels for face- 40, 121, 19,126,109,30
% good channes for hand - 116, 115, 97, 51, 46, 31, 26, 25
% good channles for limbs - 126, 124, 116, 103, 29 (-0.2 mag)

% plot chMAP of the specific channels
imaging_B1;
grid_ecog = [];
for i=1:16:128
    grid_ecog =[grid_ecog; i:i+15];
end
grid_ecog=flipud(grid_ecog);
figure;
ch=51;
[x y] = find(chMap==ch);
ch1 = grid_ecog(x,y);
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
e_h = el_add(elecmatrix(1:end,:),'color','w','msize',2);
e_h = el_add(elecmatrix(ch1,:),'color','r','msize',10);
set(gcf,'Color','w')
view(-99,21)


%%%%% ROI specific activity in the various regions
chmap=TrialData.Params.ChMap;
%hand_elec=[97 100 115 116 103 106 25 31 26 3 9 27];
%hand_elec=[97 100 115 116 103 106 25 31 26 3 9 27];
%hand_elec =[22	5	20 111	122	117 105	120	107]; %pmv
%hand_elec=[99	101	121	127	105	120];%pmd
hand_elec = [49	64	58	59
    54	39	47	42
    18	1	8	15];% pmv
% get the average activity for each imagined action with C.I.
roi_mean=[];
roi_dist_mean=[];
task_state = TrialData.TaskState;
%idx = [ find(task_state==3)];
idx=3001:6000;
%idx=idx(5:15);
for i=1:length(ImaginedMvmt)
    disp(i)
    data = ERP_Data{i};
    data = data(idx,hand_elec,:);
    data = squeeze(mean(data,1)); % time
    %data = squeeze(mean(data,1)); % channels
    data = data(:);
    roi_mean(i) = mean(data);
    %     if i==19
    %         roi_mean(i)=0.9;
    %     end
    roi_dist_mean(:,i) = sort(bootstrp(1000,@mean,data));
end
figure;bar(roi_mean)

y = roi_mean;
y=y';
errY(:,1) = roi_dist_mean(500,:)-roi_dist_mean(25,:);
errY(:,2) = roi_dist_mean(975,:)-roi_dist_mean(500,:);
figure;
barwitherr(errY, y);
xticks(1:30)
set(gcf,'Color','w')
set(gca,'FontSize',16)
set(gca,'LineWidth',1)
xticklabels(ImaginedMvmt)

subplot(2,1,2) % have to run the above code again to get roi specific activity
barwitherr(errY, y);
xticks(1:30)
set(gcf,'Color','w')
set(gca,'FontSize',16)
set(gca,'LineWidth',1)
xticklabels(ImaginedMvmt)


% plotting spatial map comparing right and left hand movements in sig.
% channels (1:9, 10:18)
rt_channels = sum(abs(sig_ch(1:9,:)));
lt_channels = sum(abs(sig_ch(10:18,:)));
figure;imagesc(rt_channels(chMap));caxis([0 8])
figure;imagesc(lt_channels(chMap));caxis([0 8])

figure;stem(abs(rt_channels))
hold on
stem(abs(lt_channels))

% PmD test
pmd=[65	85	83	68
    37	56	48	43
    53	55	52	35
    2	10	21	30];pmd=pmd(:);
lt_pmd = (abs(sig_ch(10:18,pmd)));
rt_pmd = (abs(sig_ch(1:9,pmd)));
[sum(lt_pmd(:)) sum(rt_pmd(:))]


% cosine distance between cortical network
D=zeros(30);
for i=1:size(sig_ch,1)
    A=sig_ch(i,:);
    for j=i+1:size(sig_ch,1)
        B=sig_ch(j,:);
        D(i,j) = pdist([(A);(B)],'jaccard');
        D(j,i) = D(i,j);
    end
end
% plotting hierarhical similarity
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

%%% building a MLP classifier based on average activity across channels
% 2.3s to 3.3s after go cue, averaged over that one second window
res_overall=[];
condn_data=[];  % features by samples
Y=[];
%using average activity plus a smoothing parameter
for i=1:length(ERP_Data)
    tmp = ERP_Data{i};
    % heavily smooth each channel's single trial activity
    for j=1:size(tmp,2)
        for k=1:size(tmp,3)
            tmp(:,j,k) = smooth(tmp(:,j,k),500);
        end
    end
    m = squeeze(mean(tmp(3000:4500,:,:),1));
    s = squeeze(std(tmp(3000:4500,:,:),1));
    condn_data=cat(2,condn_data,[m;s]);
    Y = [Y;i*ones(size(m,2)*1,1)];
end

for iter=1:20
    disp(iter)

    N=condn_data;

    % partition into training and testing
    idx = randperm(size(condn_data,2),round(0.8*size(condn_data,2)));
    YTrain = Y(idx);
    NTrain = N(:,idx);
    I = ones(size(condn_data,2),1);
    I(idx)=0;
    YTest = Y(logical(I));
    NTest = N(:,logical(I));

    T1=YTrain;
    T = zeros(size(T1,1),30);
    for i=1:30
        [aa bb]=find(T1==i);
        %T(aa(1):aa(end),i)=1;
        T(aa,i)=1;
    end

    clear net
    net = patternnet([64]) ;
    net.performParam.regularization=0.2;
    net.divideParam.trainRatio = 0.85;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.0;
    net = train(net,NTrain,T','UseGPU','yes');

    % test on held out data
    out = net(NTest);
    D=zeros(30);
    for i=1:length(YTest)
        [aa bb]=max(out(:,i));
        D(YTest(i),bb)=D(YTest(i),bb)+1;
    end
    for i=1:size(D,1)
        D(i,:)= D(i,:)/sum(D(i,:));
    end
    %figure;stem(diag(D))
    %xticks(1:30)
    %xticklabels(ImaginedMvmt)
    res_overall(iter,:)=diag(D);
    res_overall_map(iter,:,:) = D;
end
figure;stem(mean(res_overall,1))
xticks(1:30)
xticklabels(ImaginedMvmt)
hline(1/30)
figure;
imagesc(squeeze(mean(res_overall_map,1)))

% use a classifier based on

%%%% dPCA analyses on the imagined movement data.. ROI X Time X Mvmt-Type
%avg activity: N x C x L x T
% Channels X Conditions X Laterality X Time
M1 =[27	9	3
    26	31	25
    116	103	106
    115	100	97];M1=M1(:);
pmd=[94	91	79
    73	90	67
    61	51	34
    40	46	41];pmd=pmd(:);
pmv=[96	84	76	95
    92	65	85	83
    62	37	56	48
    45	53	55	52];pmv=pmv(:);
m1_ant=[19	2	10	21
    24	13	6	4
    124	126	128	119
    102	109	99	101];m1_ant=m1_ant(:);
central=[33	49	64	58
    50	54	39	47
    28	18	1	8
    5	20	14	11];central=central(:);

%condn_idx = [1 23  3  24 ];
condn_idx = [1 25  10  26 ];

Data_left=[];Data_right=[];
for i=1:length(condn_idx)
    tmp = ERP_Data{condn_idx(i)};
    tmp=squeeze(mean(tmp,3));
    %tmp = tmp(:,M1);
    if i>2
        Data_left=cat(3,Data_left,tmp');
    else
        Data_right=cat(3,Data_right,tmp');
    end
end
clear Data
Data(:,:,1,:) = permute(Data_right,[1 3 2]);
Data(:,:,2,:) = permute(Data_left,[1 3 2]);
firingRatesAverage=Data;

% plot all right/left finger ERPs
idx = [10:18];
ch=106;
figure;
hold on
col = parula(length(idx));
for i=1:length(idx)
    data = ERP_Data{idx(i)};
    chdata = squeeze((data(:,ch,:)));
    m=mean(chdata,2);
    tt=linspace(-1,7,size(data,1));
    plot(tt,m,'LineWidth',1,'Color',col(i,:));
end
vline([0 2 6],'r')
set(gcf,'Color','w')
set(gca,'FontSize',12)
legend(ImaginedMvmt(idx))


% sig channel from the ERP analyses
% head
tmp = sig_ch(27:28,:);
tmp=mean(abs(tmp),1);
tmp(tmp~=0)=1;
figure;imagesc(abs(tmp(chMap)))
set(gcf,'Color','w')
axis off
set(gcf,'Color','w')
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh')
e_h = el_add(elecmatrix(find(tmp~=0),:),'color','b')
set(gcf,'Color','w')

% getting a map of significant channels based on an ANOVA model. % Rt
% single digit, Lt single digit, Head, Lips/Tongue, Rt Proximal (Bi/Tri),
% Lt Proximal (Bi/Tri), Distal (Lt leg)


% get the data
% right sindle digit
rt_idx = [10:18];rt_data=[];
for i=1:length(rt_idx)
    rt_data(i,:,:,:) = ERP_Data{rt_idx(i)};
end
rt_data = squeeze(mean(rt_data,1));
% left sindle digit
lt_idx = [10:18];lt_data=[];
for i=1:length(lt_idx)
    lt_data(i,:,:,:) = ERP_Data{lt_idx(i)};
end
lt_data = squeeze(mean(lt_data,1));
% head
head = ERP_Data{20};
% lips tong
lips_tong=[];
lips_tong = cat(4,lips_tong,ERP_Data{29},ERP_Data{30},ERP_Data{20});
lips_tong = permute(lips_tong,[4 1 2 3]);
lips_tong = squeeze(mean(lips_tong,4));
% rt prximal
rt_prox=[];
rt_prox = cat(4,rt_prox,ERP_Data{21},ERP_Data{22},...
    ERP_Data{23},ERP_Data{24},ERP_Data{25},ERP_Data{26});
rt_prox = permute(rt_prox,[4 1 2 3]);
rt_prox = squeeze(mean(rt_prox,4));
% lt proximal
lt_prox=[];
lt_prox = cat(4,lt_prox,ERP_Data{24},ERP_Data{24});
lt_prox = squeeze(mean(lt_prox,4));
% legs
legs=[];
legs = cat(4,legs,ERP_Data{27},ERP_Data{28});
legs = permute(legs,[4 1 2 3]);
legs = squeeze(mean(legs,4));

% run the test at each channel, avg activity 3200-5000 time-pts
anova_sig_ch_pval=[];
time_idx=3200:5000;
for i=1:128
    anova_data = [squeeze(mean(rt_data(time_idx,i,:)))...
        squeeze(mean(lt_data(time_idx,i,:)))...
        squeeze(mean(head(time_idx,i,:)))...
        squeeze(mean(lips_tong(time_idx,i,:)))...
        squeeze(mean(rt_prox(time_idx,i,:)))...
        squeeze(mean(lt_prox(time_idx,i,:)))...
        squeeze(mean(legs(time_idx,i,:)))];
    [p,tbl,stats]=anova1(anova_data,[],'off');
    anova_sig_ch_pval(i)=p;
end
sum(anova_sig_ch_pval<=0.05)

% anova on movements within a body part set e.g., all rt hand movements on
% a sample by sample basis
anova_sig_ch_pval=[];
time_idx=4000:5500;
for i=1:128
    disp(i)
    parfor time_idx=3000:5000
        anova_data = squeeze(mean(rt_data(:,time_idx,i,:),2))';
        tmp=zscore(anova_data);
        bad_idx = abs(tmp)>2.0;
        anova_data(logical(bad_idx))=NaN;
        [p,tbl,stats]=anova1(anova_data,[],'off');
        anova_sig_ch_pval(i,time_idx)=p;
    end
end
anova_sig_ch_pval=anova_sig_ch_pval(:,3000:end);
[pfdr,pvals]=fdr(anova_sig_ch_pval(:),0.05);
sum(anova_sig_ch_pval(:)<=0.05)
%sum(anova_sig_ch_pval(:)<=pfdr)
%tmp = anova_sig_ch_pval<=pfdr;
tmp = anova_sig_ch_pval<=1e-4;
figure;imagesc(tmp)
tmp1=sum(tmp');
tmp1(tmp1>0)=1;
figure;stem(tmp1)
figure;imagesc(tmp1(chMap))

% anova on movements within a body part set e.g., all rt hand movements on
% averaged over time
anova_sig_ch_pval=[];
time_idx=3200:5000;
for i=1:128
    anova_data = squeeze(mean(rt_data(:,time_idx,i,:),2))';
    [p,tbl,stats]=anova1(anova_data,[],'off');
    anova_sig_ch_pval(i)=p;
end
[pfdr,pvals]=fdr(anova_sig_ch_pval(:),0.05);
sum(anova_sig_ch_pval(:)<=0.05)
tmp = anova_sig_ch_pval<=0.05;
figure;imagesc(tmp(chMap))
axis off
set(gcf,'Color','w')

% same as above but with bad trial removal
anova_sig_ch_pval=[];
time_idx=3200:5000;
for i=1:128
    anova_data = squeeze(mean(rt_data(:,time_idx,i,:),2))';
    tmp=zscore(anova_data);
    bad_idx = abs(tmp)>2.0;
    anova_data(logical(bad_idx))=NaN;
    [p,tbl,stats]=anova1(anova_data,[],'off');
    anova_sig_ch_pval(i)=p;
end
[pfdr,pvals]=fdr(anova_sig_ch_pval(:),0.01);
sum(anova_sig_ch_pval(:)<=0.01)
tmp = anova_sig_ch_pval<=0.05;
figure;imagesc(tmp(chMap))
axis off
set(gcf,'Color','w')



% delta -> covers rt shoulder, lt shoulder and leg but also in same hand
% knob regions
% now look at Mahab distance using this new data from 3000-6000 samples
% mean for each trial, std across trials
D=[];
for i=1:length(ERP_Data)
    A=ERP_Data{i};
    disp(i)
    for j=i:length(ERP_Data)
        B=ERP_Data{j};
        if i==j
            D(i,j)=0;
        else
            a=squeeze(mean(A(3000:6000,:,:),1))';
            % artifact correction
            m = zscore(mean(a,2));
            idx = find(abs(m)>3);
            I=ones(size(a,1),1);
            I(idx)=0;
            a=a(logical(I),:);

            b=squeeze(mean(B(3000:6000,:,:),1))';
            % artifact correction
            m = zscore(mean(b,2));
            idx = find(abs(m)>3);
            I2=ones(size(b,1),1);
            I2(idx)=0;
            b=b(logical(I),:);

            a=A(3500:5000,:,logical(I));
            clear a1 b1
            for ii=1:size(a,3)
                a1(:,:,ii) = resample(a(:,:,ii),1,5);
            end
            a=a1;
            a=permute(a,[2 1 3]);
            a=a(:,:)';

            b=B(3500:5000,:,logical(I2));
            for ii=1:size(b,3)
                b1(:,:,ii) = resample(b(:,:,ii),1,5);
            end
            b=b1;
            b=permute(b,[2 1 3]);
            b=b(:,:)';

            D(i,j) = mahal2(a,b,2);
            D(j,i) = D(i,j);
        end
    end
end
figure;imagesc((D))



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

% mahab distance after smoothing hG activity



% saving dpng images for all erps in various bands
filepath='F:\DATA\ecog data\ECoG BCI\Results\ERPs Imagined Actions\beta';
D=dir(filepath);
for i=3:length(D)
    disp(i-2)
    filename = fullfile(D(i).folder,D(i).name);
    openfig(filename);
    set(gcf,'WindowState','maximized')
    filename_to_save = fullfile(D(i).folder,D(i).name(1:end-4));
    set(gcf,'PaperPositionMode','auto')
    print(gcf,filename_to_save,'-dpng','-r500')
    close all
end


a = [0.27 0.35 0.37 0.45];
stat = mean(a);
a = a-mean(a)+0.25;
boot=[];
for i=1:10000
    idx = randi(length(a),length(a),1);
    boot(i) = mean(a(idx));
end
figure;hist(boot,6)
vline(stat)
sum(boot>stat)/length(boot)


%% ERPs of imagined actions
% and maybe with higher sampling rate
% and looking at average activation within each ROI

clc;clear
addpath 'C:\Users\nikic\OneDrive\Documents\GitHub\ECoG_BCI_HighDim\helpers'

root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
foldernames = {'20211201','20211203','20211206','20211208','20211215','20211217',...
    '20220126','20220223','20220225'};

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
    'Rotate Right Wrist','Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp',...
    'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
    'Rotate Left Wrist','Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
    'Squeeze Both Hands',...
    'Imagined Head Movement',...
    'Right Shoulder Shrug',...
    'Left Shoulder Shrug',...
    'Right Tricep','Left Tricep',...
    'Right Bicep','Left Bicep',...
    'Right Leg','Left Leg',...
    'Lips','Tongue'};



% load the ERP data for each target
ERP_Data={};
for i=1:length(ImaginedMvmt)
    ERP_Data{i}=[];
end

for i=1:length(files)
    disp(i/length(files)*100)
    load(files{i});
    features  = TrialData.NeuralFeatures;
    features = cell2mat(features);
    %features = features(129:256,:); % delta
    features = features(769:end,:); % hG
    %features = features(513:640,:); %beta

    if size(features,2)>60
        features = features(:,1:60);
    end

    for j=1:length(ImaginedMvmt)
        if strcmp(ImaginedMvmt{j},TrialData.ImaginedAction)
            tmp = ERP_Data{j};
            tmp = cat(3,tmp,features);
            ERP_Data{j}=tmp;
            break
        end
    end
end

%save ERP_data_30DOF ERP_Data ImaginedMvmt task_state -v7.3

% now plotting ERPs
load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220225\ImaginedMvmtDAQ\113824\Imagined\Data0030.mat')
task_state = TrialData.TaskState;
chmap = TrialData.Params.ChMap;
ch=106;
data = ERP_Data{1};
ch_data = squeeze(data(ch,:,:));
figure;plot(ch_data')
figure;plot(mean(ch_data,2))
idx1=find(task_state==1);
idx2=find(task_state==2);
idx3=find(task_state==3);
idx4=find(task_state==4);
vline([idx1(1) idx2(1) idx3(1) idx4(1)])


% plot with boostrapped confidnce intervals in the form of a grid
% key point here is to get rid of bad trials
figure
ha=tight_subplot(8,16);
d = 1;
set(gcf,'Color','w')
for ch =1:size(data,1)
    [x y] = find(chmap==ch);
    if x == 1
        axes(ha(y));
        %subplot(8, 16, y)
    else
        s = 16*(x-1) + y;
        axes(ha(s));
        %subplot(8, 16, s)
    end
    ch_data = squeeze(data(ch,:,:));
    ch_data = zscore(ch_data);

    plot_with_shading(ch_data',task_state);
end


% artifact correction steps
% have to get rid of high-frequency artifacts
data1=[];
for i=1:size(data,1)
    data1(i,:,:) = zscore(squeeze(data(i,:,:)));
end
data1=data1(:,:);
[c,s,l]=pca(data1');
figure;plot(s(:,1))
data1 = s(:,2:end)*c(:,2:end)';

% ROI specific activity in the various regions
chmap=TrialData.Params.ChMap;
%hand_elec=[97 100 115 116 103 106 25 31 26 3 9 27];
hand_elec=[97 100 115 116 103 106 25 31 26 3 9 27];
%hand_elec =[22	5	20 111	122	117 105	120	107]; %pmv
%hand_elec=[99	101	121	127	105	120];%pmd
% get the average activity for each imagined action with C.I.
roi_mean=[];
roi_dist_mean=[];
task_state = TrialData.TaskState;
idx = [ find(task_state==3)];
%idx=idx(5:15);
for i=1:length(ImaginedMvmt)
    disp(i)
    data = ERP_Data{i};
    data = data(hand_elec,idx,:);
    %data = squeeze(mean(data,1)); % channels
    %data = squeeze(mean(data,1)); % time
    data = data(:);
    roi_mean(i) = mean(data);
    roi_dist_mean(:,i) = sort(bootstrp(1000,@mean,data));
end
figure;bar(roi_mean)

y = roi_mean;
y=y';
errY(:,1) = roi_dist_mean(500,:)-roi_dist_mean(25,:);
errY(:,2) = roi_dist_mean(975,:)-roi_dist_mean(500,:);
figure;
barwitherr(errY, y);
xticks(1:30)
xticklabels(ImaginedMvmt)
set(gcf,'Color','w')
title('Hand Knob activation')

% plotting the avergaed activity on the brain
data = ERP_Data{28};
idx=find(task_state==3);
data=squeeze(mean(data,3));
data = mean(data(:,idx),2);
chmap=TrialData.Params.ChMap;
figure;imagesc(data(chmap))
%caxis([-2 2])
caxis([-0.4 0.2])

% plotting hand knob channels on brian
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],...
    0,'lh');set(gcf,'Color','w')
hand_elec_brain=[3	4	5	6	7	8];
e_h1 = el_add(elecmatrix(:,:), 'color', [1 1 1],'msize',2);
e_h1 = el_add(elecmatrix((hand_elec_brain),:), 'color', [1 0 0],'msize',6);

%%%%% BRAIN PLOTTING
% if positive, yellow, if negative blue
col = parula;
col = col([1 end],:);
% scale the size of the electrode to be reflective of the magnitude
imaging_B1;close all
temp= (data);
temp = temp./max(abs(temp));
pos = data>0;
neg = data<0;
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],...
    0,'lh');set(gcf,'Color','w')
origMap = [1:16;17:32;33:48;49:64;65:80;81:96;97:112;113:128];
origMap = flipud(origMap);
e_h1 = el_add(elecmatrix(:,:), 'color', [1 1 1],'msize',2);
for j=1:length(temp)
    [x y]=find(origMap==j);
    ch=chmap(x,y);
    ms = abs(temp(ch)) * (10-3)+2;
    if pos(ch) == 1
        c = col(2,:);
    elseif neg(ch)==1
        c = col(1,:);
    end
    e_h = el_add(elecmatrix(j,:), 'color',c,'msize',ms);
end
view(-101,24)

% a few example ERPs contrasting all the right hand actions against each
% other in an example channel
f1 = ERP_Data{1};
f2 = ERP_Data{1};
f3 = ERP_Data{1};
f4 = ERP_Data{1};
f5 = ERP_Data{1};

ch=106;
figure;hold on
set(gcf,'Color','w')
cmap={'r','g','b','y','m'};
task_state=TrialData.TaskState;
for i=1:3
    data = ERP_Data{i};
    data=squeeze(data(ch,:,:));
    % baseline each trial to the first 5 bins
    for j=1:size(data,2)
        data(:,j) = data(:,j)-mean(data(13:25,j));
    end
    data=data';
    m=mean(data,1);
    mb = sort(bootstrp(1000,@mean,data));
    tt=(1/5)*(0:size(data,2)-1);
    %tt=tt-0.6;
    [fillhandle,msg]=jbfill(tt,mb(25,:),mb(975,:)...
        ,cmap{i},cmap{i},1,.2);
    hold on
    plot(tt,m,'Color',cmap{i},'LineWidth',1)
end
h=vline([3]);
h.LineWidth=1.5;
h.Color='r';
h=vline([5]);
h.LineWidth=1.5;
h.Color='g';
hline(0,'k')
xlim([2 10])



%% PLOTTING DIFFERENCES IN POOLED VS UNPOOLED SMOOTHED DATA
% PLOTTING ERPS ON SINGLE CHANNEL AS COMPARED TO POOLED DATA

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
foldernames = {'20210915'};
cd(root_path)

files=[];
for i=length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=3:length(D)
        folderpath,D(j).name
        filepath=fullfile(folderpath,D(j).name,'BCI_Fixed');
        files = [files;findfiles('',filepath)'];
    end
end


% load the data for each target
D1=[];
D1pool=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
D7pool=[];
time_to_target=zeros(2,7);
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.NeuralFeatures;
    features = cell2mat(features);
    features = features(769:end,:);

    features1  = TrialData.SmoothedNeuralFeatures;
    features1 = cell2mat(features1);
    features1 = features1(769:end,:);



    %features = features(513:640,:);
    fs = TrialData.Params.UpdateRate;
    kinax = TrialData.TaskState;
    state1 = find(kinax==1);
    state2 = find(kinax==2);
    state3 = find(kinax==3);
    state4 = find(kinax==4);
    tmp_data = features(:,state3);
    tmp_data_p = features1(:,state3);
    idx1= ones(length(state1),1);
    idx2= 2*ones(length(state2),1);
    idx3= 3*ones(length(state3),1);
    idx4= 4*ones(length(state4),1);

    % interpolate
    tb = (1/fs)*[1:size(tmp_data,2)];
    t=(1/fs)*[1:10];
    tb = tb*t(end)/tb(end);
    tmp_data1 = interp1(tb,tmp_data',t,'spline')';
    tmp_data_p = interp1(tb,tmp_data_p',t,'spline')';
    idx3 = interp1(tb,idx3,t,'spline');

    % now stick all the data together
    trial_dur = (length(state3)-TrialData.Params.ClickCounter)*(1/fs);
    data = [features(:,[state1 state2]) tmp_data1 features(:,[state4])];
    data_p = [features1(:,[state1 state2]) tmp_data_p features1(:,[state4])];

    % correction
    if size(data,2)<40
        len = size(data,2);
        lend = 40-len;
        data = [data(:,1:lend) data];
        data_p = [data_p(:,1:lend) data_p];
    end
    %
    %     % store the time to target data
    %     time_to_target(2,TrialData.TargetID) = time_to_target(2,TrialData.TargetID)+1;
    %     if trial_dur<=2
    %         time_to_target(1,TrialData.TargetID) = time_to_target(1,TrialData.TargetID)+1;
    %     end

    % now get the ERPs
    if TrialData.TargetID == TrialData.SelectedTargetID && trial_dur<=10
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
        end
    end



    % do it now for pooled data
    temp=data_p;
    new_temp=[];
    [xx yy] = size(TrialData.Params.ChMap);
    for k=1:size(temp,2)
        tmp1 = temp(1:128,k);tmp1 = tmp1(TrialData.Params.ChMap);
        %tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
        %tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
        pooled_data=[];
        for ii=1:2:xx
            for j=1:2:yy
                delta = (tmp1(ii:ii+1,j:j+1));delta=mean(delta(:)); % this is actualy hg
                % beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                % hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                %pooled_data = [pooled_data; delta; beta ;hg];
                pooled_data = [pooled_data; delta; ];
            end
        end
        new_temp= [new_temp pooled_data];
    end
    temp_data=new_temp;
    data = temp_data;

    if TrialData.TargetID == TrialData.SelectedTargetID && trial_dur<=10
        if TrialData.TargetID == 1
            D1pool = cat(3,D1pool,data);
        end
    end

    if TrialData.TargetID == TrialData.SelectedTargetID && trial_dur<=10
        if TrialData.TargetID == 7
            D7pool = cat(3,D7pool,data);
        end
    end

end

tmp = squeeze(D1pool(29,:,:));
figure;plot(tmp,'Color',[.5 .5 .5])
hold on
plot(mean(tmp,2),'b','LineWidth',1)

tmp = squeeze(D1(3,:,:));
figure;plot(tmp,'Color',[.5 .5 .5])
hold on
plot(mean(tmp,2),'b','LineWidth',1)


% plotting a comparison of pooled and unpooled data
tmp = D1;
% first smooth each trial
for i=1:size(tmp,3)
    for j=1:size(tmp,1)
        tmp(j,:,i) = smooth(squeeze(tmp(j,:,i)),5);
    end
end
% now spatially pool
chmap=TrialData.Params.ChMap;
ch = [100 97 103 106];
tmp_p = tmp(ch,:,:);
tmp_p = squeeze(mean(tmp_p,1));

figure;
tt = (1/5)*[0:size(D1,2)-1];
plot(tt,squeeze(D1(97,:,:)),'Color',[.5 .5 .5 .5],'LineWidth',1)
hold on
plot(tt,squeeze(mean(D1(97,:,:),3)),'Color','b','LineWidth',1)
tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);
timm = tim*(1/5);
hline(0,'--r')
axis tight
h=vline(timm(1),'r');
h1=vline(timm(2),'g');
%set(h,'Color','k')
set(gcf,'Color','w')
set(gca,'FontSize',14)
set(gca,'LineWidth',1)
xlabel('Time (s)')
ylabel('z-score')
title('Ch Data')
%xticks ''
%xlabel ''

figure;
tt = (1/5)*[0:size(D1,2)-1];
plot(tt,tmp_p,'Color',[.5 .5 .5 .5],'LineWidth',1)
hold on
plot(tt,mean(tmp_p,2),'Color','b','LineWidth',1)
tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);
timm = tim*(1/5);
hline(0,'--r')
axis tight
h=vline(timm(1),'r');
h1=vline(timm(2),'g');
%set(h,'Color','k')
set(gcf,'Color','w')
set(gca,'FontSize',14)
set(gca,'LineWidth',1)
xlabel('Time (s)')
ylabel('z-score')
title('Ch pooled Data')
%xticks ''
%xlabel ''


% plot the variances across channels for single trials
var_raw=[];
var_pool=[];
for i=1:size(D1,3)
    tmp_raw =  squeeze(D1(:,:,i));
    var_raw = [var_raw var(tmp_raw(:))];

    tmp_pool =  squeeze(D1pool(:,:,i));
    var_pool = [var_pool var(tmp_pool(:))];
end

figure;
boxplot([var_raw' var_pool'])
xticklabels({'Raw','Pooled'})
set(gca,'FontSize',12)
set(gcf,'Color','w')
ylabel('Variance')

% plot an image of hG activity in the hand knob region
tmp = squeeze(mean(D1(:,15,12:16),3));
figure;
imagesc(tmp(chmap))
axis off
set(gcf,'Color','w')

% plot ERPs single trial for target 1 for example
chMap=TrialData.Params.ChMap;
figure
ha=tight_subplot(8,16);
d = 1;
set(gcf,'Color','w')
tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);
for i = 1:size(D1,1)
    [x y] = find(chMap==d);
    if x == 1
        axes(ha(y));
        %subplot(8, 16, y)
    else
        s = 16*(x-1) + y;
        axes(ha(s));
        %subplot(8, 16, s)
    end
    hold on
    erps =  squeeze(D1(i,:,:));
    plot(erps, 'color', [0.4 0.4 0.4 ]);
    ylabel (num2str(i))
    axis tight
    ylim([-2 2])
    plot(mean(erps,2),'r','LineWidth',1.5)
    %set(gca,'LineWidth',1)
    %vline([time(2:4)])
    h=vline(tim);
    %set(h,'LineWidth',1)
    set(h,'Color','b')
    h=hline(0);
    set(h,'LineWidth',1.5)
    if i~=102
        yticklabels ''
        xticklabels ''
    else
        %xticks([tim])
        %xticklabels({'S1','S2','S3','S4'})
    end
    d = d+1;
end



% plot the ERPs with bootstrapped C.I. shading
chMap=TrialData.Params.ChMap;
figure
ha=tight_subplot(8,16);
d = 1;
set(gcf,'Color','w')
tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);
tim(3)=25;
for i = 1:size(D1,1)
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
    erps =  squeeze(D1(i,:,:));

    chdata = erps;
    % zscore the data to the first 5 time-bins
    tmp_data=chdata(1:length(idx1),:);
    m = mean(tmp_data(:));
    s = std(tmp_data(:));
    chdata = (chdata -m)./s;




    % get the confidence intervals and plot
    m = mean(chdata,2);
    mb = sort(bootstrp(1000,@mean,chdata'));
    tt=1:size(D1,2);

    % plotting confidence intervals
    %[fillhandle,msg]=jbfill(tt,mb(25,:),mb(975,:)...
    %    ,[0.3 0.3 0.7],[0.3 0.3 0.7],1,.2);

    % plotting the raw ERPs
    plot(chdata,'Color',[.5 .5 .5 .5],'LineWidth',1)
    hold on
    plot(m,'b','LineWidth',1)
    %plot(mb(25,:),'--b')
    %plot(mb(975,:),'--b')
    %hline(0)





    % shuffle the data for null confidence intervals
    tmp_mean=[];
    for j=1:1000
        %tmp = circshift(chdata,randperm(size(chdata,1),1));
        tmp = chdata;
        tmp(randperm(numel(chdata))) = tmp;
        tmp_data=tmp(1:8,:);
        m = mean(tmp_data(:));
        s = std(tmp_data(:));
        tmp = (tmp -m)./s;
        tmp_mean(j,:) = mean(tmp,2);
    end

    % plot null interval
    tmp_mean = sort(tmp_mean);
    %plot(tmp_mean(25,:),'--r')
    %plot(tmp_mean(975,:),'--r')
    [fillhandle,msg]=jbfill(tt,tmp_mean(25,:),tmp_mean(975,:)...
        ,[0.7 0.3 0.3],[0.7 0.3 0.3],1,.2);


    % statistical test
    % if the mean is outside confidence intervals in state 3
    m = mean(chdata,2);
    idx=13:25;
    mstat = m((idx));
    pval=[];
    for j=1:length(idx)
        pval(j) = (sum(abs(mstat(j)) >= abs(tmp_mean(:,idx(j)))))./(size(tmp_mean,1));
    end

    res=sum((1-pval)<=0.01);
    if res>=9
        suc=1;
    else
        suc=0;
    end

    % beautify
    ylabel (num2str(i))
    axis tight
    ylim([-3 6])
    %     if max(chdata(:))>2
    %         ylim([-2 max(chdata(:))]);
    %     end
    %set(gca,'LineWidth',1)
    %vline([time(2:4)])
    h=vline(tim(1:2));
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


% plot individual channel
erps =  squeeze(D1(25,:,:));
chdata = erps;
% zscore the data to the first 5 time-bins
tmp_data=chdata(1:length(idx1),:);
m = mean(tmp_data(:));
s = std(tmp_data(:));
chdata = (chdata -m)./s;
% now plot
figure;hold on
tt = (1/5)*[0:size(chdata,1)-1];
timm = tim*(1/5);
timm = timm/max(tt);
tt = tt./max(tt);
plot(tt,chdata,'Color',[.5 .5 .5 ],'LineWidth',1);
plot(tt,mean(chdata,2),'LineWidth',2,'Color','b')
hline(0)
axis tight
h=vline(timm(1:2));
set(h,'Color','k')
set(gcf,'Color','w')
set(gca,'FontSize',14)
set(gca,'LineWidth',1)
xlabel('Normalized trial length (percent)')
ylabel('z-score')
title('Channel 25 Day 2')
xticks ''
xlabel ''





%% ERPs for B2 from a session of imagined data

clc;clear
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')

filepath ='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2';
cd(filepath)
folders = {'20210324'};

% imagined ERPs
files=[];
for i=1:length(folders)
    full_path = fullfile(filepath,folders{i},'CenterOut');
    tmp = findfiles('.mat',full_path,1)';
    for j=1:length(tmp)
        if ~isempty(regexp(tmp{j},'Data'))
            files=[files;tmp(j)];
        end
    end
end

[D1,D2,D3,D4] = load_erp_data_B2_imag(files);

% plot ERPs with all the imagined data
load('ECOG_Grid_8596-002131.mat')


% plot the ERPs with bootstrapped C.I. shading
chMap=ecog_grid;
figure
ha=tight_subplot(8,16);
d = 1;
set(gcf,'Color','w')
for i = 1:size(D1,1)
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
    erps =  squeeze(D1(i,:,:));

    chdata = erps;
    % zscore the data to the first 10 time-bins
    tmp_data=chdata(1:10,:);
    m = mean(tmp_data(:));
    s = std(tmp_data(:));
    chdata = (chdata -m)./s;

    % get the confidence intervals
    m = mean(chdata,2);
    mb = sort(bootstrp(1000,@mean,chdata'));
    tt=1:size(D2,2);
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
        tmp_data=tmp(1:10,:);
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
    idx=10:40;
    mstat = m((idx));
    pval=[];
    for j=1:length(idx)
        pval(j) = (sum(abs(mstat(j)) >= abs(tmp_mean(:,idx(j)))))./(size(tmp_mean,1));
    end

    res=sum((1-pval)<=0.05);
    if res>=15
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
    h=vline(5);
    %set(h,'LineWidth',1)
    set(h,'Color','k')
    h=hline(0);
    set(h,'LineWidth',1.5)
    if i~=107
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

%% ERPs for B2 from sessions of online data

clc;clear
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')

filepath ='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2';
cd(filepath)
folders = {'20210331'};%

% get the file names
files=[];
for i=1:length(folders)
    full_path = fullfile(filepath,folders{i},'DiscreteArrow');
    tmp = findfiles('.mat',full_path,1)';
    for j=1:length(tmp)
        if ~isempty(regexp(tmp{j},'Data'))
            files=[files;tmp(j)];
        end
    end
end

files_motor = files(1:72);
files_tong = files(73:end);
[D1,D2,D3,D4,idx1,idx2,idx3,idx4] = load_erp_data_online_B2(files);
[D5,D6,D7,D8,idx1,idx2,idx3,idx4] = load_erp_data_online_B2(files_tong);

% plot ERPs with all the imagined data
load('ECOG_Grid_8596-002131.mat')


% plot the ERPs with bootstrapped C.I. shading
chMap=ecog_grid;


% plot the ERPs with bootstrapped C.I. shading
figure
ha=tight_subplot(8,16);
d = 1;
set(gcf,'Color','w')
tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);
for i = 1:size(D4,1)
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
    erps =  squeeze(D4(i,:,:));

    chdata = erps;
    % zscore the data to the first 6 time-bins
    tmp_data=chdata(1:6,:);
    m = mean(tmp_data(:));
    s = std(tmp_data(:));
    chdata = (chdata -m)./s;

    % get the confidence intervals
    m = mean(chdata,2);
    mb = sort(bootstrp(1000,@mean,chdata'));
    tt=1:size(D1,2);
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
    idx=13:23;
    mstat = m((idx));
    pval=[];
    for j=1:length(idx)
        pval(j) = (sum(abs(mstat(j)) >= abs(tmp_mean(:,idx(j)))))./(size(tmp_mean,1));
    end

    [pfdr,~] = fdr(1-pval,0.05);
    % fdr approach
    res = sum((1-pval)<=pfdr);
    % nominal approach
    res=sum((1-pval)<=0.05);
    if res>=5
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
    if i~=107
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

% save
ERP_Data{1} = D1;
ERP_Data{2} = D2;
ERP_Data{3} = D3;
ERP_Data{4} = D4;
ERP_Data{5} = D5;
ERP_Data{6} = D6;
save ERP_Data_20210324_B2 -v7.3
ImaginedMvmt = {'Rt. Thumb','Leg','Lt Thumb','Head','Tongue','Lips'};

% get the average activity in M1 hand knob channels
hand_elec = [6 10 116 99 102 104 ];
roi_mean=[];
roi_dist_mean=[];
idx=13:22;
for i=1:length(ERP_Data)
    disp(i)
    data = ERP_Data{i};
    data = data(hand_elec,idx,:);
    data = squeeze(mean(data,2)); % time
    %data = squeeze(mean(data,1)); % channels
    data = data(:);
    roi_mean(i) = mean(data);
    %     if i==19
    %         roi_mean(i)=0.9;
    %     end
    roi_dist_mean(:,i) = sort(bootstrp(1000,@mean,data));
end
figure;bar(roi_mean)

y = roi_mean;
y=y';
errY(:,1) = roi_dist_mean(500,:)-roi_dist_mean(25,:);
errY(:,2) = roi_dist_mean(975,:)-roi_dist_mean(500,:);
figure;
barwitherr(errY, y);
xticks(1:6)
set(gcf,'Color','w')
set(gca,'FontSize',16)
set(gca,'LineWidth',1)
xticklabels(ImaginedMvmt)

%% ERPs B2 with higher samplign rate

clc;clear
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath('C:\Users\nikic\Documents\MATLAB')

filepath ='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2';
cd(filepath)
folders = {'20210324'};

% get the file names
files=[];
for i=1:length(folders)
    full_path = fullfile(filepath,folders{i},'DiscreteArrow');
    tmp = findfiles('.mat',full_path,1)';
    for j=1:length(tmp)
        if ~isempty(regexp(tmp{j},'Data'))
            files=[files;tmp(j)];
        end
    end
end

ImaginedMvmt = {'Right Thumb','Leg','Left Thumb','Head'};



% load the ERP data for each target
ERP_Data={};
for i=1:length(ImaginedMvmt)
    ERP_Data{i}=[];
end

% TIMING INFORMATION FOR THE TRIALS
Params.InterTrialInterval = 1; % rest period between trials
Params.InstructedDelayTime = 1; % only arrow environment
Params.CueTime = 1; % target appears
Params.ImaginedMvmtTime = 8; % Go time

% low pass filter of raw
lpFilt = designfilt('lowpassiir','FilterOrder',4, ...
    'PassbandFrequency',5,'PassbandRipple',0.2, ...
    'SampleRate',1e3);


%
% % log spaced hg filters
% Params.Fs = 1000;
% Params.FilterBank(1).fpass = [70,77];   % high gamma1
% Params.FilterBank(end+1).fpass = [77,85];   % high gamma2
% Params.FilterBank(end+1).fpass = [85,93];   % high gamma3
% Params.FilterBank(end+1).fpass = [93,102];  % high gamma4
% Params.FilterBank(end+1).fpass = [102,113]; % high gamma5
% Params.FilterBank(end+1).fpass = [113,124]; % high gamma6
% Params.FilterBank(end+1).fpass = [124,136]; % high gamma7
% Params.FilterBank(end+1).fpass = [136,150]; % high gamma8
% Params.FilterBank(end+1).fpass = [0.5,4]; % delta
% Params.FilterBank(end+1).fpass = [13,19]; % beta1
% Params.FilterBank(end+1).fpass = [19,30]; % beta2
%
% % compute filter coefficients
% for i=1:length(Params.FilterBank),
%     [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
%     Params.FilterBank(i).b = b;
%     Params.FilterBank(i).a = a;
% end

for i=1:length(files)
    disp(i/length(files)*100)
    load(files{i});
    features  = TrialData.BroadbandData;
    %features = cell2mat(features');
    Params = TrialData.Params;

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
    fidx=9:16;%filters idx, 9-15 is hg, 1 is delta and 4:5 is beta

    %%%%% state 1
    % extract state 1 data hG and resample it to 1s
    features1 = cell2mat(features(state1)');
    filtered_data=[];
    k=1;
    for ii=fidx
        filtered_data(:,:,k) =  abs(hilbert(filtfilt(...
            Params.FilterBank(ii).b, ...
            Params.FilterBank(ii).a, ...
            features1)));
        k=k+1;
    end
    if length(size(filtered_data))>2
        features1 = squeeze(mean(filtered_data,3));
    else
        features1 = filtered_data;
    end
    % interpolation
    if size(features1,1)~=1000
        %         tb = [1:size(features1,1)]*1e-3;
        %         t = [1:1000]*1e-3;

        tb = [1:size(features1,1)]*1e-3;
        t = [1:1000]*1e-3;
        tb = tb*t(end)/tb(end);
        features1 = interp1(tb,features1,t);
    end

    %%%%% state 2
    % extract state 1 data hG and resample it to 1s
    features2 = cell2mat(features(state2)');
    filtered_data=[];
    k=1;
    for ii=fidx
        filtered_data(:,:,k) =  abs(hilbert(filtfilt(...
            Params.FilterBank(ii).b, ...
            Params.FilterBank(ii).a, ...
            features2)));
        k=k+1;
    end
    if length(size(filtered_data))>2
        features2 = squeeze(mean(filtered_data,3));
    else
        features2 = filtered_data;
    end
    % interpolation
    if size(features2,1)~=1000
        tb = [1:size(features2,1)]*1e-3;
        t = [1:1000]*1e-3;
        tb = tb*t(end)/tb(end);
        features2 = interp1(tb,features2,t);
    end

    %%%%% state 3
    % extract state 1 data hG and resample it to 1s
    features3 = cell2mat(features(state3)');
    filtered_data=[];
    k=1;
    for ii=fidx
        filtered_data(:,:,k) =  abs(hilbert(filtfilt(...
            Params.FilterBank(ii).b, ...
            Params.FilterBank(ii).a, ...
            features3)));
        k=k+1;
    end
    if length(size(filtered_data))>2
        features3 = squeeze(mean(filtered_data,3));
    else
        features3 = filtered_data;
    end
    % interpolation
    if size(features3,1)~=3000
        tb = [1:size(features3,1)]*1e-3;
        t = [1:3000]*1e-3;
        tb = tb*t(end)/tb(end);
        features3 = interp1(tb,features3,t);
    end

    %%%%% state 4
    % extract state 1 data hG and resample it to 1s
    features4 = cell2mat(features(state4)');
    filtered_data=[];
    k=1;
    for ii=fidx
        filtered_data(:,:,k) =  abs(hilbert(filtfilt(...
            Params.FilterBank(ii).b, ...
            Params.FilterBank(ii).a, ...
            features4)));
        k=k+1;
    end
    if length(size(filtered_data))>2
        features4 = squeeze(mean(filtered_data,3));
    else
        features4 = filtered_data;
    end
    % interpolation
    if size(features4,1)~=1000
        tb = [1:size(features4,1)]*1e-3;
        t = [1:1000]*1e-3;
        tb = tb*t(end)/tb(end);
        features4 = interp1(tb,features4,t);
    end

    % now stitch the raw data back together
    data = [features1;features2;features3;features4];
    data = smoothdata(data,'movmean',25);
    figure;imagesc(data)
    title(num2str(i))

    % now z-score to the first 1s of data
    m = mean(data(1:1000,:));
    s = std(data(1:1000,:));
    data = (data-m)./s;

    targetID = TrialData.TargetID;
    tmp = ERP_Data{targetID};
    tmp = cat(3,tmp,data);
    ERP_Data{targetID} = tmp;
end

%save high_res_erp_beta_data_4Dir -v7.3
save high_res_erp_delta_data_4Dir -v7.3
%save high_res_erp_hg_data_4Dir -v7.3


% plot the ERPs for hg for instance
load('ECOG_Grid_8596-002131.mat')
chMap=ecog_grid;
figure
ha=tight_subplot(8,16);
d = 1;
set(gcf,'Color','w')
tim = cumsum([1000 1000 3000 1000]);
D1 = ERP_Data{1};
D1 = permute(D1,[2 1 3]);
for i = 1:size(D1,1)
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
    erps =  squeeze(D1(i,:,:));
    chdata = erps;

    for j=1:size(chdata,2)
        tmp = chdata(:,j);
        tmp(abs(tmp)>3) = median(tmp);


    end

    % get the confidence intervals
    m = mean(chdata,2);
    mb = sort(bootstrp(1000,@mean,chdata'));
    tt=1:size(D1,2);
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
        tmp_data=tmp(1:8,:);
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
    idx=13:27;
    mstat = m((idx));
    pval=[];
    for j=1:length(idx)
        pval(j) = (sum(abs(mstat(j)) >= abs(tmp_mean(:,idx(j)))))./(size(tmp_mean,1));
    end

    res=sum((1-pval)<=0.05);
    if res>=7
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




%% get all robot3D data B1
% especially useful for Jensen
% and beta analyses

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
                    %neural_feat = feat(513:640,:);
                    % getting kin variables
                    kin=TrialData.CursorState;
                    kin=kin(:,idx);
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

% is there a difference in beta power around the time when errors are made?
bad_neural=[];
good_neural=[];
chmap=TrialData.Params.ChMap;
beta_act=[];
acc=[];
beta_just_before_error=[];
for i=1:length(Robot3DTrials_onlyBetaFeatures)
    disp(i)
    kin = Robot3DTrials_onlyBetaFeatures(i).kin;
    target_loc = Robot3DTrials_onlyBetaFeatures(i).target_position;
    neural_feat = Robot3DTrials_onlyBetaFeatures(i).neural_feat;
    [c,s,l]=pca(neural_feat');
    s(:,1)=smooth(s(:,1));
    % check points where consequent update in position increased the error
    idx=[];
    if length(target_loc)<3
        target_loc(3)=0;
    end
    d = norm(target_loc' - kin(1:3,1));
    for j=2:size(kin,2)
        d1 = norm(target_loc' - kin(1:3,j));
        if d1>d
            idx(j)=1;
            %bad_neural = [bad_neural;s(j,1)];
        else
            idx(j)=0;
            %good_neural = [good_neural;s(j,1)];
        end
        d=d1;
    end
    tb = (1/5)*[0:length(idx)-1];
    t=(1/5)*[0:100];
    tb = tb*t(end)/tb(end);
    idx1 = interp1(tb,idx,t,'spline');
    s1 = interp1(tb,s(:,1),t,'spline');
    idx1(idx1>0.5)=1;
    idx1(idx1~=1)=0;
    acc = [acc;idx1];
    beta_act = [beta_act;s1];

    % trial level split
    gn = find(idx==1);
    bn = find(idx==0);
    good_neural=[good_neural;median(s(gn,1))];
    bad_neural=[bad_neural;median(s(bn,1))];

    % just before an error occured
    aa=diff(idx);
    aa=find(aa==1);
    tmp=[];
    for j=1:length(aa)
        tmp(j,:) = aa(j)-5:aa(j)-1;
    end
    tmp=tmp(:);
    tmp=tmp(tmp>0);
    beta_just_before_error = [beta_just_before_error;median(s(tmp,1))];
end

gn = bootstrp(1000,@nanmean,good_neural);
bn = bootstrp(1000,@nanmean,bad_neural);
jbn = bootstrp(1000,@nanmean,beta_just_before_error);

[h p tb st]=ttest2(bad_neural,beta_just_before_error)

figure;boxplot([gn jbn  bn])
ylabel('PC1 Beta Activity')
set(gcf,'Color','w')
xticks(1:3)
xticklabels({'Correct decodes','Just Before error','Bad Decodes'})
set(gca,'FontSize',12)
set(gca,'LineWidth',1)
title('Comparing beta power in cont. virtual 3D task')

figure;boxplot([good_neural bad_neural])

figure;
target_loc(3)=0
plot3(target_loc(1),target_loc(2),target_loc(3),'.k','MarkerSize',20)
hold on
xlim([-500 500])
ylim([-500 500])
zlim([-500 500])
for i=1:length(idx)
    if idx(i)==0
        plot3(kin(1,i),kin(2,i),kin(3,i),'.b');
    else
        plot3(kin(1,i),kin(2,i),kin(3,i),'.r');
    end
    pause(0.15)
end


% looking at accuracies around beta peaks i.e. beta bursts or positive
% fluctuations
bad_neural=[];
good_neural=[];
all_other=[];
chmap=TrialData.Params.ChMap;
beta_act=[];
acc=[];
beta_just_before_error=[];
for i=1:length(Robot3DTrials_onlyBetaFeatures)
    disp(i)
    kin = Robot3DTrials_onlyBetaFeatures(i).kin;
    target_loc = Robot3DTrials_onlyBetaFeatures(i).target_position;
    neural_feat = Robot3DTrials_onlyBetaFeatures(i).neural_feat;
    [c,s,l]=pca(neural_feat');
    s=smooth(s(:,1));
    s=s-mean(s);
    [aa idx]=findpeaks(s);
    idx = idx(aa>0);
    aa =  aa(aa>0);
    aa = aa(idx>5);
    idx = idx(idx>5);
    %figure;plot(s);
    %vline(idx)
    % now get only the top 50% of the beta peaks
    tmp = sort(aa,'ascend');
    if ~isempty(tmp)
        thresh = tmp(round(0.5*length(tmp)));
        idx=idx(aa>thresh);
        aa=aa(aa>thresh);
        beta_pks_idx = idx;
        %hold on
        %vline(idx,'k')

        % check points where consequent update in position increased the error
        idx=[];
        if length(target_loc)<3
            target_loc(3)=0;
        end
        d = norm(target_loc' - kin(1:3,1));
        for j=2:size(kin,2)
            d1 = norm(target_loc' - kin(1:3,j));
            if d1>d
                idx(j)=1;
                %bad_neural = [bad_neural;s(j,1)];
            else
                idx(j)=0;
                %good_neural = [good_neural;s(j,1)];
            end
            d=d1;
        end

        if length(idx)>120
            % get accuracies around the peak beta bursts
            % or maybe after 5-10 bins post beta peak coz of the model filter
            beta_tmp=[];
            for j=1:length(beta_pks_idx)
                beta_tmp(j,:) = beta_pks_idx(j)+[-7:+7];
            end
            beta_tmp = beta_tmp(:);
            beta_tmp = beta_tmp(beta_tmp<length(s));
            beta_tmp = beta_tmp(beta_tmp>0);

            % immediately following beta bursts
            non_beta_idx=[];
            for j=1:length(beta_pks_idx)
                non_beta_idx(j,:) = beta_pks_idx(j)+[8:12];
            end
            non_beta_idx = non_beta_idx(:);
            non_beta_idx = non_beta_idx(non_beta_idx<length(s));
            non_beta_idx = non_beta_idx(non_beta_idx>0);

            %find all other timepoints
            beta_tmp1 = unique([beta_tmp;non_beta_idx]);
            non_beta_idx1 = ones(length(idx),1);
            non_beta_idx1(beta_tmp1) = 0;
            non_beta_idx1 = logical(non_beta_idx1);

            % now split the accuracy data into those time periods
            bad_neural = [bad_neural;nanmean(idx(beta_tmp))];
            good_neural = [good_neural;nanmean(idx(non_beta_idx))];
            all_other = [all_other;nanmean(idx(non_beta_idx1))];
        end
    end

end

figure;boxplot([good_neural bad_neural])


gn = bootstrp(1000,@nanmean,good_neural);
bn = bootstrp(1000,@nanmean,bad_neural);
an = bootstrp(1000,@nanmean,all_other);


[h p tb st]=ttest2(good_neural,bad_neural)

figure;boxplot([gn  bn an])
ylabel('Accuracy')
set(gcf,'Color','w')
xticks(1:3)
xticklabels({'Immd. after beta peaks','Beta peaks','All other'})
set(gca,'FontSize',12)
set(gca,'LineWidth',1)
title('Continuous Task')

figure;plot(s,'LineWidth',1)
vline(beta_pks_idx)
set(gcf,'Color','w')
set(gca,'FontSize',12)
xlabel('Time bins in 200ms steps')
ylabel('Beta PC1')
set(gca,'LineWidth',1)
axis tight

%% LOOKING AT TRAVELING WAVES IN THE CONTINUOUS DATA

task_state= TrialData.TaskState;
feat = cell2mat(TrialData.BroadbandData');

% filter it in the theta range
Fs=1e3;
[b,a]=butter(3,[4 7]/(Fs/2));
[b1,a1]=butter(3,[70 150]/(Fs/2));
theta = abs(hilbert(filter(b1,a1,feat)));
theta = filter(b,a,theta);

% plot as a movie with a timestamp
tt=(0:size(theta,1))*(1/Fs);
chmap=TrialData.Params.ChMap;
figure;
set(gcf,'Color','w')
for i=1:size(theta,1)
    tmp = theta(i,:);
    imagesc(tmp(chmap))
    caxis([-0.06 0.06])
    colormap bone
    axis off
    title(num2str(tt(i)))
    pause(0.0005)
end

%% Looking at real-time signals


clc;clear

% filter design
Params=[];
Params.Fs = 1000;
Params.FilterBank(1).fpass = [0.5,4]; % low pass
Params.FilterBank(end+1).fpass = [4,8]; % theta
Params.FilterBank(end+1).fpass = [8,13]; % alpha
Params.FilterBank(end+1).fpass = [13,19]; % beta1
Params.FilterBank(end+1).fpass = [19,30]; % beta2
Params.FilterBank(end+1).fpass = [70,77];   % high gamma1
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

%load a block of data and filter it, with markers.
folderpath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220601\Robot3DArrow'
foldernames = {'105544'}
filepath = fullfile(folderpath,foldernames{1},'Imagined')
files= findfiles('',filepath)';

delta=[];
theta=[];
alpha=[];
beta=[];
hg=[];
raw=[];
trial_len=[];
for i=1:length(files)
    load(files{i})
    if TrialData.TargetID==1
        raw_data = cell2mat(TrialData.BroadbandData');
        trial_len=[trial_len;size(raw_data,1)];
        raw=[raw;raw_data];
    end
end
trial_len=cumsum(trial_len);

% extracting band specific information
delta = filter(Params.FilterBank(1).b,...
    Params.FilterBank(1).a,...
    raw);
theta = filter(Params.FilterBank(2).b,...
    Params.FilterBank(2).a,...
    raw);
alpha = filter(Params.FilterBank(3).b,...
    Params.FilterBank(3).a,...
    raw);
beta1 = filter(Params.FilterBank(4).b,...
    Params.FilterBank(4).a,...
    raw);
beta2 = filter(Params.FilterBank(5).b,...
    Params.FilterBank(5).a,...
    raw);
%beta1=(beta1.^2);
%beta2=(beta2.^2);
%beta = log10((beta1+beta2)/2);
beta = (abs(hilbert(beta1)) + abs(hilbert(beta2)))/2;

% hg filter bank approach -> square samples, log 10 and then average across
% bands
hg_bank=[];
for i=6:length(Params.FilterBank)
    tmp = filter(Params.FilterBank(i).b,...
        Params.FilterBank(i).a,...
        raw);
    %tmp=tmp.^2;
    tmp=abs(hilbert(tmp));
    hg_bank = cat(3,hg_bank,tmp);
end
%hg = log10(squeeze(mean(hg_bank,3)));
hg = (squeeze(mean(hg_bank,3)));


figure;
subplot(4,1,1)
plot(raw(:,3))
vline(trial_len,'r')
title('raw')
axis tight

subplot(4,1,2)
plot(abs(hilbert(delta(:,3))))
vline(trial_len,'r')
title('delta')
axis tight

subplot(4,1,3)
plot(beta(:,3))
vline(trial_len,'r')
title('beta')
axis tight

subplot(4,1,4)
plot(hg(:,4))
vline(trial_len,'r')
title('hg')
axis tight

sgtitle('Target 1, Ch3')
set(gcf,'Color','w')

%% Looking at real-time signals and getting higher res ERPs
% ERPs higher res

clc;clear

% filter design
Params=[];
Params.Fs = 1000;
Params.FilterBank(1).fpass = [0.5,4]; % low pass
Params.FilterBank(end+1).fpass = [4,8]; % theta
Params.FilterBank(end+1).fpass = [8,13]; % alpha
Params.FilterBank(end+1).fpass = [13,19]; % beta1
Params.FilterBank(end+1).fpass = [19,30]; % beta2
Params.FilterBank(end+1).fpass = [70,77];   % high gamma1
Params.FilterBank(end+1).fpass = [77,85];   % high gamma2
Params.FilterBank(end+1).fpass = [85,93];   % high gamma3
Params.FilterBank(end+1).fpass = [93,102];  % high gamma4
Params.FilterBank(end+1).fpass = [102,113]; % high gamma5
Params.FilterBank(end+1).fpass = [113,124]; % high gamma6
Params.FilterBank(end+1).fpass = [124,136]; % high gamma7
Params.FilterBank(end+1).fpass = [136,150]; % high gamma8
Params.FilterBank(end+1).fpass = [20]; % raw

% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end

%load a block of data and filter it, with markers.
folderpath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220601\Robot3DArrow'
foldernames = {'105544','110134','110453','110817'}
files=[];
for i=1:length(foldernames)
    filepath = fullfile(folderpath,foldernames{i},'Imagined');
    files= [files; findfiles('',filepath)'];
end

delta=[];
theta=[];
alpha=[];
beta=[];
hg=[];
raw=[];
trial_len=[];
state1_len=[];
for i=1:length(files)
    load(files{i})
    if TrialData.TargetID==1
        raw_data = cell2mat(TrialData.BroadbandData');
        trial_len=[trial_len;size(raw_data,1)];
        raw=[raw;raw_data];
        idx=find(TrialData.TaskState==1);
        tmp=cell2mat(TrialData.BroadbandData(idx)');
        state1_len=[state1_len size(tmp,1)];
    end
end
trial_len_total=cumsum(trial_len);

% extracting band specific information
delta = filter(Params.FilterBank(1).b,...
    Params.FilterBank(1).a,...
    raw);
delta=abs(hilbert(delta));

theta = filter(Params.FilterBank(2).b,...
    Params.FilterBank(2).a,...
    raw);

alpha = filter(Params.FilterBank(3).b,...
    Params.FilterBank(3).a,...
    raw);

beta1 = filter(Params.FilterBank(4).b,...
    Params.FilterBank(4).a,...
    raw);
beta2 = filter(Params.FilterBank(5).b,...
    Params.FilterBank(5).a,...
    raw);
%beta1=(beta1.^2);
%beta2=(beta2.^2);
%beta = log10((beta1+beta2)/2);
beta = (abs(hilbert(beta1)) + abs(hilbert(beta2)))/2;

% hg filter bank approach -> square samples, log 10 and then average across
% bands
hg_bank=[];
for i=6:length(Params.FilterBank)-1
    tmp = filter(Params.FilterBank(i).b,...
        Params.FilterBank(i).a,...
        raw);
    %tmp=tmp.^2;
    tmp=abs(hilbert(tmp));
    hg_bank = cat(3,hg_bank,tmp);
end
%hg = log10(squeeze(mean(hg_bank,3)));
hg = (squeeze(mean(hg_bank,3)));

% lpf the raw
raw = filter(Params.FilterBank(end).b,...
    Params.FilterBank(end).a,...
    raw);
for j=1:size(raw,2)
    raw(:,j) = smooth(raw(:,j),100);
end

% now going and referencing each trial to state 1 data
raw_ep={};
delta_ep={};
beta_ep={};
hg_ep={};
trial_len_total=[0 ;trial_len_total];
for i=1:length(trial_len_total)-1
    % raw
    tmp_data = raw(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    raw_ep = cat(2,raw_ep,tmp_data);

    %delta
    tmp_data = delta(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    delta_ep = cat(2,delta_ep,tmp_data);

    %beta
    tmp_data = beta(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    beta_ep = cat(2,beta_ep,tmp_data);

    %hg
    tmp_data = hg(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    hg_ep = cat(2,hg_ep,tmp_data);
end


figure;
temp=cell2mat(hg_ep');
plot(temp(:,3));
axis tight
vline(trial_len_total,'r')


%
% figure;
% subplot(4,1,1)
% plot(raw(:,3))
% vline(trial_len,'r')
% title('raw')
% axis tight
%
% subplot(4,1,2)
% plot(abs(hilbert(delta(:,3))))
% vline(trial_len,'r')
% title('delta')
% axis tight
%
% subplot(4,1,3)
% plot(beta(:,3))
% vline(trial_len,'r')
% title('beta')
% axis tight
%
% subplot(4,1,4)
% plot(hg(:,4))
% vline(trial_len,'r')
% title('hg')
% axis tight
%
% sgtitle('Target 1, Ch3')
% set(gcf,'Color','w')

%hg erps - take the frst 7800 time points
hg_data=[];
for i=1:length(hg_ep)
    tmp=hg_ep{i};
    hg_data = cat(3,hg_data,tmp(1:7800,:));
end

figure;
subplot(2,1,1)
ch=106;
plot(squeeze(hg_data(:,ch,:)),'Color',[.5 .5 .5 .5])
hold on
plot(squeeze(mean(hg_data(:,ch,:),3)),'Color','b','LineWidth',2)
vline(TrialData.Params.InstructedDelayTime*1e3,'r')
vline(TrialData.Params.InstructedDelayTime*1e3 + TrialData.Params.CueTime*1e3,'g')
vline(TrialData.Params.MaxReachTime*1e3 + TrialData.Params.InstructedDelayTime*1e3 ...
    +  TrialData.Params.CueTime*1e3, 'k')
hline(0)
hline(0)
set(gcf,'Color','w')
xlabel('Time in ms')
ylabel('uV')
title('Hg Ch 106 Left Leg')
set(gca,'LineWidth',1)
set(gca,'FontSize',14)
axis tight
ylim([-5 10])

%delta erps - take the frst 7800 time points
delta_data=[];
for i=1:length(delta_ep)
    tmp=delta_ep{i};
    %     for j=1:size(tmp,2)
    %         tmp(:,j)=smooth(tmp(:,j),100);
    %     end
    delta_data = cat(3,delta_data,tmp(1:7800,:));
end

%figure;
subplot(2,1,2)
ch=106;
plot(squeeze(delta_data(:,ch,:)),'Color',[.5 .5 .5 .5])
hold on
plot(squeeze(mean(delta_data(:,ch,:),3)),'Color','b','LineWidth',2)
vline(TrialData.Params.InstructedDelayTime*1e3,'r')
vline(TrialData.Params.InstructedDelayTime*1e3 + TrialData.Params.CueTime*1e3,'g')
vline(TrialData.Params.MaxReachTime*1e3 + TrialData.Params.InstructedDelayTime*1e3 ...
    +  TrialData.Params.CueTime*1e3, 'k')
hline(0)
set(gcf,'Color','w')
xlabel('Time in ms')
ylabel('uV')
title('Raw Ch 106 Left Leg')
set(gca,'LineWidth',1)
set(gca,'FontSize',14)
axis tight
ylim([-5 5])

%
% ch=106;
% tmp=squeeze(delta_data(:,ch,:));
% for i=1:size(tmp,2)
%     figure;plot(tmp(:,i))
%     vline(TrialData.Params.InstructedDelayTime*1e3,'r')
%     vline(TrialData.Params.InstructedDelayTime*1e3 + TrialData.Params.CueTime*1e3,'g')
%     vline(TrialData.Params.MaxReachTime*1e3 + TrialData.Params.InstructedDelayTime*1e3 ...
%         +  TrialData.Params.CueTime*1e3, 'k')
%     hline(0)
% end
%  ylim([-3 3])


% plotting covariance of raw
tmp=zscore(raw);
figure;imagesc(cov(raw))
[c,s,l]=pca((raw));
chmap=TrialData.Params.ChMap;
tmp1=c(:,1);
figure;imagesc(tmp1(chmap))
figure;
stem(cumsum(l)./sum(l))


%% Looking at real-time signals and getting higher res ERPs
% ERPs higher res
% with traveling waves

clc;clear

% filter design
Params=[];
Params.Fs = 1000;
Params.FilterBank(1).fpass = [0.5,4]; % low pass
Params.FilterBank(end+1).fpass = [4,8]; % theta
Params.FilterBank(end+1).fpass = [8,13]; % alpha
Params.FilterBank(end+1).fpass = [13,19]; % beta1
Params.FilterBank(end+1).fpass = [19,30]; % beta2
Params.FilterBank(end+1).fpass = [70,77];   % high gamma1
Params.FilterBank(end+1).fpass = [77,85];   % high gamma2
Params.FilterBank(end+1).fpass = [85,93];   % high gamma3
Params.FilterBank(end+1).fpass = [93,102];  % high gamma4
Params.FilterBank(end+1).fpass = [102,113]; % high gamma5
Params.FilterBank(end+1).fpass = [113,124]; % high gamma6
Params.FilterBank(end+1).fpass = [124,136]; % high gamma7
Params.FilterBank(end+1).fpass = [136,150]; % high gamma8
Params.FilterBank(end+1).fpass = [20]; % raw

% compute filter coefficients
for i=1:length(Params.FilterBank),
    [b,a] = butter(3,Params.FilterBank(i).fpass/(Params.Fs/2));
    Params.FilterBank(i).b = b;
    Params.FilterBank(i).a = a;
end

%load a block of data and filter it, with markers.
folderpath='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20220601\Robot3DArrow'
foldernames = {'105544','110134','110453','110817'}
files=[];
for i=1:length(foldernames)
    filepath = fullfile(folderpath,foldernames{i},'Imagined');
    files= [files; findfiles('',filepath)'];
end

delta=[];
theta=[];
alpha=[];
beta=[];
hg=[];
raw=[];
trial_len=[];
state1_len=[];
for i=1:length(files)
    load(files{i})
    if TrialData.TargetID==1
        raw_data = cell2mat(TrialData.BroadbandData');
        trial_len=[trial_len;size(raw_data,1)];
        raw=[raw;raw_data];
        idx=find(TrialData.TaskState==1);
        tmp=cell2mat(TrialData.BroadbandData(idx)');
        state1_len=[state1_len size(tmp,1)];
    end
end
trial_len_total=cumsum(trial_len);

% extracting band specific information
delta = filter(Params.FilterBank(1).b,...
    Params.FilterBank(1).a,...
    raw);
delta=abs(hilbert(delta));

theta = filter(Params.FilterBank(2).b,...
    Params.FilterBank(2).a,...
    raw);

alpha = filter(Params.FilterBank(3).b,...
    Params.FilterBank(3).a,...
    raw);

beta1 = filter(Params.FilterBank(4).b,...
    Params.FilterBank(4).a,...
    raw);
beta2 = filter(Params.FilterBank(5).b,...
    Params.FilterBank(5).a,...
    raw);
%beta1=(beta1.^2);
%beta2=(beta2.^2);
%beta = log10((beta1+beta2)/2);
beta = (abs(hilbert(beta1)) + abs(hilbert(beta2)))/2;

% hg filter bank approach -> square samples, log 10 and then average across
% bands
hg_bank=[];
for i=6:length(Params.FilterBank)-1
    tmp = filter(Params.FilterBank(i).b,...
        Params.FilterBank(i).a,...
        raw);
    %tmp=tmp.^2;
    tmp=abs(hilbert(tmp));
    hg_bank = cat(3,hg_bank,tmp);
end
%hg = log10(squeeze(mean(hg_bank,3)));
hg = (squeeze(mean(hg_bank,3)));

% lpf the raw
raw = filter(Params.FilterBank(end).b,...
    Params.FilterBank(end).a,...
    raw);
for j=1:size(raw,2)
    raw(:,j) = smooth(raw(:,j),100);
end

% now going and referencing each trial to state 1 data
raw_ep={};
delta_ep={};
beta_ep={};
hg_ep={};
trial_len_total=[0 ;trial_len_total];
for i=1:length(trial_len_total)-1
    % raw
    tmp_data = raw(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    raw_ep = cat(2,raw_ep,tmp_data);

    %delta
    tmp_data = delta(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    delta_ep = cat(2,delta_ep,tmp_data);

    %beta
    tmp_data = beta(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    beta_ep = cat(2,beta_ep,tmp_data);

    %hg
    tmp_data = hg(trial_len_total(i)+1:trial_len_total(i+1),:);
    tmp_baseline = tmp_data(1:state1_len(i),:);
    m = mean(tmp_baseline);
    s = std(tmp_baseline);
    tmp_data = (tmp_data -m)./s;
    hg_ep = cat(2,hg_ep,tmp_data);
end


figure;
temp=cell2mat(hg_ep');
plot(temp(:,3));
axis tight
vline(trial_len_total,'r')


%
% figure;
% subplot(4,1,1)
% plot(raw(:,3))
% vline(trial_len,'r')
% title('raw')
% axis tight
%
% subplot(4,1,2)
% plot(abs(hilbert(delta(:,3))))
% vline(trial_len,'r')
% title('delta')
% axis tight
%
% subplot(4,1,3)
% plot(beta(:,3))
% vline(trial_len,'r')
% title('beta')
% axis tight
%
% subplot(4,1,4)
% plot(hg(:,4))
% vline(trial_len,'r')
% title('hg')
% axis tight
%
% sgtitle('Target 1, Ch3')
% set(gcf,'Color','w')

%hg erps - take the frst 7800 time points
hg_data=[];
for i=1:length(hg_ep)
    tmp=hg_ep{i};
    hg_data = cat(3,hg_data,tmp(1:7800,:));
end

figure;
subplot(2,1,1)
ch=106;
plot(squeeze(hg_data(:,ch,:)),'Color',[.5 .5 .5 .5])
hold on
plot(squeeze(mean(hg_data(:,ch,:),3)),'Color','b','LineWidth',2)
vline(TrialData.Params.InstructedDelayTime*1e3,'r')
vline(TrialData.Params.InstructedDelayTime*1e3 + TrialData.Params.CueTime*1e3,'g')
vline(TrialData.Params.MaxReachTime*1e3 + TrialData.Params.InstructedDelayTime*1e3 ...
    +  TrialData.Params.CueTime*1e3, 'k')
hline(0)
hline(0)
set(gcf,'Color','w')
xlabel('Time in ms')
ylabel('uV')
title('Hg Ch 106 Left Leg')
set(gca,'LineWidth',1)
set(gca,'FontSize',14)
axis tight
ylim([-5 10])

%delta erps - take the frst 7800 time points
delta_data=[];
for i=1:length(delta_ep)
    tmp=delta_ep{i};
    %     for j=1:size(tmp,2)
    %         tmp(:,j)=smooth(tmp(:,j),100);
    %     end
    delta_data = cat(3,delta_data,tmp(1:7800,:));
end

%figure;
subplot(2,1,2)
ch=106;
plot(squeeze(delta_data(:,ch,:)),'Color',[.5 .5 .5 .5])
hold on
plot(squeeze(mean(delta_data(:,ch,:),3)),'Color','b','LineWidth',2)
vline(TrialData.Params.InstructedDelayTime*1e3,'r')
vline(TrialData.Params.InstructedDelayTime*1e3 + TrialData.Params.CueTime*1e3,'g')
vline(TrialData.Params.MaxReachTime*1e3 + TrialData.Params.InstructedDelayTime*1e3 ...
    +  TrialData.Params.CueTime*1e3, 'k')
hline(0)
set(gcf,'Color','w')
xlabel('Time in ms')
ylabel('uV')
title('Raw Ch 106 Left Leg')
set(gca,'LineWidth',1)
set(gca,'FontSize',14)
axis tight
ylim([-5 5])

%
% ch=106;
% tmp=squeeze(delta_data(:,ch,:));
% for i=1:size(tmp,2)
%     figure;plot(tmp(:,i))
%     vline(TrialData.Params.InstructedDelayTime*1e3,'r')
%     vline(TrialData.Params.InstructedDelayTime*1e3 + TrialData.Params.CueTime*1e3,'g')
%     vline(TrialData.Params.MaxReachTime*1e3 + TrialData.Params.InstructedDelayTime*1e3 ...
%         +  TrialData.Params.CueTime*1e3, 'k')
%     hline(0)
% end
%  ylim([-3 3])


% plotting covariance of raw
tmp=zscore(raw);
figure;imagesc(cov(raw))
[c,s,l]=pca((raw));
chmap=TrialData.Params.ChMap;
tmp1=c(:,1);
figure;imagesc(tmp1(chmap))
figure;
stem(cumsum(l)./sum(l))




%% boostrapping

fs=1e3;
t=1:2e3;
f1=2;
f2=4;
x=sin(2*pi*(f1/fs)*(t))' + randn(length(t),1);
y=sin(2*pi*(f2/fs)*(t))' + randn(length(t),1);

[C,lags]=xcorr(x,y);
figure;plot(lags,C)

% bootstrapping with circular shuffle
Cboot=[];
pow_boot=[];
for iter=1:1000
    k = randperm(length(x),1);
    l = randperm(length(y),1);
    x1 = circshift(x,k);
    y1 = circshift(y,k);
    Cboot(iter,:) = xcorr(x1,y)-C;
    [pow_boot(iter,:), F] = pwelch(Cboot(iter,:),[],[],[],fs);
end
figure;plot(lags,Cboot','Color',[.5 .5 .5 .25])
hold on
plot(lags,C,'Color','b','LineWidth',1)
title('Circ Shuffle - phase is different')


% bootstrapping with random permutation
Cboot=[];
pow_boot=[];
for iter=1:1000
    k = randperm(length(x));
    x1 = x(k);
    l = randperm(length(y));
    y1 = y(l);
    Cboot(iter,:) = xcorr(x1,y1);
    [pow_boot(iter,:), F] = pwelch(Cboot(iter,:),[],[],[],fs);
end
figure;plot(lags,Cboot','Color',[.5 .5 .5 .25])
hold on
plot(lags,C,'Color','b','LineWidth',1)
title('Random Permutation - No oscillation present')


figure;plot(F,log10(abs(pow_boot)),'Color',[.5 .5 .5 .25])
hold on
pow = pwelch(C,[],[],[],fs);
plot(F,log10(abs(pow)),'b','LineWidth',1)
xlim([0 100])
title('Significant rhythmicity based on FFT of correlation')

%% LOOKING THE VARIANCE OF HIGH NOISE CHANNELS LATE DEC 2023


clc;clear
root_path='F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
cd(root_path)
addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'
addpath 'C:\Users\nikic\Documents\MATLAB'

% bad folders
foldernames = {'20231218','20231220','20231228','20231229','20231129'};
files=[];
for i=1:length(foldernames)
    folderpath = fullfile(root_path,foldernames{i},'Robot3DArrow');
    D=dir(folderpath);
    for j=3:length(D)
        filepath = fullfile(folderpath,D(j).name,'BCI_Fixed');
        if exist(filepath)
            files=[files;findfiles('',filepath)'];
        end
    end
end

% now get the variance of individual channels in hG
var_hg=[];
for i=1:length(files)
    disp(i/length(files)*100)
    files_loaded=1;
    try
        load(files{i})
    catch
        files_loaded=0;
    end
    if files_loaded
        features=cell2mat(TrialData.NeuralFeatures);
        hg_features = features(1537:1792,:);
        %bad_ch = [108 113 118];
        good_ch = ones(size(hg_features,1),1);
        %good_ch(bad_ch)=0;
        hg_features = hg_features(logical(good_ch),:);
        %var_hg=[var_hg;std(hg_features',1)];
        var_hg=[var_hg (hg_features)];
    end
end

load('ECOG_Grid_8596_000067_B3.mat')    
chmap=ecog_grid;
%m = median(var_hg,1);
m = std(var_hg');
figure;imagesc(m(chmap'))

%%  ROBOT CENTER OUT DATA LOOKING AT TRAJECTORIES TO OFF DIAGONAL TARGETS

clc;clear
foldersnames = {''}





