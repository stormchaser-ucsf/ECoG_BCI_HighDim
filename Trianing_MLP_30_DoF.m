

% perform classification for 30DoF imagined mvmt data

clc;clear
% enter the root path from the Data folder


dates = {'20211201','20211203','20211206','20211208','20211215'};
%20210917 has issues with the way the order of targets were flipped, need
%to manually input the correct folders for this particular date


root_path = '/media/reza/WindowsDrive/BRAVO1/CursorPlatform/Data';

cd(root_path)

%FOR IMAGINED MOVEMENT DATA

% structure to host all the data

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
D11=[];
D12=[];
D13=[];
D14=[];
D15=[];
D16=[];
D17=[];
D18=[];
D19=[];
D20=[];
D21=[];
D22=[];
D23=[];
D24=[];
D25=[];
D26=[];
D27=[];
D28=[];
D29=[];
D30=[];



% load the target names from an example file
targets={'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
    'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
    'Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp',...
    'Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
    'Squeeze Both Hands',...
    'Rotate Right Wrist','Rotate Left Wrist',...
    'Imagined Head Movement',...
    'Right Shoulder Shrug',...
    'Left Shoulder Shrug',...
    'Right Bicep','Left Bicep',...
    'Right Tricep','Left Tricep',...
    'Right Leg','Left Leg',...
    'Lips','Tongue'};

for ii=1:length(dates)
    disp(ii/length(dates))

    folderpath = fullfile(root_path, dates{ii},'ImaginedMvmtDAQ');
    tmp = findfiles('mat',folderpath,1)';
    files=[];
    for i=1:length(tmp)
        if regexp(tmp{i},'Imagined')
            files=[files;tmp(i)];
        end
    end

    % now load the data
    for j=1:length(files)

        load(files{j})
        features  = TrialData.SmoothedNeuralFeatures;
        kinax = [ find(TrialData.TaskState==3)];
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

        % find out which target it belongs to
        idx=[];
        for k=1:length(targets)
            if regexp(targets{k},TrialData.ImaginedAction)
                idx=k;
                break
            end
        end

        if idx == 1
            D1 = [D1 temp];
        elseif idx == 2
            D2 = [D2 temp];
        elseif idx == 3
            D3 = [D3 temp];
        elseif idx == 4
            D4 = [D4 temp];
        elseif idx == 5
            D5 = [D5 temp];
        elseif idx == 6
            D6 = [D6 temp];
        elseif idx == 7
            D7 = [D7 temp];
        elseif idx == 8
            D8 = [D8 temp];
        elseif idx == 9
            D9 = [D9 temp];
        elseif idx == 10
            D10 = [D10 temp];
        elseif idx == 11
            D11 = [D11 temp];
        elseif idx == 12
            D12 = [D12 temp];
        elseif idx == 13
            D13 = [D13 temp];
        elseif idx == 14
            D14 = [D14 temp];
        elseif idx == 15
            D15 = [D15 temp];
        elseif idx == 16
            D16 = [D16 temp];
        elseif idx == 17
            D17 = [D17 temp];
        elseif idx == 18
            D18 = [D18 temp];
        elseif idx == 19
            D19 = [D19 temp];
        elseif idx == 20
            D20 = [D20 temp];
        elseif idx == 21
            D21 = [D21 temp];
        elseif idx == 22
            D22 = [D22 temp];
        elseif idx == 23
            D23 = [D23 temp];
        elseif idx == 24
            D24 = [D24 temp];
        elseif idx == 25
            D25 = [D25 temp];
        elseif idx == 26
            D26 = [D26 temp];
        elseif idx == 27
            D27 = [D27 temp];
        elseif idx == 28
            D28 = [D28 temp];
        elseif idx == 29
            D29 = [D29 temp];
        elseif idx == 30
            D30 = [D30 temp];
        end
    end
end


clear condn_data
% combing delta beta and high gamma
idx=[1:size(D1,1)];
condn_data{1}=[ D1(idx,:) ]';
condn_data{2}= [ D2(idx,:)]';
condn_data{3}=[ D3(idx,:)]';
condn_data{4}=[ D4(idx,:)]';
condn_data{5}=[ D5(idx,:)]';
condn_data{6}=[ D6(idx,:)]';
condn_data{7}=[ D7(idx,:)]';
condn_data{8}=[ D8(idx,:)]';
condn_data{9}=[ D9(idx,:)]';
condn_data{10}=[ D10(idx,:)]';
condn_data{11}=[ D11(idx,:)]';
condn_data{12}=[ D12(idx,:)]';
condn_data{13}=[ D13(idx,:)]';
condn_data{14}=[ D14(idx,:)]';
condn_data{15}=[ D15(idx,:)]';
condn_data{16}=[ D16(idx,:)]';
condn_data{17}=[ D17(idx,:)]';
condn_data{18}=[ D18(idx,:)]';
condn_data{19}=[ D19(idx,:)]';
condn_data{20}=[ D20(idx,:)]';
condn_data{21}=[ D21(idx,:)]';
condn_data{22}=[ D22(idx,:)]';
condn_data{23}=[ D23(idx,:)]';
condn_data{24}=[ D24(idx,:)]';
condn_data{25}=[ D25(idx,:)]';
condn_data{26}=[ D26(idx,:)]';
condn_data{27}=[ D27(idx,:)]';
condn_data{28}=[ D28(idx,:)]';
condn_data{29}=[ D29(idx,:)]';
condn_data{30}=[ D30(idx,:)]';



N=[];
T1=[];
for i=1:length(condn_data)
    tmp=condn_data{i};
    N = [N tmp'];
    T1 = [T1;i*ones(size(tmp,1),1)];
end


T = zeros(size(T1,1),7);
for i=1:30
    [aa bb]=find(T1==i);
    T(aa(1):aa(end),i)=1;
end



% splitting into training and testing
T_train=[];
N_train=[];
T_test=[];
N_test=[];
for i=1:size(T,2)
    idx =find(T(:,i)==1);
    idx=idx(randperm(length(idx)));
    % set aside 400 for training
    T_train = [T_train; T(idx(1:480),:)];
    T_test = [T_test; T(idx(480:end),:)];
    N_train = [N_train N(:,idx(1:480))];
    N_test = [N_test N(:,idx(480:end))];
end




%
% A = condn_data{1};
% B = condn_data{2};
% C = condn_data{3};
% D = condn_data{4};
% E = condn_data{5};
% F = condn_data{6};
% G = condn_data{7};
%
% clear N
% N = [A' B' C' D' E' F' G'];
% T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
%     5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1)];
%
% T = zeros(size(T1,1),7);
% [aa bb]=find(T1==1);
% T(aa(1):aa(end),1)=1;
% [aa bb]=find(T1==2);
% T(aa(1):aa(end),2)=1;
% [aa bb]=find(T1==3);
% T(aa(1):aa(end),3)=1;
% [aa bb]=find(T1==4);
% T(aa(1):aa(end),4)=1;
% [aa bb]=find(T1==5);
% T(aa(1):aa(end),5)=1;
% [aa bb]=find(T1==6);
% T(aa(1):aa(end),6)=1;
% [aa bb]=find(T1==7);
% T(aa(1):aa(end),7)=1;


% % randomize order first
% idx = randperm(size(T,1));
% T = T(idx,:);
% N=N(:,idx);
%
% train_idx = randperm(size(T,1),round(0.8*size(T,1)));
% I = ones(size(T,1),1);
% I(train_idx)=0;
% test_idx=find(I==1);
%
% Ntrain = N(:,train_idx);
% Ntest = N(:,test_idx);
% Ttrain = T(train_idx,:);
% Ttest = T(test_idx,:);


% splitting into training and testing

clear net
net = patternnet([64 64 64]) ;
net.performParam.regularization=0.2;
net.divideParam.trainRatio=0.8;
net.divideParam.valRatio=0.2;
net.divideParam.testRatio=0.0;
net = train(net,N_train,T_train','useParallel','yes');
%net = train(net,N,T','useParallel','yes');
% classifier_name = 'MLP_PreTrained_7DoF_Days1to11_924pm2'; % enter the name
% genFunction(pretrain_net,classifier_name); % make sure to update GetParams

out = net(N_test);
acc = zeros(30,30);
for i=1:size(out,2)
    tmp = out(:,i);
    [aa bb]=max(tmp);
    idx = T_test(i,:);
    idx=find(idx==1);
    acc(idx,bb)=acc(idx,bb)+1;
end
for i=1:size(acc,1)
    acc(i,:)=acc(i,:)./sum(acc(i,:));
end
figure;imagesc(acc)
colormap bone
xticks(1:30)
xticklabels(targets)
yticks(1:30)
yticklabels(targets)
set(gcf,'Color','w')
colorbar
set(gca,'FontSize',18)
caxis([0 0.65])


figure;stem(1:30,diag(acc))
xticks(1:30)
xticklabels(targets)

%%%%%%% CODE SNIPPET FOR TRAINING A MODEL FROM SCRATCH %%%%%
% training a simple MLP
% IMPORTANT, CLICK THE CONFUSION MATRIX BUTTON IN GUI TO VERIFY THAT THE
% TEST VALIDATION DOESN'T HAVE NaNs AND THAT PERFORMANCE IS REASONABLE
%  clear net
%  net = patternnet([64 64 64]) ;
%  net.performParam.regularization=0.2;

% cd('/home/ucsf/Projects/bci/clicker')
% load net net
% %
%  net = train(net,N,T');
%


%% same as above but trial level classification


% split the data into training and testing trials


clc;clear
% enter the root path from the Data folder


dates = {'20211201','20211203','20211206','20211208','20211215'};
%20210917 has issues with the way the order of targets were flipped, need
%to manually input the correct folders for this particular date


root_path = '/media/reza/WindowsDrive/BRAVO1/CursorPlatform/Data';

cd(root_path)

%FOR IMAGINED MOVEMENT DATA




% load the target names from an example file
targets={'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
    'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
    'Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp',...
    'Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
    'Squeeze Both Hands',...
    'Rotate Right Wrist','Rotate Left Wrist',...
    'Imagined Head Movement',...
    'Right Shoulder Shrug',...
    'Left Shoulder Shrug',...
    'Right Bicep','Left Bicep',...
    'Right Tricep','Left Tricep',...
    'Right Leg','Left Leg',...
    'Lips','Tongue'};

files=[];
labels={};
for ii=1:length(dates)
    disp(ii/length(dates))

    folderpath = fullfile(root_path, dates{ii},'ImaginedMvmtDAQ');
    tmp = findfiles('mat',folderpath,1)';
    for i=1:length(tmp)
        if isempty(regexp(tmp{i},'kf_params'))
            files=[files;tmp(i)];
        end
    end
end

acc_overall=[];
for iter=1:20
    disp(iter)

    % structure to host all the data
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
    D11=[];
    D12=[];
    D13=[];
    D14=[];
    D15=[];
    D16=[];
    D17=[];
    D18=[];
    D19=[];
    D20=[];
    D21=[];
    D22=[];
    D23=[];
    D24=[];
    D25=[];
    D26=[];
    D27=[];
    D28=[];
    D29=[];
    D30=[];





    % split into training and testing files
    idx=randperm(length(files));
    files=files(idx);
    len = round(0.8*length(idx));
    files_train = files(1:len);
    files_test = files(len+1:end);

    % now load the data
    for j=1:length(files_train)

        load(files_train{j})
        features  = TrialData.SmoothedNeuralFeatures;
        kinax = [ find(TrialData.TaskState==3)];
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

        % find out which target it belongs to
        idx=[];
        for k=1:length(targets)
            if regexp(targets{k},TrialData.ImaginedAction)
                idx=k;
                break
            end
        end

        if idx == 1
            D1 = [D1 temp];
        elseif idx == 2
            D2 = [D2 temp];
        elseif idx == 3
            D3 = [D3 temp];
        elseif idx == 4
            D4 = [D4 temp];
        elseif idx == 5
            D5 = [D5 temp];
        elseif idx == 6
            D6 = [D6 temp];
        elseif idx == 7
            D7 = [D7 temp];
        elseif idx == 8
            D8 = [D8 temp];
        elseif idx == 9
            D9 = [D9 temp];
        elseif idx == 10
            D10 = [D10 temp];
        elseif idx == 11
            D11 = [D11 temp];
        elseif idx == 12
            D12 = [D12 temp];
        elseif idx == 13
            D13 = [D13 temp];
        elseif idx == 14
            D14 = [D14 temp];
        elseif idx == 15
            D15 = [D15 temp];
        elseif idx == 16
            D16 = [D16 temp];
        elseif idx == 17
            D17 = [D17 temp];
        elseif idx == 18
            D18 = [D18 temp];
        elseif idx == 19
            D19 = [D19 temp];
        elseif idx == 20
            D20 = [D20 temp];
        elseif idx == 21
            D21 = [D21 temp];
        elseif idx == 22
            D22 = [D22 temp];
        elseif idx == 23
            D23 = [D23 temp];
        elseif idx == 24
            D24 = [D24 temp];
        elseif idx == 25
            D25 = [D25 temp];
        elseif idx == 26
            D26 = [D26 temp];
        elseif idx == 27
            D27 = [D27 temp];
        elseif idx == 28
            D28 = [D28 temp];
        elseif idx == 29
            D29 = [D29 temp];
        elseif idx == 30
            D30 = [D30 temp];
        end
    end



    clear condn_data
    % combing delta beta and high gamma
    idx=[1:size(D1,1)];
    condn_data{1}=[ D1(idx,:) ]';
    condn_data{2}= [ D2(idx,:)]';
    condn_data{3}=[ D3(idx,:)]';
    condn_data{4}=[ D4(idx,:)]';
    condn_data{5}=[ D5(idx,:)]';
    condn_data{6}=[ D6(idx,:)]';
    condn_data{7}=[ D7(idx,:)]';
    condn_data{8}=[ D8(idx,:)]';
    condn_data{9}=[ D9(idx,:)]';
    condn_data{10}=[ D10(idx,:)]';
    condn_data{11}=[ D11(idx,:)]';
    condn_data{12}=[ D12(idx,:)]';
    condn_data{13}=[ D13(idx,:)]';
    condn_data{14}=[ D14(idx,:)]';
    condn_data{15}=[ D15(idx,:)]';
    condn_data{16}=[ D16(idx,:)]';
    condn_data{17}=[ D17(idx,:)]';
    condn_data{18}=[ D18(idx,:)]';
    condn_data{19}=[ D19(idx,:)]';
    condn_data{20}=[ D20(idx,:)]';
    condn_data{21}=[ D21(idx,:)]';
    condn_data{22}=[ D22(idx,:)]';
    condn_data{23}=[ D23(idx,:)]';
    condn_data{24}=[ D24(idx,:)]';
    condn_data{25}=[ D25(idx,:)]';
    condn_data{26}=[ D26(idx,:)]';
    condn_data{27}=[ D27(idx,:)]';
    condn_data{28}=[ D28(idx,:)]';
    condn_data{29}=[ D29(idx,:)]';
    condn_data{30}=[ D30(idx,:)]';



    N=[];
    T1=[];
    for i=1:length(condn_data)
        tmp=condn_data{i};
        N = [N tmp'];
        T1 = [T1;i*ones(size(tmp,1),1)];
    end


    T = zeros(size(T1,1),7);
    for i=1:30
        [aa bb]=find(T1==i);
        T(aa(1):aa(end),i)=1;
    end




    %training the model
    clear net
    net = patternnet([64 64 64]) ;
    net.performParam.regularization=0.2;
    net.divideParam.trainRatio=0.8;
    net.divideParam.valRatio=0.2;
    net.divideParam.testRatio=0.0;
    net = train(net,N,T','useParallel','yes');
    %net = train(net,N,T','useParallel','yes');
    % classifier_name = 'MLP_PreTrained_7DoF_Days1to11_924pm2'; % enter the name
    % genFunction(pretrain_net,classifier_name); % make sure to update GetParams


    % now  use the trained  network on the held out trial data
    acc=zeros(30,30);
    for j=1:length(files_test)

        load(files_test{j})
        features  = TrialData.SmoothedNeuralFeatures;
        kinax = [ find(TrialData.TaskState==3)];
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

        % find out which target it belongs to
        idx=[];
        for k=1:length(targets)
            if regexp(targets{k},TrialData.ImaginedAction)
                idx=k;
                break
            end
        end

        % classify the data
        D=temp;
        out=net(D);
        decodes=[];
        for i=1:size(out,2)
            [aa bb] = max(out(:,i));
            decodes = [decodes bb];
        end
        out  = mode(decodes);

        % store results
        acc(idx,out) = acc(idx,out)+1;
    end


    for i=1:size(acc,1)
        acc(i,:)=acc(i,:)./sum(acc(i,:));
    end
    acc_overall(iter,:,:) = acc;
    % figure;imagesc(acc)
    % colormap bone
    % xticks(1:30)
    % xticklabels(targets)
    % yticks(1:30)
    % yticklabels(targets)
    % set(gcf,'Color','w')
    % colorbar
    % set(gca,'FontSize',18)
    % caxis([0 0.65])
    %
    % figure;stem(diag(acc))
    % xticks(1:30)
    % xticklabels(targets)

end

acc=squeeze(mean(acc_overall,1));
figure;imagesc(acc);
caxis([0 0.5])
colormap bone
figure;stem((diag(acc)))
xticks(1:30)
xticklabels(targets)
mean(diag(acc))



%% same as above but trial level classification for B3


% split the data into training and testing trials


clc;clear
% enter the root path from the Data folder


dates = {'20230111','20230118','20230119','20230125','20230126','20230201','20230203'};

%root_path = '/media/reza/WindowsDrive/BRAVO1/CursorPlatform/Data';
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath 'C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim\helpers'
addpath('C:\Users\nikic\Documents\MATLAB')
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')


cd(root_path)


% load the target names from an example file
targets= {'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
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

files=[];
labels={};
for ii=1:length(dates)
    disp(ii/length(dates))

    folderpath = fullfile(root_path, dates{ii},'ImaginedMvmtDAQ');
    tmp = findfiles('mat',folderpath,1)';
    for i=1:length(tmp)
        if isempty(regexp(tmp{i},'kf_params'))
            files=[files;tmp(i)];
        end
    end
end

acc_overall=[];
for iter=1:5
    disp(iter)

    % structure to host all the data
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
    D11=[];
    D12=[];
    D13=[];
    D14=[];
    D15=[];
    D16=[];
    D17=[];
    D18=[];
    D19=[];
    D20=[];
    D21=[];
    D22=[];
    D23=[];
    D24=[];
    D25=[];
    D26=[];
    D27=[];
    D28=[];
    D29=[];
    D30=[];
    D31=[];
    D32=[];





    % split into training and testing files
    idx=randperm(length(files));
    files=files(idx);
    len = round(0.9*length(idx));
    files_train = files(1:len);
    files_test = files(len+1:end);

    % now load the data
    for j=1:length(files_train)

        file_loaded=true;
        try
            load(files_train{j})
        catch
            file_loaded=false;
        end

        if file_loaded
            features  = TrialData.SmoothedNeuralFeatures;
            kinax = [ find(TrialData.TaskState==3)];
            temp = cell2mat(features(kinax));
            

            if size(temp,1)==1792
                temp = temp([257:512 1025:1280 1537:1792],:);
                % remove the bad channels 108, 113 118
                bad_ch = [108 113 118];
                good_ch = ones(size(temp,1),1);
                for ii=1:length(bad_ch)
                    bad_ch_tmp = bad_ch(ii)*[1 2 3];
                    good_ch(bad_ch_tmp)=0;
                end

                temp = temp(logical(good_ch),3:end);                
            else
                temp=[];
            end

            %         % get the pooled data
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

            % find out which target it belongs to
          

            if ~isempty(temp)
                idx=[];
                for k=1:length(targets)
                    if regexp(targets{k},TrialData.ImaginedAction)
                        idx=k;
                        break
                    end
                end

                if idx == 1
                    D1 = [D1 temp];
                elseif idx == 2
                    D2 = [D2 temp];
                elseif idx == 3
                    D3 = [D3 temp];
                elseif idx == 4
                    D4 = [D4 temp];
                elseif idx == 5
                    D5 = [D5 temp];
                elseif idx == 6
                    D6 = [D6 temp];
                elseif idx == 7
                    D7 = [D7 temp];
                elseif idx == 8
                    D8 = [D8 temp];
                elseif idx == 9
                    D9 = [D9 temp];
                elseif idx == 10
                    D10 = [D10 temp];
                elseif idx == 11
                    D11 = [D11 temp];
                elseif idx == 12
                    D12 = [D12 temp];
                elseif idx == 13
                    D13 = [D13 temp];
                elseif idx == 14
                    D14 = [D14 temp];
                elseif idx == 15
                    D15 = [D15 temp];
                elseif idx == 16
                    D16 = [D16 temp];
                elseif idx == 17
                    D17 = [D17 temp];
                elseif idx == 18
                    D18 = [D18 temp];
                elseif idx == 19
                    D19 = [D19 temp];
                elseif idx == 20
                    D20 = [D20 temp];
                elseif idx == 21
                    D21 = [D21 temp];
                elseif idx == 22
                    D22 = [D22 temp];
                elseif idx == 23
                    D23 = [D23 temp];
                elseif idx == 24
                    D24 = [D24 temp];
                elseif idx == 25
                    D25 = [D25 temp];
                elseif idx == 26
                    D26 = [D26 temp];
                elseif idx == 27
                    D27 = [D27 temp];
                elseif idx == 28
                    D28 = [D28 temp];
                elseif idx == 29
                    D29 = [D29 temp];
                elseif idx == 30
                    D30 = [D30 temp];
                elseif idx == 31
                    D31 = [D31 temp];
                elseif idx == 32
                    D32 = [D32 temp];
                end
            end
        end
    end



    clear condn_data
    % combing delta beta and high gamma
    idx=[1:size(D1,1)];
    condn_data{1}=[ D1(idx,:) ]';
    condn_data{2}= [ D2(idx,:)]';
    condn_data{3}=[ D3(idx,:)]';
    condn_data{4}=[ D4(idx,:)]';
    condn_data{5}=[ D5(idx,:)]';
    condn_data{6}=[ D6(idx,:)]';
    condn_data{7}=[ D7(idx,:)]';
    condn_data{8}=[ D8(idx,:)]';
    condn_data{9}=[ D9(idx,:)]';
    condn_data{10}=[ D10(idx,:)]';
    condn_data{11}=[ D11(idx,:)]';
    condn_data{12}=[ D12(idx,:)]';
    condn_data{13}=[ D13(idx,:)]';
    condn_data{14}=[ D14(idx,:)]';
    condn_data{15}=[ D15(idx,:)]';
    condn_data{16}=[ D16(idx,:)]';
    condn_data{17}=[ D17(idx,:)]';
    condn_data{18}=[ D18(idx,:)]';
    condn_data{19}=[ D19(idx,:)]';
%     condn_data{20}=[ D20(idx,:)]';
%     condn_data{21}=[ D21(idx,:)]';
%     condn_data{22}=[ D22(idx,:)]';
%     condn_data{23}=[ D23(idx,:)]';
%     condn_data{24}=[ D24(idx,:)]';
%     condn_data{25}=[ D25(idx,:)]';
%     condn_data{26}=[ D26(idx,:)]';
%     condn_data{27}=[ D27(idx,:)]';
%     condn_data{28}=[ D28(idx,:)]';
%     condn_data{29}=[ D29(idx,:)]';
%     condn_data{30}=[ D30(idx,:)]';
%     condn_data{31}=[ D31(idx,:)]';
%     condn_data{32}=[ D32(idx,:)]';



    N=[];
    T1=[];
    for i=1:length(condn_data)
        tmp=condn_data{i};
        N = [N tmp'];
        T1 = [T1;i*ones(size(tmp,1),1)];
    end


    T = zeros(size(T1,1),length(condn_data));
    for i=1:length(condn_data)
        [aa bb]=find(T1==i);
        T(aa(1):aa(end),i)=1;
    end




    %training the model
    clear net
    net = patternnet([64 64]) ;
    net.performParam.regularization=0.2;
    net.divideParam.trainRatio=0.8;
    net.divideParam.valRatio=0.2;
    net.divideParam.testRatio=0.0;
    %net = train(net,N,T','useGPU','yes');
    net = train(net,N,T','useParallel','yes');
    % classifier_name = 'MLP_PreTrained_7DoF_Days1to11_924pm2'; % enter the name
    % genFunction(pretrain_net,classifier_name); % make sure to update GetParams


    % now  use the trained  network on the held out trial data
    acc=zeros(length(condn_data),length(condn_data));
    for j=1:length(files_test)

        file_loaded = true;
        try
            load(files_test{j})
        catch
            file_loaded=false;
        end

        if file_loaded

            features  = TrialData.SmoothedNeuralFeatures;
            kinax = [ find(TrialData.TaskState==3)];
            temp = cell2mat(features(kinax));

            if size(temp,1)==1792
                temp = temp([257:512 1025:1280 1537:1792],:);
                % remove the bad channels 108, 113 118
                bad_ch = [108 113 118];
                good_ch = ones(size(temp,1),1);
                for ii=1:length(bad_ch)
                    bad_ch_tmp = bad_ch(ii)*[1 2 3];
                    good_ch(bad_ch_tmp)=0;
                end
                
                temp = temp(logical(good_ch),3:end);
            else
                temp=[];
            end

            %         % get the pooled data
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

            % find out which target it belongs to
            if ~isempty(temp)
                idx=[];
                for k=1:length(targets)
                    if regexp(targets{k},TrialData.ImaginedAction)
                        idx=k;
                        break
                    end
                end

                if idx<=19

                    % classify the data
                    D=temp;
                    out=net(D);
                    decodes=[];
                    for i=1:size(out,2)
                        [aa bb] = max(out(:,i));
                        decodes = [decodes bb];
                    end
                    out  = mode(decodes);

                    % store results
                    acc(idx,out) = acc(idx,out)+1;
                end
            end
        end
    end


    for i=1:size(acc,1)
        acc(i,:)=acc(i,:)./sum(acc(i,:));
    end
    acc_overall(iter,:,:) = acc;
    % figure;imagesc(acc)
    % colormap bone
    % xticks(1:30)
    % xticklabels(targets)
    % yticks(1:30)
    % yticklabels(targets)
    % set(gcf,'Color','w')
    % colorbar
    % set(gca,'FontSize',18)
    % caxis([0 0.65])
    %
    % figure;stem(diag(acc))
    % xticks(1:30)
    % xticklabels(targets)

end

% 
% save trial_level_classification_B3 -v7.3

acc=squeeze(nanmean(acc_overall,1));
figure;imagesc(acc);
caxis([0 0.8])
xticks(1:32)
xticklabels(targets)
yticks(1:32)
yticklabels(targets)
set(gcf,'Color','w')
colormap bone
figure;stem((diag(acc)),'LineWidth',1)
xticks(1:32)
xticklabels(targets)
hline(1/32,'r')
set(gcf,'Color','w')
ylabel('Accuracy')
title('Trial-level accuracy')
axis tight
ylim([0 1])
xlim([0 33])

nanmean(diag(acc))

%plotting for individual actions
save_path='F:\DATA\ecog data\ECoG BCI\Results\B3 results\ImaginedMvmt\Indiv';
for idx=1:length(targets)
    figure;stem(acc(idx,:),'LineWidth',1)
    title(['Classification for ' targets(idx)])
    set(gcf,'Color','w')
    %set(gcf,'WindowState','maximized')
    set(gcf,'Position',[680,379,1057,599])
    ylabel('Accuracy')
    xticks(1:32)
    xticklabels(targets)
    xlim([0 33])
    set(gca,'FontSize',14)
    ylim([0 1])

    filename = fullfile(save_path,targets{idx});
    %saveas(gcf,filename)
    set(gcf,'PaperPositionMode','auto')
    print(gcf,filename,'-dpng','-r300')
    close
end

%% LOOKING AT MAHALANOBIS DISTANCE BETWEEN THE HAND TRIALS IN THE DATA
% append all the trials together and get mean and variance across all time
% in stage 3 of the task


clc;clear
% enter the root path from the Data folder


dates = {'20211201','20211203','20211206','20211208','20211215'};
%20210917 has issues with the way the order of targets were flipped, need
%to manually input the correct folders for this particular date


root_path = '/media/reza/WindowsDrive/BRAVO1/CursorPlatform/Data';

cd(root_path)

%FOR IMAGINED MOVEMENT DATA

% structure to host all the data

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
D11=[];
D12=[];
D13=[];
D14=[];
D15=[];
D16=[];
D17=[];
D18=[];
D19=[];
D20=[];
D21=[];
D22=[];
D23=[];
D24=[];
D25=[];
D26=[];
D27=[];
D28=[];
D29=[];
D30=[];



% load the target names from an example file
targets={'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
    'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
    'Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp',...
    'Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
    'Squeeze Both Hands',...
    'Rotate Right Wrist','Rotate Left Wrist',...
    'Imagined Head Movement',...
    'Right Shoulder Shrug',...
    'Left Shoulder Shrug',...
    'Right Bicep','Left Bicep',...
    'Right Tricep','Left Tricep',...
    'Right Leg','Left Leg',...
    'Lips','Tongue'};

for ii=1:length(dates)
    disp(ii/length(dates))

    folderpath = fullfile(root_path, dates{ii},'ImaginedMvmtDAQ');
    tmp = findfiles('mat',folderpath,1)';
    files=[];
    for i=1:length(tmp)
        if regexp(tmp{i},'Imagined')
            files=[files;tmp(i)];
        end
    end

    % now load the data
    for j=1:length(files)

        load(files{j})
        features  = TrialData.SmoothedNeuralFeatures;
        kinax = [ find(TrialData.TaskState==3)];
        temp = cell2mat(features(kinax));

        % get only the delta band data
        temp = temp(769:end,:);

        % find out which target it belongs to
        idx=[];
        for k=1:length(targets)
            if regexp(targets{k},TrialData.ImaginedAction)
                idx=k;
                break
            end
        end

        if idx == 1
            D1 = [D1 temp];
        elseif idx == 2
            D2 = [D2 temp];
        elseif idx == 3
            D3 = [D3 temp];
        elseif idx == 4
            D4 = [D4 temp];
        elseif idx == 5
            D5 = [D5 temp];
        elseif idx == 6
            D6 = [D6 temp];
        elseif idx == 7
            D7 = [D7 temp];
        elseif idx == 8
            D8 = [D8 temp];
        elseif idx == 9
            D9 = [D9 temp];
        elseif idx == 10
            D10 = [D10 temp];
        elseif idx == 11
            D11 = [D11 temp];
        elseif idx == 12
            D12 = [D12 temp];
        elseif idx == 13
            D13 = [D13 temp];
        elseif idx == 14
            D14 = [D14 temp];
        elseif idx == 15
            D15 = [D15 temp];
        elseif idx == 16
            D16 = [D16 temp];
        elseif idx == 17
            D17 = [D17 temp];
        elseif idx == 18
            D18 = [D18 temp];
        elseif idx == 19
            D19 = [D19 temp];
        elseif idx == 20
            D20 = [D20 temp];
        elseif idx == 21
            D21 = [D21 temp];
        elseif idx == 22
            D22 = [D22 temp];
        elseif idx == 23
            D23 = [D23 temp];
        elseif idx == 24
            D24 = [D24 temp];
        elseif idx == 25
            D25 = [D25 temp];
        elseif idx == 26
            D26 = [D26 temp];
        elseif idx == 27
            D27 = [D27 temp];
        elseif idx == 28
            D28 = [D28 temp];
        elseif idx == 29
            D29 = [D29 temp];
        elseif idx == 30
            D30 = [D30 temp];
        end
    end
end

% compute mahalonobis distance between subjects especially for hand
clear D
D{1}=D1;
D{end+1}=D2;
D{end+1}=D3;
D{end+1}=D4;
D{end+1}=D5;
D{end+1}=D6;
D{end+1}=D7;
D{end+1}=D8;
D{end+1}=D9;
D{end+1}=D10;
D{end+1}=D11;
D{end+1}=D12;
D{end+1}=D13;
D{end+1}=D14;
D{end+1}=D15;
D{end+1}=D16;
D{end+1}=D17;
D{end+1}=D18;
D{end+1}=D19;

mahal_dist=zeros(size(D,2));
for i=1:size(D,2)
    a=D{i};
    for j=i+1:size(D,2)
        b=D{j};
        mahal_dist(i,j) = mahal2(a',b',2);
        mahal_dist(j,i) =  mahal_dist(i,j);
    end
end
figure;imagesc(mahal_dist)
caxis([0 15])
xticks(1:19)
xticklabels(targets(1:19))
yticks(1:19)
yticklabels(targets(1:19))
set(gcf,'Color','w')
set(gca,'FontSize',12)
title('Distance between movements centroids')
colormap hot

Z = linkage(mahal_dist,'average');
figure;
H=dendrogram(Z);
x = get(gca,'xticklabels');
x=string(x);
lab = x;
for i=1:length(x)
    tmp = str2num(x(i));
    lab(i) = targets(tmp);
end
lab = char(lab);
set(gca,'xticklabels',lab)
set(gcf,'Color','w')
set(gca,'FontSize',12)
title('Hierarchical clustering')

