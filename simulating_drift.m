% simulating drift

clc;clear
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
addpath 'C:\Users\nikic\Documents\MATLAB'

% create 4 classes of 2D gaussian distribution

Sigma = [0.5 0.25; 0.25 0.5];
a = mvnrnd([2 2]',Sigma,100);
b = mvnrnd([0 0]',Sigma,100);
c = mvnrnd([0 2]',Sigma,100);
d = mvnrnd([2 0]',Sigma,100);

figure;plot(a(:,1),a(:,2),'.','MarkerSize',20)
hold on
plot(b(:,1),b(:,2),'.','MarkerSize',20)
plot(c(:,1),c(:,2),'.','MarkerSize',20)
plot(d(:,1),d(:,2),'.','MarkerSize',20)

data=[a;b;c;d];
idx = [ones(100,1);2*ones(100,1);3*ones(100,1);4*ones(100,1)];
num_classes =length(unique(idx));
condn_data_overall={};
for i=1:length(idx)
    condn_data_overall(i).neural = data(i,:);
    condn_data_overall(i).targetID = idx(i,:);
end

% acc=[];
% for i=1:5
%     acc(i) = run_classifer_simulation(condn_data_overall);
% end
% cv_perf=mean(acc)

%%%% CASE 1 now add a sysmetatic drift and see how it changes acrsos days
drift = 4.5*rand(1,2); % add this to each day
data_old = data;
idx_old  = idx;
across_day_data={};
across_day_data{1} = data_old;
cmap=parula(10);
figure;hold on
for days=2:10
     drift = 4.5*rand(1,2); % add this to each day
     data_new = data_old + (drift);
     %data_new = data_old + ((days-1) * drift);
     across_day_data{days} = data_new;
     plot(data_new(1:100,1),data_new(1:100,2),'.','MarkerSize',20,'Color',cmap(days,:))
     plot(data_new(101:200,1),data_new(101:200,2),'o','MarkerSize',20,'Color',cmap(days,:))
     plot(data_new(201:300,1),data_new(201:300,2),'+','MarkerSize',20,'Color',cmap(days,:))
     plot(data_new(301:400,1),data_new(301:400,2),'x','MarkerSize',20,'Color',cmap(days,:))
end

%train a classifier on increasing number of days and see how it generalizes
across_day_acc={};
for days=1:10-1
    training_days=1:days;
    testing_days = days+1:10;

    data_new=[];idx_new=[];
    for i=1:length(training_days)
        tmp = across_day_data{training_days(i)};
        data_new = [data_new;tmp];
        idx_new = [idx_new;idx_old];
    end

    condn_data_overall={};
    for i=1:length(idx_new)
        condn_data_overall(i).neural = data_new(i,:);
        condn_data_overall(i).targetID = idx_new(i,:);
    end

    % train the network
    [~,~,net]= run_classifer_simulation(condn_data_overall,0);

    % get held out days and apply neural network    
    acc=[];
    for ii=1:length(testing_days)
        tmp = across_day_data{testing_days(ii)};
        XTest = [tmp];
        YTest = [idx_old];
        YTest = categorical(YTest);

        % apply network on held out days
        out=predict(net,XTest);
        decodes=[];
        conf_matrix=zeros(num_classes);
        for i=1:size(out,1)
            [aa bb]=max(out(i,:));
            decodes=[decodes;bb];
            conf_matrix(double(YTest(i)),bb)=conf_matrix(double(YTest(i)),bb)+1;
        end
        decodes=categorical(decodes);
        cv_perf = sum(decodes==YTest)/length(decodes);
        for i=1:size(conf_matrix,2)
            conf_matrix(i,:) = conf_matrix(i,:)./sum(conf_matrix(i,:));
        end
        acc = [acc cv_perf];
    end

    across_day_acc{days} = acc;
    median(acc)
    
end

figure;hold on
for i=1:length(across_day_acc)
    tmp = across_day_acc{i};
    I = i*ones(length(tmp),1);
    scatter(I,tmp,50)
    plot(i,median(tmp),'.r','MarkerSize',30)
end
ylim([0 1])

