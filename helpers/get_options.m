function [options,XTrain,YTrain] = get_options(condn_data_overall,val_idx,train_idx,lr)

if nargin<4
    lr=0.001;
end

XTest=[];
YTest=[];
for i=1:length(val_idx)
    tmp=condn_data_overall(val_idx(i)).neural;
    %tmp=condn_data_overall(val_idx(i)).neural;
    if ~isempty(tmp)
        XTest = [XTest;tmp'];
        tmp1 = condn_data_overall(val_idx(i)).targetID;
        YTest = [YTest;repmat(tmp1,size(tmp,2),1)];
    end
end
YTest=categorical((YTest));

XTrain=[];
YTrain=[];
for i=1:length(train_idx)
    tmp=condn_data_overall(train_idx(i)).neural;
    if ~isempty(tmp)
        XTrain = [XTrain;tmp'];
        tmp1 = condn_data_overall(train_idx(i)).targetID;
        YTrain = [YTrain;repmat(tmp1,size(tmp,2),1)];
    end
end
YTrain=categorical((YTrain));

batch_size=32;
%val_freq = floor((9/10)*length(XTrain)/batch_size);
val_freq = floor(length(XTrain)/batch_size);
options = trainingOptions('adam', ...
    'MaxEpochs',125, ...
    'MiniBatchSize',batch_size, ...
    'GradientThreshold',10, ...
    'Verbose',true, ...
    'ValidationFrequency',val_freq,...
    'Shuffle','every-epoch', ...
    'ValidationData',{XTest,YTest},...
    'ValidationPatience',6,...
    'Plots','training-progress',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'OutputNetwork','best-validation-loss',...
    'LearnRateDropPeriod',50,...
    'InitialLearnRate',lr,...
    'Plots','none');
end

%'Plots','training-progress'