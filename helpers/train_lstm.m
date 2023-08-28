function net = train_lstm(XTrain,XTest,YTrain,YTest,numHiddenUnits1,drop1,num_classes)






% specify lstm structure
inputSize = size(XTrain{1},1);
%numHiddenUnits1 = [  90 120 250 128 325];
%drop1 = [ 0.2 0.2 0.3  0.3 0.4];
numClasses = num_classes;
%for i=3%1:length(drop1)
numHiddenUnits=numHiddenUnits1;
drop=drop1;
layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','sequence','Name','lstm_1')
    dropoutLayer(drop)
    layerNormalizationLayer
    gruLayer(round(numHiddenUnits/2),'OutputMode','last','Name','lstm_2')
    dropoutLayer(drop)
    %layerNormalizationLayer
    fullyConnectedLayer(25)
    leakyReluLayer
    batchNormalizationLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];



% options
batch_size=64;
val_freq = floor(length(XTrain)/batch_size);
options = trainingOptions('adam', ...
    'MaxEpochs',140, ...
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
    'LearnRateDropPeriod',75,...
    'InitialLearnRate',0.001);

% train the model
net = trainNetwork(XTrain,YTrain,layers,options);

%end


end

