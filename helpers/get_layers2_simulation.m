function layers = get_layers2_simulation(j,k,input_size,num_classes)
%function layers = get_layers2_simulation(j,k,input_size,num_classes)



if nargin<4
    num_classes=7;
end


% using custom layers
layers = [ ...
    featureInputLayer(input_size)
    dropoutLayer(0.3)
    fullyConnectedLayer(j)
    leakyReluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(k)
    leakyReluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(num_classes)
    softmaxLayer
    classificationLayer
    ];


end




