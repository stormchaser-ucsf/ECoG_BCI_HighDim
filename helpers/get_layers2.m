function layers = get_layers2(j,k,input_size)



% using custom layers



layers = [ ...
    featureInputLayer(input_size)
    dropoutLayer(0.4)
    fullyConnectedLayer(j)
    leakyReluLayer
    dropoutLayer(0.4)
    fullyConnectedLayer(k)
    leakyReluLayer
    dropoutLayer(0.4)
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer
    ];


end




