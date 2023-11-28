function layers = get_layers(j,k,l,input_size)



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
    fullyConnectedLayer(l)    
    leakyReluLayer    
    dropoutLayer(0.4)
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer
    ];


end