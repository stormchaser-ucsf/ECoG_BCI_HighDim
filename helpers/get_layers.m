function layers = get_layers(j,k,l,input_size)



% using custom layers



layers = [ ...
    featureInputLayer(input_size)    
    fullyConnectedLayer(j)    
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(k)    
    reluLayer    
    dropoutLayer(0.3)
    fullyConnectedLayer(l)    
    reluLayer    
    dropoutLayer(0.3)
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer
    ];


end