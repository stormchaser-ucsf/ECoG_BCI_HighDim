function layers = get_layers2(j,k,input_size)



% using custom layers



layers = [ ...
    featureInputLayer(input_size)    
    fullyConnectedLayer(j)    
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(k)    
    reluLayer    
    dropoutLayer(0.3)    
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer
    ];


end