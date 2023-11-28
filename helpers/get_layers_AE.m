function layers = get_layers_AE(j,k,input_size)



% using custom layers
layers = [ ...
    featureInputLayer(input_size)        
    

    fullyConnectedLayer(j)            
    leakyReluLayer
    dropoutLayer(0.4)
    fullyConnectedLayer(k)
    leakyReluLayer
    

    fullyConnectedLayer(j)            
    leakyReluLayer
    dropoutLayer(0.4)
    fullyConnectedLayer(input_size)        
    regressionLayer    
    ];
end

% 
% 
% layers = [ ...
%     featureInputLayer(input_size)    
%     fullyConnectedLayer(7)
%     softmaxLayer
%     classificationLayer
%     ];