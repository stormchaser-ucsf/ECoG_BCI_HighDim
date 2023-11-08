function layers = get_layers1(j,input_size)



% using custom layers
layers = [ ...
    featureInputLayer(input_size)        
    fullyConnectedLayer(j)    
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer
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