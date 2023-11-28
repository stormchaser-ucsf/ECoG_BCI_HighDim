function layers = get_layers0(input_size)



% using custom layers
layers = [ ...
    featureInputLayer(input_size)   
    dropoutLayer(0.4)
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