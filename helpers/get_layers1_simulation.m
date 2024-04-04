function layers = get_layers1_simulation(j,input_size,num_classes)
%function layers = get_layers1_simulation(j,input_size,num_classes)

if nargin<3
    num_classes=7;
end


% using custom layers
layers = [ ...
    featureInputLayer(input_size)        
    dropoutLayer(0.4)
    fullyConnectedLayer(j)            
    leakyReluLayer
    dropoutLayer(0.4)
    fullyConnectedLayer(num_classes)
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