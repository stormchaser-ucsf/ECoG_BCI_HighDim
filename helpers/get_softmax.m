function [Y,labels] = get_softmax(condn_data,decoder_name)
%function [Y,labels] = get_softmax(condn_data,decoder_name)

Y=[];labels=[];
for i=1:length(condn_data)
   X = condn_data{i};
   Decision_Prob = feval(decoder_name,X');
   Y = [Y  Decision_Prob];
   labels = [labels ;i*ones(size(Decision_Prob,2),1)];    
end

end
