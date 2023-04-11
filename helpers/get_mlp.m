function net = get_mlp(condn_data,node_size)
% function net = get_mlp(condn_data)
% ASSUMED 7DOF DATA

% 2-norm the data
for i=1:length(condn_data)
   tmp = condn_data{i}; 
   for j=1:size(tmp,1)
       tmp(j,:) = tmp(j,:)./norm(tmp(j,:));
   end
   condn_data{i}=tmp;
end



A = condn_data{1};
B = condn_data{2};
C = condn_data{2};
D = condn_data{4};
E = condn_data{5};
F = condn_data{6};
G = condn_data{7};



% organize data
clear N
N = [A' B' C' D' E' F' G'];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1)];
T = zeros(size(T1,1),7);
[aa bb]=find(T1==1);
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);
T(aa(1):aa(end),4)=1;
[aa bb]=find(T1==5);
T(aa(1):aa(end),5)=1;
[aa bb]=find(T1==6);
T(aa(1):aa(end),6)=1;
[aa bb]=find(T1==7);
T(aa(1):aa(end),7)=1;

% train MLP
net = patternnet([node_size node_size node_size ]) ;
net.performParam.regularization=0.2;
net = train(net,N,T','UseGPU','yes');


end