function [N,T,condn_data] = get_training_samples_mlp(condn_data_overall,train_idx)
%function [N,T,condn_data] = get_training_samples_mlp(condn_data_overall,train_idx)


D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
condn_data_overall_tmp =  condn_data_overall(train_idx);

for i=1:length(condn_data_overall_tmp)

    temp  = condn_data_overall_tmp(i).neural;

    % 2-norm
    for ii=1:size(temp,2)
        temp(:,ii) = temp(:,ii)./norm(temp(:,ii));
    end

    TrialData.TargetID = condn_data_overall_tmp(i).targetID;

    if TrialData.TargetID == 1
        D1 = [D1 temp];
    elseif TrialData.TargetID == 2
        D2 = [D2 temp];
    elseif TrialData.TargetID == 3
        D3 = [D3 temp];
    elseif TrialData.TargetID == 4
        D4 = [D4 temp];
    elseif TrialData.TargetID == 5
        D5 = [D5 temp];
    elseif TrialData.TargetID == 6
        D6 = [D6 temp];
    elseif TrialData.TargetID == 7
        D7 = [D7 temp];

    end

end




% build patternet
clear condn_data
idx=1;
condn_data{1}=[D1(idx:end,:) ]'; % right thumb
condn_data{2}= [D2(idx:end,:)]'; % both feet
condn_data{3}=[D3(idx:end,:)]'; % left pinch
condn_data{4}=[D4(idx:end,:)]'; % head
condn_data{5}=[D5(idx:end,:)]'; % lips
condn_data{6}=[D6(idx:end,:)]'; % tong
condn_data{7}=[D7(idx:end,:)]'; % both hands


A = condn_data{1};
B = condn_data{2};
C = condn_data{3};
D = condn_data{4};
E= condn_data{5};
F= condn_data{6};
G= condn_data{7};


clear N
N = [A' B' C' D' E' F' G'];
T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
    5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1)];
T = zeros(size(T1,1),7);
[aa bb]=find(T1==1);[aa(1) aa(end)];
T(aa(1):aa(end),1)=1;
[aa bb]=find(T1==2);[aa(1) aa(end)];
T(aa(1):aa(end),2)=1;
[aa bb]=find(T1==3);[aa(1) aa(end)];
T(aa(1):aa(end),3)=1;
[aa bb]=find(T1==4);[aa(1) aa(end)];
T(aa(1):aa(end),4)=1;
[aa bb]=find(T1==5);[aa(1) aa(end)];
T(aa(1):aa(end),5)=1;
[aa bb]=find(T1==6);[aa(1) aa(end)];
T(aa(1):aa(end),6)=1;
[aa bb]=find(T1==7);[aa(1) aa(end)];
T(aa(1):aa(end),7)=1;



