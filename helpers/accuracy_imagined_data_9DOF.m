function [acc,train_permutations] = accuracy_imagined_data_9DOF(condn_data, iterations)

num_trials = length(condn_data);
train_permutations = zeros(num_trials,iterations)';
acc_permutations=[];
for iter = 1:iterations % loop over 20 times
    train_idx = randperm(num_trials,round(0.8*num_trials));
    test_idx = ones(num_trials,1);
    test_idx(train_idx) = 0;
    test_idx = find(test_idx==1);
    train_permutations(iter,train_idx) = train_permutations(iter,train_idx)+1;

    % build a MLP from the training data
    train_data = condn_data(train_idx);
    test_data = condn_data(test_idx);
    D1=[];
    D2=[];
    D3=[];
    D4=[];
    D5=[];
    D6=[];
    D7=[];
    D8=[];
    D9=[];
    for i=1:length(train_data)
        temp = train_data(i).neural;
        if train_data(i).targetID == 1
            D1 = [D1 temp];
        elseif train_data(i).targetID == 2
            D2 = [D2 temp];
        elseif train_data(i).targetID == 3
            D3 = [D3 temp];
        elseif train_data(i).targetID == 4
            D4 = [D4 temp];
        elseif train_data(i).targetID == 5
            D5 = [D5 temp];
        elseif train_data(i).targetID == 6
            D6 = [D6 temp];
        elseif train_data(i).targetID == 7
            D7 = [D7 temp];
        elseif train_data(i).targetID == 8
            D8 = [D8 temp];
        elseif train_data(i).targetID == 9
            D9 = [D9 temp];
        end
    end
    A = D1';
    B = D2';
    C = D3';
    D = D4';
    E = D5';
    F = D6';
    G = D7';
    H = D8';
    I = D9';

    % organize data
    clear N
    N = [A' B' C' D' E' F' G' H' I'];
    T1 = [ones(size(A,1),1);2*ones(size(B,1),1);3*ones(size(C,1),1);4*ones(size(D,1),1);...
        5*ones(size(E,1),1);6*ones(size(F,1),1);7*ones(size(G,1),1);8*ones(size(H,1),1);...
        9*ones(size(I,1),1)];
    T = zeros(size(T1,1),9);
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
    [aa bb]=find(T1==8);
    T(aa(1):aa(end),8)=1;
    [aa bb]=find(T1==9);
    T(aa(1):aa(end),9)=1;

    % train MLP
    net = patternnet([64 64 64 ]) ;
    net.performParam.regularization=0.2;
    net = train(net,N,T');

    % test it out on the held out trials using a mode filter
    acc = zeros(9);
    for i=1:length(test_data)
        features = test_data(i).neural;
        if ~isempty(features)
            out = net(features);
            out(out<0.4)=0; % thresholding
            [prob,idx] = max(out); % getting the decodes
            decodes = mode_filter(idx); % running it through a 5 sample mode filter
            decodes_sum=[];
            for ii=1:7
                decodes_sum(ii) = sum(decodes==ii);
            end
            [aa bb]=max(decodes_sum);
            acc(test_data(i).targetID,bb) = acc(test_data(i).targetID,bb)+1;
        end
    end
    for i=1:size(acc,1)
        acc1(i,:) = acc(i,:)/sum(acc(i,:));
    end
    %acc1
    acc_permutations(iter,:,:) = acc1;
end
acc=acc_permutations;
end

