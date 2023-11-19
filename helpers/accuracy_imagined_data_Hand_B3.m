function [acc,train_permutations] = accuracy_imagined_data_Hand_B3(condn_data, iterations)

num_trials = length(condn_data);
train_permutations = zeros(num_trials,iterations)';
acc_permutations=[];
for iter = 1:iterations % loop over 20 times
    train_idx = randperm(num_trials,round(0.75*num_trials));
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
    D10=[];
    D11=[];
    D12=[];
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
        elseif train_data(i).targetID == 10
            D10 = [D10 temp];
        elseif train_data(i).targetID == 11
            D11 = [D11 temp];
        elseif train_data(i).targetID == 12
            D12 = [D12 temp];
        end
    end


    clear condn_data1
    idx=1;
    condn_data1{1}=[D1(idx:end,:) ]';
    condn_data1{2}= [D2(idx:end,:)]';
    condn_data1{3}=[D3(idx:end,:)]';
    condn_data1{4}=[D4(idx:end,:)]';
    condn_data1{5}=[D5(idx:end,:)]';
    condn_data1{6}=[D6(idx:end,:)]';
    condn_data1{7}=[D7(idx:end,:)]';
    condn_data1{8}=[D8(idx:end,:)]';
    condn_data1{9}=[D9(idx:end,:)]';
    condn_data1{10}=[D10(idx:end,:)]';
    condn_data1{11}=[D11(idx:end,:)]';
    condn_data1{12}=[D12(idx:end,:)]';


    N=[];
    T1=[];
    for i=1:length(condn_data1)
        tmp=condn_data1{i};
        N = [N tmp'];
        T1 = [T1;i*ones(size(tmp,1),1)];
    end


    T = zeros(size(T1,1),length(condn_data1));
    for i=1:length(condn_data1)
        [aa bb]=find(T1==i);
        T(aa(1):aa(end),i)=1;
    end

    % train MLP
    net = patternnet([120]) ;
    net.performParam.regularization=0.2;
    net = train(net,N,T','UseParallel','yes');

    % test it out on the held out trials using a mode filter
    acc = zeros(length(condn_data1));
    for i=1:length(test_data)
        features = test_data(i).neural;
        if ~isempty(features) && test_data(i).targetID<=length(condn_data1)
            out = net(features);
            out(out<0.4)=0; % thresholding
            [prob,idx] = max(out); % getting the decodes
            decodes = mode_filter(idx,length(condn_data1)); % running it through a 5 sample mode filter
            decodes_sum=[];
            for ii=1:length(condn_data1)
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

