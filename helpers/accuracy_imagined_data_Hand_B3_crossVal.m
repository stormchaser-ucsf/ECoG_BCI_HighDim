function [acc,train_permutations] = accuracy_imagined_data_Hand_B3_crossVal(condn_data, iterations)

num_trials = length(condn_data);
train_permutations = zeros(num_trials,iterations)';
acc_permutations=[];
for iter = 1:iterations % loop over 20 times


    % split into training and testing trials, 60% train, 20% test, 20% val 
    xx=1;xx1=1;xx2=1;yy=0;
    while xx<12 || xx1<12 || xx2<12
        prop = 0.2;
        test_idx = randperm(length(condn_data),round(prop*length(condn_data)));
        test_idx=test_idx(:);
        I = ones(length(condn_data),1);
        I(test_idx)=0;
        train_val_idx = find(I~=0);
        prop1 = (0.6);
        tmp_idx = randperm(length(train_val_idx),floor(prop1*length(condn_data)));
        train_idx = train_val_idx(tmp_idx);train_idx=train_idx(:);
        I = ones(length(condn_data),1);
        I([train_idx;test_idx])=0;
        val_idx = find(I~=0);val_idx=val_idx(:);
        xx = length(unique([condn_data(train_idx).targetID]));
        xx1 = length(unique([condn_data(val_idx).targetID]));
        xx2 = length(unique([condn_data(test_idx).targetID]));
        yy=yy+1;
    end    
    
    train_data = condn_data(train_idx);
    val_data = condn_data(val_idx);
    test_data = condn_data(test_idx);

    % get training and validation data samples
    D={};D_val={};
    for i=1:(num_classes)
        idx = find([train_data(1:end).targetID]==i);
        idx_val = find([val_data(1:end).targetID]==i);

        tmp=[];
        for j=1:length(idx)
            tmp = [tmp train_data(idx(j)).neural];            
        end

        tmp_val=[];
        for j=1:length(idx_val)
            tmp_val = [tmp_val val_data(idx_val(j)).neural];
        end

        D{i}=tmp';
        D_val{i} = tmp_val';
    end
    
    % arrange data
    condn_data1=D;
    N=[];
    T1=[];
    for i=1:length(condn_data1)
        tmp=condn_data1{i};
        N = [N tmp'];
        T1 = [T1;i*ones(size(tmp,1),1)];
    end
    train_samples = size(N,2);

    condn_data1=D_val; xx=0;   
    for i=1:length(condn_data1)
        tmp=condn_data1{i};
        xx=xx+size(tmp,1);
        N = [N tmp'];
        T1 = [T1;i*ones(size(tmp,1),1)];
    end

    %arrange idx
    T = zeros(size(T1,1),length(condn_data1));
    for i=1:length(condn_data1)
        [aa bb]=find(T1==i);
        T(aa,i)=1;
    end    

    % train MLP, based on randomly sorted index
    net = patternnet([120 ]) ;
    net.performParam.regularization=0.2;
    net.divideFcn='divideind';
    net.divideParam.trainInd = 1:train_samples;
    net.divideParam.valInd = train_samples+1:size(T,1);
    %net.divideParam.testRatio = 0;
    net.trainParam.showWindow = 1;
    net = train(net,N,T');

    % test it out on the held out trials using a mode filter
    acc = zeros(length(condn_data1));
    for i=1:length(test_data)
        features = test_data(i).neural;
        if ~isempty(features) && test_data(i).targetID<=length(condn_data1)
            out = net(features);
            out(out<0.4)=0; % thresholding
            [prob,idx] = max(out); % getting the decodes
            decodes=idx;
            %decodes = mode_filter(idx,length(condn_data1)); % running it through a 5 sample mode filter
            decodes_sum=[];
            for ii=1:length(condn_data1)
                decodes_sum(ii) = sum(decodes==ii);
            end
            [aa bb]=max(decodes_sum);

            if sum(aa==decodes_sum)==1
                acc(test_data(i).targetID,bb) = acc(test_data(i).targetID,bb)+1; % trial level
            else
                disp(['error trial ' num2str(i)])
                xx=mean(out,2);
                [aa bb]=max(xx);
                acc(test_data(i).targetID,bb) = acc(test_data(i).targetID,bb)+1; % trial level
            end



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

