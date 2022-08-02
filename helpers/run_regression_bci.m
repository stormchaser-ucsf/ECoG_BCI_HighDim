function [r2]=run_regression_bci(neural_data_a,kin_data_a,dim)



% split into training and testing -> single action 
overall_r2=[];
parfor iter=1:100
    %disp(iter)
    train_idx=  randperm(length(neural_data_a),round(0.7*length(neural_data_a)));
    test_idx = ones(length(neural_data_a),1);
    test_idx(train_idx)=0;
    test_idx=logical(test_idx);
    train_neural = neural_data_a(train_idx);
    train_kin = kin_data_a(train_idx);
    test_neural = neural_data_a(test_idx);
    test_kin = kin_data_a(test_idx);

    % build the regression model
    train_neural = cell2mat(train_neural');
    train_kin = cell2mat(train_kin');
    train_kin=train_kin(:,1);
    [c,s,l]=pca(train_neural(:,1:128));
    [c1,s1,l1]=pca(train_neural(:,129:256));
    %[c2,s2,l2]=pca(train_neural(:,257:384));
    train_neural = [s(:,1:dim) s1(:,1:dim) ];

    % regression
    x=train_neural;
    y=train_kin;
    for j=1:size(y,2)
        y(:,j)=smooth(y(:,j),100);
    end
    y=y-mean(y);
    % OLS or ridge or weiner etc
    bhat = pinv(x)*y;
    %bhat = (x'*x  + 0.01*eye(size(x,2)))\(x'*y);
    yhat = x*bhat;

    % test it out on held out data
    test_neural = cell2mat(test_neural');
    %[c,s,l]=pca(test_neural(:,1:128));
    %[c1,s1,l1]=pca(test_neural(:,129:256));
    s=test_neural(:,1:128)*c(:,1:dim);
    s1=test_neural(:,129:256)*c1(:,1:dim);
    %s2=test_neural(:,257:384)*c2(:,1:dim);
    test_neural = [s(:,1:dim) s1(:,1:dim) ];
    test_kin = cell2mat(test_kin');
    test_kin=test_kin(:,1);
    x=test_neural;
    y=test_kin;
    for j=1:size(y,2)
       y(:,j)=smooth(y(:,j),100);
    end
    y=y-mean(y);
    % get prediction
    yhat = x*bhat;

    % looking at r2
    %figure;plot(y(:,1));
    %hold on
    %plot(yhat(:,1))
    [r,p]=corrcoef(y,yhat);
    overall_r2(iter)=r(1,2);
end
%figure;boxplot(overall_r2)
r2=overall_r2;



end

