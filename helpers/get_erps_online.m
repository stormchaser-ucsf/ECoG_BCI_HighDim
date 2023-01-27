function [erps,sig_ch] = get_erps_online(files)



% load the data for each target
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
for i=1:length(files)
    disp(i)
    load(files{i});
    features  = TrialData.SmoothedNeuralFeatures;
    features = cell2mat(features);
    features = features(769:end,:); % hG
    %features = features(513:640,:); %beta
    fs = TrialData.Params.UpdateRate;
    kinax = TrialData.TaskState;
    state1 = find(kinax==1);
    state2 = find(kinax==2);
    state3 = find(kinax==3);
    state4 = find(kinax==4);
    tmp_data = features(:,state3);
    idx1= ones(length(state1),1);
    idx2= 2*ones(length(state2),1);
    idx3= 3*ones(length(state3),1);
    idx4= 4*ones(length(state4),1);

    % interpolate
    tb = (1/fs)*[1:size(tmp_data,2)];
    t=(1/fs)*[1:25];
    tb = tb*t(end)/tb(end);
    tmp_data1 = interp1(tb,tmp_data',t,'spline')';
    idx3 = interp1(tb,idx3,t,'spline');

    % now stick all the data together
    data = [features(:,[state1 state2]) tmp_data1 features(:,[state4])];

    % now get the ERPs
    if TrialData.TargetID == 1
        D1 = cat(3,D1,data);
    elseif TrialData.TargetID == 2
        D2 = cat(3,D2,data);
    elseif TrialData.TargetID == 3
        D3 = cat(3,D3,data);
    elseif TrialData.TargetID == 4
        D4 = cat(3,D4,data);
    elseif TrialData.TargetID == 5
        D5 = cat(3,D5,data);
    elseif TrialData.TargetID == 6
        D6 = cat(3,D6,data);
    elseif TrialData.TargetID == 7
        D7 = cat(3,D7,data);
    end
end

D{1} = D1;
D{2} = D2;
D{3} = D3;
D{4} = D4;
D{5} = D5;
D{6} = D6;
D{7} = D7;
chMap=TrialData.Params.ChMap;
sig_ch = zeros([size(chMap),7]);
for ii=1:length(D)

    % plot the ERPs with bootstrapped C.I. shading    
    figure
    ha=tight_subplot(8,16);
    d = 1;
    set(gcf,'Color','w')
    tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);    
    sgtitle(num2str(ii))
    tmp_D = D{ii};
    for i = 1:size(tmp_D,1)
        [x y] = find(chMap==i);
        if x == 1
            axes(ha(y));
            %subplot(8, 16, y)
        else
            s = 16*(x-1) + y;
            axes(ha(s));
            %subplot(8, 16, s)
        end
        hold on
        erps =  squeeze(tmp_D(i,:,:));

        chdata = erps;
        % zscore the data to the first 5 time-bins
        tmp_data=chdata(1:5,:);
        m = mean(tmp_data(:));
        s = std(tmp_data(:));
        chdata = (chdata -m)./s;

        % get the confidence intervals using bootstrap
        %     m = mean(chdata,2);
        %     mb = sort(bootstrp(1000,@mean,chdata'));
        %     tt=1:size(erps,1);
        %     [fillhandle,msg]=jbfill(tt,mb(25,:),mb(975,:)...
        %         ,[0.3 0.3 0.7],[0.3 0.3 0.7],1,.2);
        %     hold on
        %     plot(m,'b')
        %plot(mb(25,:),'--b')
        %plot(mb(975,:),'--b')
        %hline(0)

        % get the confidence interval using the variance
        m = mean(chdata,2);
        mb = std(chdata',1)';
        mlow = m-mb;
        mhigh = m+mb;
        tt=1:size(erps,1);
        [fillhandle,msg]=jbfill(tt,mlow',mhigh'...
            ,[0.3 0.3 0.7],[0.3 0.3 0.7],1,.2);
        hold on
        plot(m,'b')


        % shuffle the data for null confidence intervals
        tmp_mean=[];
        for j=1:1000
            %tmp = circshift(chdata,randperm(size(chdata,1),1));
            tmp = chdata;
            tmp(randperm(numel(chdata))) = tmp;
            tmp_data=tmp(1:5,:);
            m = mean(tmp_data(:));
            s = std(tmp_data(:));
            tmp = (tmp -m)./s;
            tmp_mean(j,:) = mean(tmp,2);
        end

        tmp_mean = sort(tmp_mean);
        %plot(tmp_mean(25,:),'--r')
        %plot(tmp_mean(975,:),'--r')
        [fillhandle,msg]=jbfill(tt,tmp_mean(25,:),tmp_mean(975,:)...
            ,[0.7 0.3 0.3],[0.7 0.3 0.3],1,.2);


        % statistical test
        % if the mean is outside confidence intervals in state 3
        m = mean(chdata,2);
        idx=12:30;
        mstat = m((idx));
        pval=[];
        for j=1:length(idx)
            pval(j) = (sum(abs(mstat(j)) >= abs(tmp_mean(:,idx(j)))))./(size(tmp_mean,1));
        end

        [pfdr,~] = fdr(1-pval,0.05);

        % just counting approach
        %     res=sum((1-pval)<=0.05);
        %     if res>=floor(length(idx)/2)
        %         suc=1;
        %     else
        %         suc=0;
        %     end

        % fdr approach
        res = sum((1-pval)<=pfdr);
        if res > 1
            suc = 1;
            sig_ch(x,y,ii)=1;
        else
            suc =  0;
        end

        % beautify
        ylabel (num2str(i))
        axis tight
        ylim([-2 5])
        %set(gca,'LineWidth',1)
        %vline([time(2:4)])
        h=vline(tim);
        %set(h,'LineWidth',1)
        set(h,'Color','k')
        h=hline(0);
        set(h,'LineWidth',1.5)
        if i~=102
            yticklabels ''
            xticklabels ''
        else
            %xticks([tim])
            %xticklabels({'S1','S2','S3','S4'})
        end

        if suc==1
            box on
            set(gca,'LineWidth',2)
            set(gca,'XColor','g')
            set(gca,'YColor','g')
        end
        d = d+1;
    end
end

erps = D;

end