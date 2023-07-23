function pmask = sig_ch_erps(D,TrialData,tim)



% plot the ERPs with bootstrapped C.I. shading
chMap=TrialData.Params.ChMap;
%figure
%ha=tight_subplot(8,16);
d = 1;
%set(gcf,'Color','w')
pmask=zeros(8,16);
for i = 1:size(D,1)
     [x y] = find(chMap==i);
%     if x == 1
%         axes(ha(y));
%         %subplot(8, 16, y)
%     else
%         s = 16*(x-1) + y;
%         axes(ha(s));
%         %subplot(8, 16, s)
%     end
%     hold on
    erps =  squeeze(D(i,:,:)); % change this to the action to generate ERPs

    chdata = erps;
    % zscore the data to the first 8 time-bins
    tmp_data=chdata(1:8,:);
    m = mean(tmp_data(:));
    s = std(tmp_data(:));
    chdata = (chdata -m)./s;

    % get the confidence intervals
    m = mean(chdata,2);
    mb = sort(bootstrp(1000,@mean,chdata'));
    tt=1:size(D,2);
%     [fillhandle,msg]=jbfill(tt,mb(25,:),mb(975,:)...
%         ,[0.3 0.3 0.7],[0.3 0.3 0.7],1,.2);
%     hold on
%     plot(m,'b')
    %plot(mb(25,:),'--b')
    %plot(mb(975,:),'--b')
    %hline(0)

    % shuffle the data for null confidence intervals
    tmp_mean=[];
    for j=1:1000
        %tmp = circshift(chdata,randperm(size(chdata,1),1));
        tmp = chdata;
        tmp(randperm(numel(chdata))) = tmp;
        tmp_data=tmp(1:8,:);
        m = mean(tmp_data(:));
        s = std(tmp_data(:));
        tmp = (tmp -m)./s;
        tmp_mean(j,:) = mean(tmp,2);
    end

    tmp_mean = sort(tmp_mean);
    %plot(tmp_mean(25,:),'--r')
    %plot(tmp_mean(975,:),'--r')
%     [fillhandle,msg]=jbfill(tt,tmp_mean(25,:),tmp_mean(975,:)...
%         ,[0.7 0.3 0.3],[0.7 0.3 0.3],1,.2);


    % statistical test
    % if the mean is outside confidence intervals in state 3
    m = mean(chdata,2);
    idx=10:25;
    mstat = m((idx));
    pval=[];
    for j=1:length(idx)
        pval(j) = (sum(abs(mstat(j)) >= abs(tmp_mean(:,idx(j)))))./(size(tmp_mean,1));
    end

    res=sum((1-pval)<=0.05);
    if res>=7
        suc=1;
        pmask(x,y)=1;
    else
        suc=0;
    end

%     % beautify
%     ylabel (num2str(i))
%     axis tight
%     ylim([-2 4])
%     %set(gca,'LineWidth',1)
%     %vline([time(2:4)])
%     h=vline(tim);
%     %set(h,'LineWidth',1)
%     set(h,'Color','k')
%     h=hline(0);
%     set(h,'LineWidth',1.5)
%     if i~=102
%         yticklabels ''
%         xticklabels ''
%     else
%         xticks([tim])
%         xticklabels({'S1','S2','S3','S4'})
%         yticks([-2:2:4])
%         yticklabels({'-2','0','2','4'})
%     end
% 
%     if suc==1
%         box on
%         set(gca,'LineWidth',2)
%         set(gca,'XColor','g')
%         set(gca,'YColor','g')
%     end
    d = d+1;
end

