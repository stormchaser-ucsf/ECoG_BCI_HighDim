function [t,pvalues,cluster_sums]=temporal_clust_1D(a,alp)
%
% USGAE function [t]=temporal_clust(a,alp,ch)
%INPUT
% a - 2D matrix, rows-trials, columns - time points
% alp - alpha value (0.05, 0.01, etc.)
%OUTPUT
% t - cluster of time points. Vector as long as the number of time-points.
% 0 - not significant. 1 -  significant temporal clusters
% EXAMPLE
% a = randn(50,5000);
% t = temporal_clust(a,0.05);
%
% figure;
% subplot(211)
% plot(mean(a,1));
% subplot(212)
% plot(t)
% ylim([0 1.1])
% legend('sig. temporal clusters of difference')
%
% NN 2024


[h0 p0 ci stats]=ttest(a);
t0=stats.tstat;


% extract temporal clusters
sta=[];stp=[]; ind=1;
while ind<length(p0)
    if p0(ind) <=alp
        sta=[sta ind];
        while p0(ind) <= alp && ind+1 <= length(p0)
            ind=ind+1;
        end
        stp=[stp ind-1];
    else
        ind=ind+1;
    end
end

% form the cluster values
clus=[];
for i=1:length(sta)
    clus=[clus sum(t0(sta(i):stp(i)))];
end

% ground truth values
sta0=sta;
stp0=stp;
clus0=clus;


% BOOTSTRAPPING: COMPARE TO SETTING THE MEAN TO ZERO
% temporally shuffle within a trial
%anew=a;
% for i=1:size(anew,1)
%     anew(i,:) = anew(i,randperm(size(anew,2)));
% end
% anew = anew-mean(anew);
anew=a-mean(a); % set the mean to zero
asize=size(anew);
clus_boot=[];
parfor i=1:2000

    a1= anew(randi(asize(1),[asize(1) 1]),:); % sample with replacement

    h0=[];p0=[];ci=[];stats=[];t0=[];
    [h0 p0 ci stats]=ttest(a1);
    t0=stats.tstat;

    % extract temporal clusters
    sta=[];stp=[]; ind=1;
    while ind<length(p0)
        if p0(ind) <=alp
            sta=[sta ind];
            while p0(ind) <= alp && ind+1 <= length(p0)
                ind=ind+1;
            end
            stp=[stp ind-1];
        else
            ind=ind+1;
        end
    end

    % form the cluster values
    clus=[];
    for ii=1:length(sta)
        clus=[clus sum(t0(sta(ii):stp(ii)))];
    end

    if length(clus)>0
        clus_boot=[clus_boot max(abs(clus))];
    else
        clus_boot=[clus_boot 0];
    end
end

clus_boot=sort(clus_boot,'ascend');
I=find(abs(clus0)>clus_boot(floor(length(clus_boot)*(1-alp))));
% for i=1:length(clus0)
%     pval(i) = 1-sum(abs(clus0(i))>clus_boot)/length(clus_boot);
% end
t=zeros(size(a,2),1);
for i=1:length(I)
    t(sta(I(i)): stp(I(i)))=1;
end
if ~isempty(I)
    for i=1:length(I)
        pvalues(i) = 1-sum(abs(clus0(I(i)))>clus_boot)/length(clus_boot);
    end
else
    pvalues=[];
end

cluster_sums = clus0(I);

%
% figure;plot(mean(a))
% hold on
% idx=find(t==1);
% vline(idx)
% hline(0,'k')


end