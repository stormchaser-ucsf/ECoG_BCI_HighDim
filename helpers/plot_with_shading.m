function [out] = plot_with_shading(indata,task_state)
% function [out] = plot_with_shading(indata)
% indata:     Make sure that the columns correspond to the features/channels
%             etc and the rows are observations/trials etc. 
% task_state: Task state to draw vertical lines nad reference things to

% reference the data to the first 5 bins of task state 1
%ref_bins = find(task_state==1);
ref_bins = 1:5;

for i=1:size(indata,1)
    %indata(i,:) = detrend(indata(i,:));
    %indata(i,:) = (indata(i,:) - mean(indata(i,ref_bins)))./std(indata(i,ref_bins));
    indata(i,:) = (indata(i,:) - mean(indata(i,ref_bins)));
end

m = mean(indata,1);
mb = sort(bootstrp(1000,@mean,indata));

tt=1:size(indata,2);
[fillhandle,msg]=jbfill(tt,mb(25,:),mb(975,:)...
    ,[0.3 0.3 0.7],[0.3 0.3 0.7],1,.2);
hold on
plot(m,'b')
axis tight
ylim([-1 2])

[aa bb]=unique(task_state);
bb=bb(2:end);
bb=bb-1;
vline(bb)
hline(0)



end