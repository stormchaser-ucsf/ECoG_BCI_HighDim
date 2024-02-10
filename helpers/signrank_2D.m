function [t,p] = signrank_2D(chdata)
%function [t,p] = signrank_2D(chdata)
% chdata - rows correspond to time and cols to trials 

t=[];p=[];
for ii=1:size(chdata,1)
    [pp,h,stats] = signrank(chdata(ii,:));
    t(ii) = stats.zval;
    p(ii)= pp;
end