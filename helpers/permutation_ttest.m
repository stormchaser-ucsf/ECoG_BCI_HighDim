function [p bootstrp_tvalues tvalue,df ]= permutation_ttest(x,y,k,n)
%function [p bootstrp_tvalues tvalue]=bootstrap_ttest(x,y,k,n)
% x,y are data vectors
% k is 1 for paired ttest and 2 for two sample t-test
% n is number of bootstrap samples

if k==2
    x=x(~isnan(x));
    y=y(~isnan(y));
end

if k==1
    [H,P,CI,STATS]=ttest(x,y);
    tvalue=STATS.tstat;
    df=STATS.df;
elseif k==2
    [H,P,CI,STATS]=ttest2(x,y);
    tvalue=STATS.tstat;
    df=STATS.df;
end

x=x(:);
y=y(:);

z=[x;y];
z=z-mean(z);

len1 = length(x);
len2 = length(y);

t=[];
for i=1:n

    z1=  z(randperm(numel(z)));
    x1 = z1(1:len1);
    y1 = z1(len1+1:end);

    if k==1
        [H,P,CI,STATS]=ttest(x1,y1);
        t(i)=STATS.tstat;
    elseif k==2
        [H,P,CI,STATS]=ttest2(x1,y1);
        t(i)=STATS.tstat;
    end


end

p=sum(abs(t)>abs(tvalue))/length(t);
bootstrp_tvalues=t;

if p==0
    p=1/n;
end



end

% 
% tmp_a=[];
% for i=1:25:length(a)
%     xx = a(i:i+25-1);
%     tmp_a = [tmp_a mean(xx)];
% end
% 
% tmp_b=[];
% for i=1:25:length(b)
%     xx = b(i:i+25-1);
%     tmp_b = [tmp_b mean(xx)];
% end
% 
% 
% tmp_c=[];
% for i=1:25:length(c)
%     xx = c(i:i+25-1);
%     tmp_c = [tmp_c mean(xx)];
% end








