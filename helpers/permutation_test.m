function [pval] = permutation_test(a,b)

stat_val = abs(mean(a)-mean(b));
c=[a;b];
c=c-mean(c);
len = length(a);
stat_boot=[];
for i=1:5000
    idx=randperm(numel(c));
    c1 =  c(idx);
    a1 = c1(1:len);
    b1 = c1(len+1:end);
    stat_boot(i) = abs(mean(a1)-mean(b1));
end
pval = 1 - sum(stat_val>stat_boot)/length(stat_boot);
if pval==0
    pval = 1/length(stat_boot);
end

figure;hist(stat_boot)
vline(stat_val,'r')

end

