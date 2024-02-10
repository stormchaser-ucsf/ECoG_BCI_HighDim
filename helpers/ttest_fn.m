function [t_val] = ttest_fn(indata)
%function [t_val] = ttest_fn(indata)
% gets the t-value from a one sample t-test

 [h0 p0 ci stats0]=ttest(indata); % run the t-test
 t_val = stats0.tstat;
