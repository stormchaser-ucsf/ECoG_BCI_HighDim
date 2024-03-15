function out = two_means_ci(a,b)
%function out = two_means_ci(a,b)

wss1 = norm((a-mean(a)),'fro');
wss2 = norm((b-mean(b)),'fro');
wss = wss1+wss2;
c = [a;b];
tss = norm((c-mean(c)),'fro');
out = wss/tss;

end