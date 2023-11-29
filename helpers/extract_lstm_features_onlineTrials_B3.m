function out = extract_lstm_features_onlineTrials_B3(tmp,Params,lpFilt)


%get hG through filter bank approach
filtered_data=zeros(size(tmp,1),size(tmp,2),8);
for i=1:8 % only hg
    filtered_data(:,:,i) =  ((filter(...
        Params.FilterBank(i).b, ...
        Params.FilterBank(i).a, ...
        tmp)));
end
tmp_hg = squeeze(mean(filtered_data.^2,3));

% LFO low pass filtering
tmp_lp = filter(lpFilt,tmp);

% reject bad channels 
good_ch=ones(size(tmp_hg,2),1);
good_ch([108 113 118])=0;
good_ch=logical(good_ch);
tmp_hg = tmp_hg(:,good_ch);
tmp_lp = tmp_lp(:,good_ch);

% resampling the data
tmp_hg=resample(tmp_hg,size(tmp_hg,1)/10,size(tmp_hg,1))*5e2;
tmp_lp=resample(tmp_lp,size(tmp_lp,1)/10,size(tmp_lp,1));

% removing errors in the data
I = abs(tmp_hg>15);
I = sum(I);
[aa bb]=find(I>0);
tmp_hg(:,bb) = 1e-15*randn(size(tmp_hg(:,bb)));
I = abs(tmp_lp>15);
I = sum(I);
[aa bb]=find(I>0);
tmp_lp(:,bb) = 1e-15*randn(size(tmp_lp(:,bb)));

% normalizing between 0 and 1
tmp_hg = (tmp_hg - min(tmp_hg(:)))/(max(tmp_hg(:))-min(tmp_hg(:)));
tmp_lp = (tmp_lp - min(tmp_lp(:)))/(max(tmp_lp(:))-min(tmp_lp(:)));

% L2 norm of 1
tmp_hg = tmp_hg./norm(tmp_hg(:));
tmp_lp = tmp_lp./norm(tmp_lp(:));

% make new data structure
tmp = [tmp_hg tmp_lp ];
out=tmp;



