function out = extract_lstm_features_onlineTrials(tmp,Params,lpFilt)




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

% get lg thru filter bank approach
filtered_data=zeros(size(tmp,1),size(tmp,2),3);
for i=9:11 % only lg
    filtered_data(:,:,i) =  ((filter(...
        Params.FilterBank(i).b, ...
        Params.FilterBank(i).a, ...
        tmp)));
end
tmp_lg = squeeze(mean(filtered_data.^2,3));

% downsample the data
%     tmp_lp = resample(tmp_lp,200,800);
%     tmp_hg = resample(tmp_hg,200,800)*5e2;

% decimate the data, USE AN OPTIONAL SMOOTHING INFO HERE
%     tmp_hg1=[];
%     tmp_lp1=[];
%     for i=1:size(tmp_hg,2)
%         tmp_hg1(:,i) = decimate(tmp_hg(:,i),20)*5e2;
%         tmp_lp1(:,i) = decimate(tmp_lp(:,i),20);
%     end

% resample
tmp_lg=resample(tmp_lg,size(tmp_lg,1)/10,size(tmp_lg,1))*5e2;
tmp_hg=resample(tmp_hg,size(tmp_hg,1)/10,size(tmp_hg,1))*5e2;
tmp_lp=resample(tmp_lp,size(tmp_lp,1)/10,size(tmp_lp,1));

% removing errors in the data
I = abs(tmp_hg>12);
I = sum(I);
[aa bb]=find(I>0);
tmp_hg(:,bb) = 1e-5*randn(size(tmp_hg(:,bb)));

I = abs(tmp_lp>12);
I = sum(I);
[aa bb]=find(I>0);
tmp_lp(:,bb) = 1e-5*randn(size(tmp_lp(:,bb)));

I = abs(tmp_lg>12);
I = sum(I);
[aa bb]=find(I>0);
tmp_lg(:,bb) = 1e-5*randn(size(tmp_lg(:,bb)));

% normalizing between 0 and 1
tmp_hg = (tmp_hg - min(tmp_hg(:)))/(max(tmp_hg(:))-min(tmp_hg(:)));
tmp_lp = (tmp_lp - min(tmp_lp(:)))/(max(tmp_lp(:))-min(tmp_lp(:)));
tmp_lg = (tmp_lg - min(tmp_lg(:)))/(max(tmp_lg(:))-min(tmp_lg(:)));

% make new data structure
tmp = [tmp_hg tmp_lp ];
out=tmp;



