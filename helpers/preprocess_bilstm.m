function out = preprocess_bilstm(tmp,Params)
%function out = preprocess_bilstm(tmp,Params)

%get hG through filter bank approach
filtered_data=zeros(size(tmp,1),size(tmp,2),8);
for i=1:8
    filtered_data(:,:,i) =  ((filter(...
        Params.FilterBank(i).b, ...
        Params.FilterBank(i).a, ...
        tmp)));
end
tmp_hg = squeeze(mean(filtered_data.^2,3));

% get hg through the main filtering approach
% tmp_hg = filtfilt(Params.FilterBank(end).b,...
%                     Params.FilterBank(end).a,...
%                     tmp);
% tmp_hg = abs(hilbert(tmp_hg));

% tmp_hg_theta = filter(Params.FilterBank(9).b,...
%                      Params.FilterBank(9).a,...
%                      tmp_hg);

                
% LFO low pass filtering
tmp_lp = filter(Params.FilterBank(end-1).b,...
    Params.FilterBank(end-1).a,...
    tmp);

% get theta signal 
 tmp_theta = filter(Params.FilterBank(end).b,...
                     Params.FilterBank(end).a,...
                     tmp);
 %tmp_theta=abs(hilbert(tmp_theta));


% downsample the data by some factor 
len = size(tmp_hg,1);
len1 = round(len/4);
tmp_hg = resample(tmp_hg,len1,len)*5e2;
tmp_lp = resample(tmp_lp,len1,len);
tmp_theta = resample(tmp_theta,len1,len);

% concatenate and send out 
out = [tmp_hg tmp_lp];
%out=tmp_theta;

% get hilbert and linearize phase
%out=angle(hilbert(out));
%out = sin(out);


% norm data
%out = out./norm(out(:));

    
end