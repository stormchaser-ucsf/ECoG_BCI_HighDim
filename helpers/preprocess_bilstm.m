function out = preprocess_bilstm(tmp,Params)
%function filtered_data = preprocess_bilstm(tmp,Params)

%get hG through filter bank approach
% filtered_data=zeros(size(tmp,1),size(tmp,2),8);
% for i=1:8%length(Params.FilterBank)
%     filtered_data(:,:,i) =  ((filter(...
%         Params.FilterBank(i).b, ...
%         Params.FilterBank(i).a, ...
%         tmp)));
% end
% tmp_hg = squeeze(mean(filtered_data.^2,3));

% get hg through the main filtering approach
tmp_hg = filtfilt(Params.FilterBank(end).b,...
                    Params.FilterBank(end).a,...
                    tmp);
tmp_hg = abs(hilbert(tmp_hg));
% tmp_hg_theta = filter(Params.FilterBank(9).b,...
%                      Params.FilterBank(9).a,...
%                      tmp_hg);

                
% LFO low pass filtering
%tmp_lp = filter(lpFilt,tmp);

% get theta signal 
% tmp_theta = filter(Params.FilterBank(9).b,...
%                     Params.FilterBank(9).a,...
%                     tmp);
% tmp_theta=abs(hilbert(tmp_theta));


% downsample the data
%tmp_theta = resample(tmp_theta,250,500);
tmp_hg = resample(tmp_hg,125,500);
%tmp_hg_theta = resample(tmp_hg_theta,125,500);
% tmp_hg_theta = tmp_hg_theta./norm(tmp_hg_theta(:));


%tmp_lp = resample(tmp_lp,200,800);
%tmp_hg = resample(tmp_hg,200,800)*5e2;

out=tmp_hg;


    
end