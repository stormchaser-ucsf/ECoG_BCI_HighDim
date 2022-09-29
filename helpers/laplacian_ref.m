function [out_data] = laplacian_ref(indata,chmap)


[xx yy] = size(chmap);
out_data=zeros(size(indata));
parfor k=1:size(indata,2)
    tmp1 = indata(1:128,k);
    tmp1 = tmp1(chmap);    
    out=zeros(xx,yy);
    for i=1:xx
        for j=1:yy
            % get the neighbors
            i_nb = [i-1 i+1];
            j_nb = [j-1 j+1];            
            i_nb = i_nb(logical((i_nb>0) .* (i_nb<=8)));
            j_nb = j_nb(logical((j_nb>0) .* (j_nb<=16)));
            ref_ch_vals = [tmp1(i,[j_nb]) tmp1(i_nb,[j])'];
            out(i,j) = tmp1(i,j) - mean(ref_ch_vals);                        
        end
    end  
    out_data(:,k) = out(:);
end
