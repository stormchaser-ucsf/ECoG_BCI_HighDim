function [decodes] = mode_filter_B2(idx) % 5 sample mode filter

decodes(1:4) = idx(1:4);
for i=5:length(idx)
    tmp = idx(i-4:i);
    for ii=1:4
        mode_vals(ii) = sum(tmp==ii);
    end
    [aa bb] = max(mode_vals);
    if aa>=3
        decodes(i) = bb;
    else
        decodes(i)=0;
    end
end

end