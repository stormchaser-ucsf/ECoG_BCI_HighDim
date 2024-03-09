function out = get_children(z,parent)

out=[];
for j=1:length(parent)
    idx = find(z(:,end)==parent(j));
    out = [out z(idx,1:2)];
end

end