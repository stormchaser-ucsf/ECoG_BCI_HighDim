function new_parent = get_childern(z,parent)

idx = find(z(:,end)==parent);
new_parent = z(idx,1:2);

end