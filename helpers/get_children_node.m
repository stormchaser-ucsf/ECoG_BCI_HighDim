function [children_all] = get_children_node(Z,root_idx)

max_datapts = size(Z,1)+1;
children_all={};
for i=1:length(root_idx)
    children=[];
    % first step is to get the immediate below neighbors
    if root_idx(i)<=max_datapts
        children = [children root_idx(i)];
    end
    parents = Z(find(Z(:,end)==root_idx(i)),1:2);
    children = [children parents(parents<=max_datapts)];
    going_down= true;    
    while going_down        
        parents = get_children(Z,parents)  ;      
        children = [children parents(parents<=max_datapts)];
        parents = parents(parents>max_datapts);
        if all(parents<=max_datapts) || isempty(parents)
            going_down=false;
        end
    end
    children_all{i} = children(children<=max_datapts);
end