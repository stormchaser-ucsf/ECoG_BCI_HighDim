function plot_elec_wts_B3(wts,cortex,elecmatrix,chMap)
%function plot_elec_wts_B3(wts,cortex,elecmatrix)


figure
grid_ecog = [];
for i=1:23:253
    grid_ecog =[grid_ecog; i:i+22];
end
grid_ecog= fliplr((grid_ecog));

c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
e_h = el_add(elecmatrix,'color','w','msize',2);
for ch=1:length(wts)
    [x y] = find(chMap==ch);
    ch1 = grid_ecog(x,y);    
    s = wts(ch);    
    if s~=0
        e_h = el_add(elecmatrix(ch1,:),'color','r','msize',s);
    end
end
set(gcf,'Color','w')
view(-99,21)


% % also plotting as heat map
% ch_trf=[];
% for ch=1:length(wts)
%     [x y] = find(chMap==ch);
%     ch_trf(ch) = grid_ecog(x,y);        
% end
% figure;
% c_h = ctmr_gauss_plot(cortex,elecmatrix(ch_trf,:),wts,'lh',1,1,1);