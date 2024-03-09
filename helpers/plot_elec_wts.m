function plot_elec_wts(wts,cortex,elecmatrix,chMap)
%function plot_elec_wts(wts,cortex,elecmatrix)


figure
grid_ecog = [];
for i=1:16:128
    grid_ecog =[grid_ecog; i:i+15];
end
grid_ecog=flipud(grid_ecog);

c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh',1,1,1);
for ch=1:length(wts)
    [x y] = find(chMap==ch);
    ch1 = grid_ecog(x,y);    
    s = wts(ch);    
    if s~=0
        e_h = el_add(elecmatrix(ch1,:),'color','r','msize',s,'edgecol','r');
    else
        e_h1 = el_add(elecmatrix(ch1,:),'color','w','msize',1,'edgecol','w');
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