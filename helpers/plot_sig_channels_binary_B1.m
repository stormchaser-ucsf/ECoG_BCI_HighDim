function plot_sig_channels_binary_B1(indata,cortex,elecmatrix,chmap)


%%%%% BRAIN PLOTTING
data=indata;
temp= abs(data);
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],...
    0,'lh',1,1,1);
set(gcf,'Color','w')
origMap = [1:16;17:32;33:48;49:64;65:80;81:96;97:112;113:128];
origMap = flipud(origMap);
e_h1 = el_add(elecmatrix(:,:), 'color', [1 1 1],'msize',2);
for j=1:length(temp)
    [x y]=find(origMap==j);
    ch=chmap(x,y);    
    if abs(temp(ch)) == 1
        c = 'r';
        ms = abs(temp(ch)) * (10-3)+2;
    else
        c = 'r';
        ms = 3;
    end
    e_h = el_add(elecmatrix(j,:), 'color',c,'msize',ms);
end
view(-101,24)

end