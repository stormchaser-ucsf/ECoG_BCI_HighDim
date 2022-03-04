
%% IMAGING
dirn=pwd;
addpath('C:\Users\Nikhlesh\Documents\MATLAB\ctmr_gauss_plot_April2016\ctmr_gauss_plot_April2016')
cd('C:\Users\Nikhlesh\Documents\MATLAB\ctmr_gauss_plot_April2016\ctmr_gauss_plot_April2016')
load('BRAVO1_lh_pial')
load('BRAVO1_elecs_all')

ch=1:size(anatomy,1);
figure
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh');
% To plot electrodes with numbers, use the following, as example:
e_h = el_add(elecmatrix([1:length(ch)],:), 'color', 'b', 'numbers', ch);
% Or you can just plot them without labels with the default color
%e_h = el_add(elecmatrix(1:64,:)); % only loading 48 electrode data
set(gcf,'Color','w')
cd(dirn)

% plotting with a color bar denoting the phase values and the radius
% denoting the amplitude values.
ph=[linspace(pi,-pi,128)];
phMap = linspace(-pi,pi,128)';
ChColorMap=parula(128);
val = rand(128,1);
ch=1:128;
figure;
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'lh');
for j=1:length(val)
    ms = val(j)*(10-1)+1;
    [aa bb]=min(abs(ph(j) - phMap));
    c=ChColorMap(bb,:);
    e_h = el_add(elecmatrix(j,:), 'color', c,'msize',ms);
end

%% PLOTTING WEIGHTS ON THE BRAIN

cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\201190913\BCI_Fixed')
% get the Kalman matrices first
load('Data0002.mat')
chMap=TrialData.Params.ChMap;
C=TrialData.KalmanFilter{1}.C;
wts_vx = C(:,3);
wts_vy = C(:,4);

origMap = [1:16;17:32;33:48;49:64;65:80;81:96;97:112;113:128];
origMap = flipud(origMap)

addpath('C:\Users\Nikhlesh\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
% plot wts for Vy
elecmatrix1 = elecmatrix;
elecmatrix1(:,1) = elecmatrix(:,1)-5;
for i=1:128:896
    temp=wts_vx(i:i+128-1) + 1i*wts_vy(i:i+128-1);
    temp = abs(temp);
    clear ChColorMap color_blue color_red
    figure
    temp=temp./(max(abs(temp)));
    chI = find(temp~=0);
    c_h = ctmr_gauss_plot(cortex,[0 0 0],...
        0,'lh');
    ChColorMap = brewermap(length(temp),'Blues');
    [xx yy] = sort(temp);
    ChColorMap = ChColorMap(yy,:);
    
    e_h1 = el_add(elecmatrix1(:,:), 'color', [1 1 1],'msize',2);
    for j=1:length(chI)
        [x y]=find(origMap==j);
        ch=chMap(x,y);
        ms = temp(ch) * 8 + 1.75;
        
        c  = ChColorMap(ch,:);
        
        %e_h = el_add(elecmatrix(chI(j),:), 'color',ChColorMap(find(bb==j),:),'msize',ms,'edgecol','k');
        
        e_h = el_add(elecmatrix1(j,:), 'color',[.2 .2 .8],'msize',ms,'edgecol',[.2 .2 .8]);
        
        % e_h = el_add(elecmatrix(j,:), 'color',c,'msize',5);
        
        %e_h.LineWidth=1.25;
    end
    % 0 0.33 1 is a nice colormap
    set(gcf,'Color','k')
    set(gca,'FontSize',20)
    view(-101,24)
end




%% PLOTTING WEIGHTS
% plot wts for Vx
for i=1:128:256
    temp=wts_vx(i:i+128-1);
    clear ChColorMap color_blue color_red
    figure
    temp=temp./(max(abs(temp)));
    chI = find(temp~=0);
    c_h = ctmr_gauss_plot(cortex,[0 0 0],...
        0,'lh');
    ChColorMap(:,1) = linspace(0.0,0.9,length(chI));
    ChColorMap(:,2) = linspace(0.0,0.9,length(chI));
    ChColorMap(:,3) = linspace(1,1,length(chI));
    ChColorMap = flipud((ChColorMap));
    [aa bb] = sort(abs(temp(temp~=0)),'ascend');
    % BOTH +VE AND -VE PARTS
    %[aa bb] = sort((temp(temp~=0)),'ascend');
    e_h1 = el_add(elecmatrix(:,:), 'color', [1 1 1],'msize',2);
    for j=1:length(chI)
        ms = abs(temp(chI(j))) * (10-3) + 3;
        %[aa bb]=min(abs(angle(temp(chI(j))) - phMap));
        %c=ChColorMap(bb,:);%ChColorMap(bb(j),:)
        %%ChColorMap(find(bb==j),:)%[0.39 0.28 0.86]
        e_h = el_add(elecmatrix(chI(j),:), 'color',ChColorMap(find(bb==j),:),'msize',ms,'edgecol','k');
        e_h.LineWidth=1.25;
    end
    % 0 0.33 1 is a nice colormap
    set(gcf,'Color','w')
    set(gca,'FontSize',20)
    view(-101,24)
end

% plot wts for Vy
for i=1:128:256
    temp=wts_vy(i:i+128-1);
    clear ChColorMap color_blue color_red
    figure
    temp=temp./(max(abs(temp)));
    chI = find(temp~=0);
    c_h = ctmr_gauss_plot(cortex,[0 0 0],...
        0,'lh');
    ChColorMap(:,1) = linspace(0.0,0.9,length(chI));
    ChColorMap(:,2) = linspace(0.0,0.9,length(chI));
    ChColorMap(:,3) = linspace(1,1,length(chI));
    ChColorMap = flipud((ChColorMap));
    [aa bb] = sort(abs(temp(temp~=0)),'ascend');
    % BOTH +VE AND -VE PARTS
    %[aa bb] = sort((temp(temp~=0)),'ascend');
    e_h1 = el_add(elecmatrix(:,:), 'color', [1 1 1],'msize',2);
    for j=1:length(chI)
        ms = abs(temp(chI(j))) * (10-3) + 3;
        %[aa bb]=min(abs(angle(temp(chI(j))) - phMap));
        %c=ChColorMap(bb,:);%ChColorMap(bb(j),:)
        %%ChColorMap(find(bb==j),:)%[0.39 0.28 0.86]
        e_h = el_add(elecmatrix(chI(j),:), 'color',ChColorMap(find(bb==j),:),'msize',ms,'edgecol','k');
        e_h.LineWidth=1.25;
    end
    % 0 0.33 1 is a nice colormap
    set(gcf,'Color','w')
    set(gca,'FontSize',20)
    view(-101,24)
end


