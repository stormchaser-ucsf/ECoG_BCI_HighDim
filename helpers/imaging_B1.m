


dirn=pwd;
addpath('C:\Users\nikic\Documents\MATLAB\ctmr_gauss_plot_April2016\ctmr_gauss_plot_April2016')
cd('C:\Users\nikic\Documents\MATLAB\ctmr_gauss_plot_April2016\ctmr_gauss_plot_April2016')
load('BRAVO1_lh_pial')
load('BRAVO1_elecs_all')

ch=1:size(anatomy,1);
figure
c_h = ctmr_gauss_plot(cortex,[0 0 0],0,'rh');
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
