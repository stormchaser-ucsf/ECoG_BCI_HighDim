%% plotting diagonal robot trajectories (MAIN)



clc;clear
foldername = '20210115';
task_name = 'Robot';
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath 'C:\Users\nikic\Documents\MATLAB'
addpath(genpath('C:\Users\nikic\Documents\MATLAB\svg_plot'))

files={'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210115\Robot\114418\BCI_Fixed\Data0002.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210115/Robot\114613\BCI_Fixed\Data0001.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210115\Robot\114613\BCI_Fixed\Data0002.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210115\Robot\114613\BCI_Fixed\Data0004.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210115\Robot\114740\BCI_Fixed\Data0001.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0001.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0002.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0003.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0005.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0006.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0007.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0008.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0009.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0010.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0011.mat',
    'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\20210122\Robot\111630\BCI_Fixed\Data0012.mat'};

% get all the good robot 3D trial data and plot them
col = {'r','g','b','c','m','y'};
col=turbo(4);
files_suc=[];
figure;
hold on
xlim([-250,250])
ylim([-250,250])
zlim([-250,250])
tid=[];
% also get the velocities in the x and y directions
vel=[];
err_vel=[];
for i=1:length(files)
    load(files{i})
    tid = [tid TrialData.TargetID];
    if TrialData.TargetID == TrialData.SelectedTargetID
        kin = TrialData.CursorState;
        task_state = TrialData.TaskState;
        kinidx = find(task_state==3);
        kin = kin(:,kinidx);
        target = TrialData.TargetPosition;
        targetID = TrialData.TargetID-6;
        fs = TrialData.Params.UpdateRate;
        if size(kin,2)*(1/fs) < 12
            files_suc = [files_suc;files(i)];
            %plot3(kin(1,:),kin(2,:),kin(3,:),'LineWidth',2,'color',col{targetID});
            plot3(kin(1,:),kin(2,:),kin(3,:),'LineWidth',2,'color',col(targetID,:));
            vel = [vel kin(4:6,:)];
        end
        % get the velocities relative to the ideal velocity towards the
        % target
        pos = TrialData.TargetPosition(1:2)'
        start_pos = kin(1:2,1);
        ideal_vector  = pos-start_pos;
        ideal_vector = ideal_vector./norm(ideal_vector);
        tmp_vel = kin(4:6,1:end);
        idx = abs(sum(tmp_vel))>0;
        tmp_vel = tmp_vel(1:2,idx);
        for j=1:length(tmp_vel)
            tmp_vel(:,j)=tmp_vel(:,j)./norm(tmp_vel(:,j));
        end
        %%% cos angle
        %angles_err = acos(ideal_vector'*tmp_vel);
        %err_vel =[err_vel angles_err];
        %%% angle to target
        %ideal_angle = atan2(ideal_vector(2)/ideal_vector(1));
        %angles_err = atan2(tmp_vel(2,:)./tmp_vel(1,:));
        ideal_angle = atan2(ideal_vector(1),ideal_vector(2));
        angles_err = atan2(tmp_vel(1,:),tmp_vel(2,:));
        angles_err_rel = angles_err - ideal_angle;
        err_vel =[err_vel angles_err_rel];
    end
end

% histogram of the errors in decoded velocities with the ideal velocity
figure;rose(err_vel)
figure;hist(err_vel*180/pi,20)
vline(45)

% circular statistics test
addpath(genpath('C:\Users\nikic\Documents\MATLAB\CircStat2012a'))
mu = circ_mean(err_vel') % get the mean
[pval, z] = circ_rtest(err_vel); % is it uniformly distributed
[h mu ul ll]  = circ_mtest(err_vel', 0) % does it have a specific mean
[ll mu ul]*180/pi
pval = circ_medtest(err_vel',0)


%%%%%%% USING CHAT GPT %%%%

% Set of angles in radians
angles = err_vel';

% Reference direction (null hypothesis)
mu_0 = 0; % You can set this to your desired reference direction

% Compute the circular mean
mu = circ_mean(angles);

% Perform the one-sample test and get the p-value
p_value = circ_test(angles - mu_0);

% Display the results
fprintf('Circular mean: %f\n', mu);
fprintf('P-value: %f\n', p_value);

% Compare with significance level
alpha = 0.05; % Set your desired significance level
if p_value < alpha
    fprintf('Reject the null hypothesis: The mean direction is significantly different.\n');
else
    fprintf('Fail to reject the null hypothesis: The mean direction is not significantly different.\n');
end


%%%%%%%%%%%



%grid off
set(gcf,'Color','w')
set(gca,'FontSize',12)
plot2svg('3DTraj_diag1.svg');

% pca on the velocity data
[c,s,l] = pca(vel');
figure;
stem(cumsum(l)./sum(l))
figure;stem(c(:,1))
xlim([0 4])
figure;stem(c(:,2))
xlim([0 4])
figure;stem(c(:,3))
xlim([0 4])
figure;



% pca on null velocity data, 2dim
vel_null=[];
for i=1:100
    if rand(1)>0.5
        tmp = [randn(1,10)*20;zeros(1,10)];
    else
        tmp = [zeros(1,10);randn(1,10)*20];
    end
    vel_null = [vel_null tmp];
end
[c,s,l] = pca(vel_null');
figure;stem(c(:,1),'k')
xlim([0.5 2.5])
xticks([1:2])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
box off
figure;stem(c(:,2),'k')
xlim([0.5 2.5])
xticks([1:2])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
box off

% pca on the first x and y velocity alone
C = cov((vel(1:2,:))');
figure;imagesc(C)
[c,s,l] = pca(vel(1:2,:)');
figure;stem(c(:,1),'k')
xlim([0.5 2.5])
xticks([1:2])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
box off
figure;stem(c(:,2),'k')
xlim([0.5 2.5])
xticks([1:2])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
box off
% nullhypothesis via MP test
kindata1=vel(1:2,:)';
kindata1=kindata1-mean(kindata1);
q = size(kindata1,1)/size(kindata1,2);
q = sqrt(q);
sigma2 = var(kindata1(:)-mean(kindata1(:)));
lambda_thresh = sigma2*((1+1/q)^2);
figure;stem(l);
hline(lambda_thresh)




% get histogram of angles relative to the x and y axes
velxy = vel(1:2,:);
angles = [];
for i=1:size(velxy,2)
    velxy(:,i) = velxy(:,i)./norm(velxy(:,i));
    tmp = (velxy(:,i));
    angles(i) = atan(tmp(2)/tmp(1));
end
errx = acos([1 0]*abs(velxy));
erry = acos([0 1]*abs(velxy));
err = [errx erry];
err = err(~isnan(err));
angles = angles(~isnan(angles));

% null distribution to compare these angles towards:
% Fit an exponential distribution centered at 0 and pi/2 with variance
% equal to the actual data
mu = std(angles);
tmp=0:0.01:pi/2;
y = exppdf(tmp,mu);
%y = conv(y,fliplr(y),'same');
y=y+fliplr(y);
y=y./sum(y);
figure;plot(tmp,y)
figure;hist(angles)



%plot
[t,r]=rose(err_vel,20);
figure
polarplot(t,r,'LineWidth',1,'Color','k');
pax=gca;
%pax.RLim = [0 20];
thetaticks(0:30:360);
pax.ThetaAxisUnits='radians';
pax.FontSize=16;
set(gcf,'Color','w')
%pax.RTick = [5 10 15 20 ];
pax.GridAlpha = 0.25;
pax.MinorGridAlpha = 0.25;
pax.ThetaMinorGrid = 'off';
%pax.ThetaTickLabel = {'0', ' ', '\pi/2 ', ' ','\pi',' ','3\pi/2',' '};
pax.ThetaTickLabel = ''
%pax.ThetaTickLabel = {'0', ' ', ' ', ' ','\pi',' ',' ',' '};
pax.RTickLabel = {' ',' '};
pax.RAxisLocation=1;
pax.RAxis.LineWidth=1;
pax.ThetaAxis.LineWidth=1;
pax.LineWidth=1;
%pax.ThetaLim = [0 pi/2];
temp = exp(1i*err_vel);
r1 = abs(mean(temp))*1 * max(r);
phi = angle(mean(temp));
hold on;
polarplot([phi-0.01 phi],[0 r1],'LineWidth',1.5,'Color','r')
%polarplot([0.7854-0.01 0.7854],[0 r1],'LineWidth',1.5,'Color','m')
%polarplot([0 0],[0 0.25e3],'LineWidth',1.5,'Color','k')
%polarplot([pi/2 pi/2 ],[0 0.25e3],'LineWidth',1.5,'Color','k')
set(gcf,'PaperPositionMode','auto')
set(gcf,'Position',[680.0,865,120.0,113.0])


% null model of what distributions should be like
err_null = [zeros(1,400) 90*pi/180*ones(1,400)];
[t,r]=rose(err_null,50);
figure
polarplot(t,r,'LineWidth',1,'Color','k');
pax=gca;
%pax.RLim = [0 20];
thetaticks(0:30:360);
pax.ThetaAxisUnits='radians';
pax.FontSize=16;
set(gcf,'Color','w')
%pax.RTick = [5 10 15 20 ];
pax.GridAlpha = 0.25;
pax.MinorGridAlpha = 0.25;
pax.ThetaMinorGrid = 'off';
%pax.ThetaTickLabel = {'0', ' ', '\pi/2 ', ' ','\pi',' ','3\pi/2',' '};
pax.ThetaTickLabel = ''
%pax.ThetaTickLabel = {'0', ' ', ' ', ' ','\pi',' ',' ',' '};
pax.RTickLabel = {' ',' '};
pax.RAxisLocation=1;
pax.RAxis.LineWidth=1;
pax.ThetaAxis.LineWidth=1;
pax.LineWidth=1;
pax.ThetaLim = [0 90*pi/180];
hold on;
polarplot([0 0],[0 0.4e3],'LineWidth',1.5,'Color','k')
polarplot([pi/2 pi/2 ],[0 0.4e3],'LineWidth',1.5,'Color','k')
set(gcf,'PaperPositionMode','auto')
set(gcf,'Position',[680.0,865,120.0,113.0])


% now plotting a few example trials along with position, user input and
% velocity profile

idx=1; %1 and 7
load(files{idx})
kin = TrialData.CursorState;
task_state = TrialData.TaskState;
kinidx = find(task_state==3);
kin = kin(:,kinidx);
decodes = TrialData.FilteredClickerState;
fs=TrialData.Params.UpdateRate;
tt = [0:length(decodes)-1]*(1/fs);
col = turbo(8);

% plot the trajectory
figure
hold on
for i=1:length(decodes)
    c = col(decodes(i)+1,:);
    plot3(kin(1,i),-kin(2,i),kin(3,i),'.','MarkerSize',40,'Color',c);
end
xlim([-300 300])
ylim([-300 300])
zlim([-300 300])
target = TrialData.TargetPosition;
plot3(target(1),-target(2),target(3),'ok','MarkerSize',50)
%plot(-150,150,'o','MarkerSize',50,'Color','k')
set(gcf,'Color','w')
xticks([-200:200:200])
yticks([-200:200:200])
zticks([-200:200:200])

% plot as a straight line
figure;
hold on
plot3(kin(1,:),-kin(2,:),kin(3,:),'LineWidth',2,'Color','b');
xlim([-300 300])
ylim([-300 300])
zlim([-300 300])
target = TrialData.TargetPosition;
plot3(target(1),-target(2),target(3),'ok','MarkerSize',50)
%plot(-150,150,'o','MarkerSize',50,'Color','k')
set(gcf,'Color','w')
xticks([-200:200:200])
yticks([-200:200:200])
zticks([-200:200:200])
xlabel('X axis')
ylabel('Y axis')
zlabel('Z axis')
set(gcf,'Color','w')
set(gca,'FontSize',14)
%view(40,40)



% plot the decodes
figure;
set(gcf,'Color','w')
subplot(2,1,1)
hold on
for i=0:7
    h=barh(i,length(decodes),1);
    h.FaceColor = col(i+1,:);
    %h.FaceAlpha = 0.8;
    h.FaceAlpha = 1;
end
stem(decodes,'filled','LineWidth',1,'Color','k')
ii = [9 17 25 33 41 49];
xticks(ii)
xticklabels(tt(ii))
yticks ''
axis tight
% plot the velocity profile
subplot(2,1,2)
hold on
plot(kin(4,:),'r','LineWidth',1)
plot(kin(5,:),'b','LineWidth',1)
plot(kin(6,:),'g','LineWidth',1)
xticks(ii)
xticklabels(tt(ii))
axis tight

%%%%% plotting the dynamics and decodes for the virtual r2g lateral task
filename = fullfile(root_path,'20211013\RobotLateralR2G\135901\BCI_Fixed\Data0001.mat');
load(filename)
grip_mode = TrialData.OpMode;
decodes = TrialData.FilteredClickerState;
translation_decodes = decodes(grip_mode==0);
gripper_decodes = decodes(grip_mode==1);

% plot the decodes
fs=TrialData.Params.UpdateRate;
tt = [0:length(decodes)-1]*(1/fs);
col = turbo(8);
figure;
set(gcf,'Color','w')
hold on
for i=0:7
    h=barh(i,length(decodes),1);
    h.FaceColor = col(i+1,:);
    h.FaceAlpha = 0.5;
end
stem(decodes,'filled','LineWidth',1,'Color','k')
ii = [26 51 76 101 126 151 176 201 226 ];
xticks(ii)
xticklabels(tt(ii))
yticks ''
axis tight

%% B3 DIAGONAL ROBOT TRAJ

%% plotting diagonal robot trajectories (MAIN)

clc;clear
foldername = {'20231207','20231210','20231215','20231218'};
task_name = 'Robot';
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
cd(root_path)
addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
addpath 'C:\Users\nikic\Documents\MATLAB'
addpath(genpath('C:\Users\nikic\Documents\MATLAB\svg_plot'))

files=[];
for i=1:length(foldername)
    tmp = fullfile(root_path,foldername{i},'Robot3D');
    tmp=dir(tmp);
    if i==1
        idx=[7:10];
        tmp=tmp(idx);
    end
    for j=1:length(tmp)
        filename = fullfile(tmp(j).folder,tmp(j).name,'BCI_Fixed');
        files=[files;findfiles('',filename)'];
    end
end


% get all the good robot 3D trial data and plot them
col = {'r','g','b','c','m','y'};
col=turbo(4);
files_suc=[];
figure;
hold on
xlim([-250,250])
ylim([-250,250])
zlim([-250,250])
tid=[];
% also get the velocities in the x and y directions
vel=[];
err_vel=[];
for i=1:length(files)
    load(files{i})
    tid = [tid TrialData.TargetID];
    if TrialData.TargetID == TrialData.SelectedTargetID
        kin = TrialData.CursorState;
        task_state = TrialData.TaskState;
        kinidx = find(task_state==3);
        kin = kin(:,kinidx);
        target = TrialData.TargetPosition;
        targetID = TrialData.TargetID-6;
        fs = TrialData.Params.UpdateRate;
        if size(kin,2)*(1/fs) < 12
            files_suc = [files_suc;files(i)];
            %plot3(kin(1,:),kin(2,:),kin(3,:),'LineWidth',2,'color',col{targetID});
            plot3(kin(1,:),kin(2,:),kin(3,:),'LineWidth',2,'color',col(targetID,:));
            vel = [vel kin(4:6,:)];
        end
        % get the velocities relative to the ideal velocity towards the
        % target
        pos = TrialData.TargetPosition(1:2)';
        start_pos = kin(1:2,1);
        ideal_vector  = pos-start_pos;
        ideal_vector = ideal_vector./norm(ideal_vector);
        tmp_vel = kin(4:6,1:end);
        idx = abs(sum(tmp_vel))>0;
        tmp_vel = tmp_vel(1:2,idx);
        for j=1:length(tmp_vel)
            tmp_vel(:,j)=tmp_vel(:,j)./norm(tmp_vel(:,j));
        end
        %%% cos angle
        %angles_err = acos(ideal_vector'*tmp_vel);
        %err_vel =[err_vel angles_err];
        %%% angle to target
        %ideal_angle = atan2(ideal_vector(2)/ideal_vector(1));
        %angles_err = atan2(tmp_vel(2,:)./tmp_vel(1,:));
        ideal_angle = atan2(ideal_vector(1),ideal_vector(2));
        angles_err = atan2(tmp_vel(1,:),tmp_vel(2,:));
        angles_err_rel = angles_err - ideal_angle;
        err_vel =[err_vel angles_err_rel];
    end
end

% histogram of the errors in decoded velocities with the ideal velocity
figure;rose(err_vel)
figure;hist(err_vel*180/pi,20)
vline(45)

% circular statistics test
addpath(genpath('C:\Users\nikic\Documents\MATLAB\CircStat2012a'))
mu = circ_mean(err_vel') % get the mean
[pval, z] = circ_rtest(err_vel); % is it uniformly distributed
[h mu ul ll]  = circ_mtest(err_vel', 0) % does it have a specific mean
[ll mu ul]*180/pi
pval = circ_medtest(err_vel',0)


%%%%%%% USING CHAT GPT %%%%

% Set of angles in radians
angles = err_vel';

% Reference direction (null hypothesis)
mu_0 = 0; % You can set this to your desired reference direction

% Compute the circular mean
mu = circ_mean(angles);

% Perform the one-sample test and get the p-value
p_value = circ_test(angles - mu_0);

% Display the results
fprintf('Circular mean: %f\n', mu);
fprintf('P-value: %f\n', p_value);

% Compare with significance level
alpha = 0.05; % Set your desired significance level
if p_value < alpha
    fprintf('Reject the null hypothesis: The mean direction is significantly different.\n');
else
    fprintf('Fail to reject the null hypothesis: The mean direction is not significantly different.\n');
end


%%%%%%%%%%%



%grid off
set(gcf,'Color','w')
set(gca,'FontSize',12)
plot2svg('3DTraj_diag1.svg');

% pca on the velocity data
[c,s,l] = pca(vel');
figure;
stem(cumsum(l)./sum(l))
figure;stem(c(:,1))
xlim([0 4])
figure;stem(c(:,2))
xlim([0 4])
figure;stem(c(:,3))
xlim([0 4])
figure;



% pca on null velocity data, 2dim
vel_null=[];
for i=1:100
    if rand(1)>0.5
        tmp = [randn(1,10)*20;zeros(1,10)];
    else
        tmp = [zeros(1,10);randn(1,10)*20];
    end
    vel_null = [vel_null tmp];
end
[c,s,l] = pca(vel_null');
figure;stem(c(:,1),'k')
xlim([0.5 2.5])
xticks([1:2])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
box off
figure;stem(c(:,2),'k')
xlim([0.5 2.5])
xticks([1:2])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
box off

% pca on the first x and y velocity alone
C = cov((vel(1:2,:))');
figure;imagesc(C)
[c,s,l] = pca(vel(1:2,:)');
figure;stem(c(:,1),'k')
xlim([0.5 2.5])
xticks([1:2])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
box off
figure;stem(c(:,2),'k')
xlim([0.5 2.5])
xticks([1:2])
set(gcf,'Color','w')
set(gca,'LineWidth',1)
box off
% nullhypothesis via MP test
kindata1=vel(1:2,:)';
kindata1=kindata1-mean(kindata1);
q = size(kindata1,1)/size(kindata1,2);
q = sqrt(q);
sigma2 = var(kindata1(:)-mean(kindata1(:)));
lambda_thresh = sigma2*((1+1/q)^2);
figure;stem(l);
hline(lambda_thresh)




% get histogram of angles relative to the x and y axes
velxy = vel(1:2,:);
angles = [];
for i=1:size(velxy,2)
    velxy(:,i) = velxy(:,i)./norm(velxy(:,i));
    tmp = (velxy(:,i));
    angles(i) = atan(tmp(2)/tmp(1));
end
errx = acos([1 0]*abs(velxy));
erry = acos([0 1]*abs(velxy));
err = [errx erry];
err = err(~isnan(err));
angles = angles(~isnan(angles));

% null distribution to compare these angles towards:
% Fit an exponential distribution centered at 0 and pi/2 with variance
% equal to the actual data
mu = std(angles);
tmp=0:0.01:pi/2;
y = exppdf(tmp,mu);
%y = conv(y,fliplr(y),'same');
y=y+fliplr(y);
y=y./sum(y);
figure;plot(tmp,y)
figure;hist(angles)



%plot
[t,r]=rose(err_vel,20);
figure
polarplot(t,r,'LineWidth',1,'Color','k');
pax=gca;
%pax.RLim = [0 20];
thetaticks(0:30:360);
pax.ThetaAxisUnits='radians';
pax.FontSize=16;
set(gcf,'Color','w')
%pax.RTick = [5 10 15 20 ];
pax.GridAlpha = 0.25;
pax.MinorGridAlpha = 0.25;
pax.ThetaMinorGrid = 'off';
%pax.ThetaTickLabel = {'0', ' ', '\pi/2 ', ' ','\pi',' ','3\pi/2',' '};
pax.ThetaTickLabel = ''
%pax.ThetaTickLabel = {'0', ' ', ' ', ' ','\pi',' ',' ',' '};
pax.RTickLabel = {' ',' '};
pax.RAxisLocation=1;
pax.RAxis.LineWidth=1;
pax.ThetaAxis.LineWidth=1;
pax.LineWidth=1;
%pax.ThetaLim = [0 pi/2];
temp = exp(1i*err_vel);
r1 = abs(mean(temp))*1 * max(r);
phi = angle(mean(temp));
hold on;
polarplot([phi-0.01 phi],[0 r1],'LineWidth',1.5,'Color','r')
%polarplot([0.7854-0.01 0.7854],[0 r1],'LineWidth',1.5,'Color','m')
%polarplot([0 0],[0 0.25e3],'LineWidth',1.5,'Color','k')
%polarplot([pi/2 pi/2 ],[0 0.25e3],'LineWidth',1.5,'Color','k')
set(gcf,'PaperPositionMode','auto')
set(gcf,'Position',[680.0,865,120.0,113.0])


% null model of what distributions should be like
err_null = [zeros(1,400) 90*pi/180*ones(1,400)];
[t,r]=rose(err_null,50);
figure
polarplot(t,r,'LineWidth',1,'Color','k');
pax=gca;
%pax.RLim = [0 20];
thetaticks(0:30:360);
pax.ThetaAxisUnits='radians';
pax.FontSize=16;
set(gcf,'Color','w')
%pax.RTick = [5 10 15 20 ];
pax.GridAlpha = 0.25;
pax.MinorGridAlpha = 0.25;
pax.ThetaMinorGrid = 'off';
%pax.ThetaTickLabel = {'0', ' ', '\pi/2 ', ' ','\pi',' ','3\pi/2',' '};
pax.ThetaTickLabel = ''
%pax.ThetaTickLabel = {'0', ' ', ' ', ' ','\pi',' ',' ',' '};
pax.RTickLabel = {' ',' '};
pax.RAxisLocation=1;
pax.RAxis.LineWidth=1;
pax.ThetaAxis.LineWidth=1;
pax.LineWidth=1;
pax.ThetaLim = [0 90*pi/180];
hold on;
polarplot([0 0],[0 0.4e3],'LineWidth',1.5,'Color','k')
polarplot([pi/2 pi/2 ],[0 0.4e3],'LineWidth',1.5,'Color','k')
set(gcf,'PaperPositionMode','auto')
set(gcf,'Position',[680.0,865,120.0,113.0])


% now plotting a few example trials along with position, user input and
% velocity profile

idx=1; %1 and 7
load(files{idx})
kin = TrialData.CursorState;
task_state = TrialData.TaskState;
kinidx = find(task_state==3);
kin = kin(:,kinidx);
decodes = TrialData.FilteredClickerState;
fs=TrialData.Params.UpdateRate;
tt = [0:length(decodes)-1]*(1/fs);
col = turbo(8);

% plot the trajectory
figure
hold on
for i=1:length(decodes)
    c = col(decodes(i)+1,:);
    plot3(kin(1,i),-kin(2,i),kin(3,i),'.','MarkerSize',40,'Color',c);
end
xlim([-300 300])
ylim([-300 300])
zlim([-300 300])
target = TrialData.TargetPosition;
plot3(target(1),-target(2),target(3),'ok','MarkerSize',50)
%plot(-150,150,'o','MarkerSize',50,'Color','k')
set(gcf,'Color','w')
xticks([-200:200:200])
yticks([-200:200:200])
zticks([-200:200:200])

% plot as a straight line
figure;
hold on
plot3(kin(1,:),-kin(2,:),kin(3,:),'LineWidth',2,'Color','b');
xlim([-300 300])
ylim([-300 300])
zlim([-300 300])
target = TrialData.TargetPosition;
plot3(target(1),-target(2),target(3),'ok','MarkerSize',50)
%plot(-150,150,'o','MarkerSize',50,'Color','k')
set(gcf,'Color','w')
xticks([-200:200:200])
yticks([-200:200:200])
zticks([-200:200:200])
xlabel('X axis')
ylabel('Y axis')
zlabel('Z axis')
set(gcf,'Color','w')
set(gca,'FontSize',14)
%view(40,40)



% plot the decodes
figure;
set(gcf,'Color','w')
subplot(2,1,1)
hold on
for i=0:7
    h=barh(i,length(decodes),1);
    h.FaceColor = col(i+1,:);
    %h.FaceAlpha = 0.8;
    h.FaceAlpha = 1;
end
stem(decodes,'filled','LineWidth',1,'Color','k')
ii = [9 17 25 33 41 49];
xticks(ii)
xticklabels(tt(ii))
yticks ''
axis tight
% plot the velocity profile
subplot(2,1,2)
hold on
plot(kin(4,:),'r','LineWidth',1)
plot(kin(5,:),'b','LineWidth',1)
plot(kin(6,:),'g','LineWidth',1)
xticks(ii)
xticklabels(tt(ii))
axis tight

%%%%% plotting the dynamics and decodes for the virtual r2g lateral task
filename = fullfile(root_path,'20211013\RobotLateralR2G\135901\BCI_Fixed\Data0001.mat');
load(filename)
grip_mode = TrialData.OpMode;
decodes = TrialData.FilteredClickerState;
translation_decodes = decodes(grip_mode==0);
gripper_decodes = decodes(grip_mode==1);

% plot the decodes
fs=TrialData.Params.UpdateRate;
tt = [0:length(decodes)-1]*(1/fs);
col = turbo(8);
figure;
set(gcf,'Color','w')
hold on
for i=0:7
    h=barh(i,length(decodes),1);
    h.FaceColor = col(i+1,:);
    h.FaceAlpha = 0.5;
end
stem(decodes,'filled','LineWidth',1,'Color','k')
ii = [26 51 76 101 126 151 176 201 226 ];
xticks(ii)
xticklabels(tt(ii))
yticks ''
axis tight

