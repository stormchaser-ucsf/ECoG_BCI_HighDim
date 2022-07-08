

% show the difference between velocities of the cursor during the first few
% seconds of mvmt for all directions. 

clc;clear
dates = {'20190724','20190725','20190726','20190730'};
filepath = '/media/reza/WindowsDrive/BRAVO1/CursorPlatform/Data';

% get all the filenames
files=[];
for i=1:length(dates)
   foldername = fullfile(filepath,dates{i},'GangulyServer','Center-Out',...
       dates{i});
   filenames = findfiles('mat',foldername,1)';
   
   for j=1:length(filenames)
       if regexp(filenames{j},'BCI_Fixed')
           files=[files;filenames(j)];
       end
   end
end


% now get the distribution of data points within the first 2s after target
% came on 
D1=[];
D2=[];
D3=[];
D4=[];
for i=1:length(files)
    load(files{i})
    
    
    
    
end

j=1;
j=j+1;
load(files{j})
TrialData.TargetID
kin=TrialData.CursorState;
cmap = parula(size(kin,2));
figure;hold on
xlim([-300 300])
ylim([-300 300])
for i=1:size(cmap,1)
    plot(kin(1,i),kin(2,i),'.','Color',cmap(i,:),'MarkerSize',10)    
end



%% not sure which blocks have center reset on, so doing a blind search


clc;clear
filepath = '/media/reza/WindowsDrive/BRAVO1/CursorPlatform/Data';
filenames = findfiles('mat',filepath,1)';

files=[];
for i=1:length(filenames)
    filedate = str2num(filenames{i}(53:60));
    if ~isempty(regexp(filenames{i},'BCI_Fixed'))...
            && ~isempty(regexp(filenames{i},'Center-Out'))...
            && filedate>=20190618
        files = [files;filenames(i)];
    end
end

% getting only those with center reset
files_center=[];
for i=1:length(files)
    disp(i)
    load(files{i})
    if TrialData.Params.CenterReset
        files_center = [files_center;files(i)];
    end
end



% now looking at the distribution of angles within the first 2s 
rt=[];
lt=[];
up=[];
down=[];
angles_relative=[];
for i=1:length(files_center)
    disp(i)
    load(files_center{i})
    kin = TrialData.CursorState;
    f = TrialData.Params.UpdateRate;
    % get 2s of data
    t = 2*f;
    ts = TrialData.Params.InstructedDelayTime*f+2;
    final_time = ts+t-1;
    if size(kin,2)<final_time
        final_time = size(kin,2);
    end
    kin = kin(1:4,ts:final_time);
    target = TrialData.TargetPosition';
    
    % in terms of angle between final position and decoded position
    %kin_relative = target-kin(1:2,:);
    %ang = atan(kin_relative(2,:)./kin_relative(1,:));
    %angles_relative =[angles_relative;ang'];
    
    % in terms of angle between velocity update and final position
    %if TrialData.TargetID==1
    %     ang = atan(kin(4,:)./kin(3,:));
    %     target_ang = atan(target(2)/target(1));
    %     angles_relative = [angles_relative;(target_ang-ang)'];
    
    
    % in terms of angle between cursor position and final target position
    kinp = kin(1:2,:);
    kinv = kin(3:4,:);
    for j=1:size(kinp,2)
       kinp(:,j) =  kinp(:,j)./norm(kinp(:,j));
       kinv(:,j) =  kinv(:,j)./norm(kinv(:,j));
    end
    kin_norm = [kinp;kinv];
    target_norm = target./norm(target);
    ang = acos(kinv'*target_norm);
    angles_relative =[angles_relative;ang];
end
figure;rose(angles_relative,20)
figure;hist(angles_relative)


load(files_center{69})
TrialData.TargetID
kin=TrialData.CursorState;
cmap = parula(size(kin,2));
figure;hold on
xlim([-300 300])
ylim([-300 300])
for i=1:size(cmap,1)
    plot(kin(1,i),kin(2,i),'.','Color',cmap(i,:),'MarkerSize',10)    
end
plot(TrialData.TargetPosition(1),TrialData.TargetPosition(2),'.r',...
    'MarkerSize',30)


%% LOOKING AT ANGLES BETWEEN DECODES AND TARGET IN THE 3D ARROW TASK

clc;clear
foldernames = {'20210813','20210818','20210825','20210827','20210901','20210903',...
        '20210915'};

filepath_main = '/media/reza/WindowsDrive/BRAVO1/CursorPlatform/Data';

files=[];
for i=1:length(foldernames)
    filepath = fullfile(filepath_main,foldernames{i});
    filenames = findfiles('mat',filepath,1)';
    for j=1:length(filenames)
        if ~isempty(regexp(filenames{j},'BCI_Fixed')) ...
                && ~isempty(regexp(filenames{j},'Robot3DArrow'))
            files = [files;filenames(j)];
        end
    end    
end


% load the files and get the decodes
loc=[];
T1 = [1 0 0];
T2 = [0,1,0];
T3 = [-1,0,0];
T4 = [0,-1,0];
T5 = [0,0,1];
T6 = [0,0,-1];
T7=[1,1,1];
T7=T7./norm(T7);
T = [T1;T2;T3;T4;T5;T6;T7];
ang=[];
for i=1:length(files)
    disp(i)
    load(files{i})
    decodes = TrialData.FilteredClickerState;
    decodes = decodes(decodes>0);
    if length(decodes)>0
        %loc = [loc;TrialData.TargetID TrialData.TargetPosition];
        directions=[];
        for j=1:length(decodes)
            if decodes(j)==1
                directions = [directions;T1];
            elseif decodes(j)==2
                directions = [directions;T2];
            elseif decodes(j)==3
                directions = [directions;T3];
            elseif decodes(j)==4
                directions = [directions;T4];
            elseif decodes(j)==5
                directions = [directions;T5];
            elseif decodes(j)==6
                directions = [directions;T6];
            elseif decodes(j)==7
                directions = [directions;T7];
            end
        end
        ref_line = T(TrialData.TargetID,:);
        ang=[ang;acos(directions*ref_line')];
    end
end
figure;rose(real(ang),30)
figure;hist(real(ang),20)

ang1=abs(ang);
ang1=ang1+0.01*randn(size(ang1));
figure;rose((ang1),30)
figure;hist(ang1,20)

%T1 = [200 0 0];
%T2 = [0,200,0];
%T3 = [-200,0,0]
%T4 = [0,-200,0]
%T5 = [0,0,200]
%T6 = [0,0,-200];
%T7=[0,0,0];

















