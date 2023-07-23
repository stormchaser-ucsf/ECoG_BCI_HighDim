clc; clear all; close all;
%% Load Data
root_path = 'D:\B1Raw';
foldernames = {'20230616'};
task = 'Robot3DArrow';

cd(root_path)

files=[];
for i = 1:length(foldernames)
    foldernames{i}
    folderpath = fullfile(root_path,foldernames{i}, 'GangulyServer', task);
    D = dir(folderpath);
    for j = 3:length(D)
        folderpath,D(j).name
        filepath = fullfile(folderpath, D(j).name, 'Imagined');
        try
            files = [files; findfiles('mat',filepath)'];
        catch
        end
    end
end


%% Load Data for each target
D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
D8=[];
D9=[];

% define frequency bands
bands = {
    struct('name', 'Delta', 'range', [129:256]),
    struct('name', 'Theta', 'range', [257:384]),
    struct('name', 'Beta', 'range', [513:640]),
    struct('name', 'LG', 'range', [641:768]),
    struct('name', 'HG', 'range', [769:896])
    };

for b = 1:length(bands)
    time_to_target=zeros(2,9);
    for i=1:length(files)
        disp(i)
        load(files{i});
        trail_length = length(TrialData.BroadbandData);
        Neuro.BufferSamps = TrialData.Params.BufferSamps;
        Neuro.NumChannels = TrialData.Params.NumChannels;
        Neuro.NumBuffer = TrialData.Params.NumBuffer;
        Neuro.NumFeatures = TrialData.Params.NumFeatures;
        Neuro.NumPhase = TrialData.Params.NumPhase;
        Neuro.SpatialFiltering     = false;
        Neuro.NumFeatureBins = 1;
        Neuro.FeatureBufferSize = 5;
        features = {};
        Neuro.FeatureDataBuffer = zeros(Neuro.FeatureBufferSize,Neuro.NumFeatures*Neuro.NumChannels);
        Neuro.FilterDataBuf = zeros(Neuro.BufferSamps,Neuro.NumChannels,Neuro.NumBuffer);
        for j = 1:trail_length
            Neuro.NumSamps = TrialData.NeuralSamps(j);
            Neuro.FilterBank = TrialData.Params.FilterBank;
            for k=1:length(Neuro.FilterBank)
                Neuro.FilterBank(k).state = [];
            end
            Neuro.BroadbandData = TrialData.BroadbandData{j};
            Neuro = sim_ApplyFilterBank(Neuro);
            Neuro = sim_UpdateNeuroBuf(Neuro);
            Neuro = sim_CompNeuralFeatures(Neuro);
            Neuro = sim_SmoothNeuro(Neuro);
            features{end+1} = Neuro.FilteredFeatures;
        end
        features = cell2mat(features);
        features = features(bands{b}.range,:); 
        fs = TrialData.Params.UpdateRate;
        kinax = TrialData.TaskState;
        state1 = (1:5);
        state2 = (6:10);
        state3 = find(kinax==3);
        state4 = size(kinax,2)-5:size(kinax,2);
        tmp_data = features(:,state3);
        idx1= ones(length(state1),1);
        idx2= 2*ones(length(state2),1);
        idx3= 3*ones(length(state3),1);
        idx4= 4*ones(length(state4),1);

        % interpolate
        tb = (1/fs)*[1:size(tmp_data,2)];
        t=(1/fs)*[1:10];
        tb = tb*t(end)/tb(end);
        tmp_data1 = interp1(tb,tmp_data',t,'spline')';
        idx3 = interp1(tb,idx3,t,'spline');

        % now stick all the data together
        trial_dur = (length(state3)-TrialData.Params.ClickCounter)*(1/fs);
        state4_data = features(:, state4);
        % Add state4_data back into your data array
        data = [features(:,[state1 state2]) tmp_data1 features(:,[state4])];

        % correction
        if length(state1)<8
            data  =[data(:,1) data];
        end

        % store the time to target data
        time_to_target(2,TrialData.TargetID) = time_to_target(2,TrialData.TargetID)+1;
        if trial_dur<=3
            time_to_target(1,TrialData.TargetID) = time_to_target(1,TrialData.TargetID)+1;
        end

        % now get the ERPs
        % if TrialData.TargetID == TrialData.SelectedTargetID && trial_dur<=3
        if TrialData.TargetID == 1
            D1 = cat(3,D1,data);
        elseif TrialData.TargetID == 2
            D2 = cat(3,D2,data);
        elseif TrialData.TargetID == 3
            D3 = cat(3,D3,data);
        elseif TrialData.TargetID == 4
            D4 = cat(3,D4,data);
        elseif TrialData.TargetID == 5
            D5 = cat(3,D5,data);
        elseif TrialData.TargetID == 6
            D6 = cat(3,D6,data);
        elseif TrialData.TargetID == 7
            D7 = cat(3,D7,data);
        elseif TrialData.TargetID == 8
            D8 = cat(3,D8,data);
        elseif TrialData.TargetID == 9
            D9 = cat(3,D9,data);
        end
        %  end
    end

    %% Plot graph
    seriesData = {D1, D2, D3, D4, D5, D6, D7, D8, D9};
    for k = 1:9
        k
        time_to_target(1,:)./time_to_target(2,:)


        % plot the ERPs with bootstrapped C.I. shading
        chMap=TrialData.Params.ChMap;
        figure
        ha=tight_subplot(8,16);
        annotation('textbox', [0.5, 1, 0, 0], 'String',['ERP:', bands{b}.name, 'Target:', string(k)], 'HorizontalAlignment', 'center', 'FontSize', 8);
        d = 1;
        set(gcf,'Color','w')
        tim = cumsum([length(idx1) length(idx2) length(idx3) length(idx4)]);

        for i = 1:size(D2,1)
            [x y] = find(chMap==i);
            if x == 1
                axes(ha(y));
                %subplot(8, 16, y)
            else
                s = 16*(x-1) + y;
                axes(ha(s));
                %subplot(8, 16, s)
            end
            hold on
            erps =  squeeze(seriesData{k}(i,:,:));

            chdata = erps;
            % zscore the data to the first 8 time-bins
            tmp_data=chdata(1:8,:);
            m = mean(tmp_data(:));
            s = std(tmp_data(:));
            chdata = (chdata -m)./s;

            % get the confidence intervals
            m = mean(chdata,2);
            mb = sort(bootstrp(1000,@mean,chdata'));
            tt=1:size(D1,2);
            [fillhandle,msg]=jbfill(tt,mb(25,:),mb(975,:)...
                ,[0.3 0.3 0.7],[0.3 0.3 0.7],1,.2);
            hold on
            plot(m,'b')
            %plot(mb(25,:),'--b')
            %plot(mb(975,:),'--b')
            %hline(0)

            % shuffle the data for null confidence intervals
            tmp_mean=[];
            for j=1:1000
                %tmp = circshift(chdata,randperm(size(chdata,1),1));
                tmp = chdata;
                tmp(randperm(numel(chdata))) = tmp;
                tmp_data=tmp(1:8,:);
                m = mean(tmp_data(:));
                s = std(tmp_data(:));
                tmp = (tmp -m)./s;
                tmp_mean(j,:) = mean(tmp,2);
            end

            tmp_mean = sort(tmp_mean);
            %plot(tmp_mean(25,:),'--r')
            %plot(tmp_mean(975,:),'--r')
            [fillhandle,msg]=jbfill(tt,tmp_mean(25,:),tmp_mean(975,:)...
                ,[0.7 0.3 0.3],[0.7 0.3 0.3],1,.2);


            % statistical test
            % if the mean is outside confidence intervals in state 3
            m = mean(chdata,2);
            idx=10:20;
            mstat = m((idx));
            pval=[];
            for j=1:length(idx)
                pval(j) = (sum(abs(mstat(j)) >= abs(tmp_mean(:,idx(j)))))./(size(tmp_mean,1));
            end

            res=sum((1-pval)<=0.05);
            if res>=7
                suc=1;
            else
                suc=0;
            end

            % beautify
            ylabel (num2str(i))
            axis tight
            ylim([-2 4])
            %set(gca,'LineWidth',1)
            %vline([time(2:4)])
            h=vline(tim);
            %set(h,'LineWidth',1)
            set(h,'Color','k')
            h=hline(0);
            set(h,'LineWidth',1.5)
            if i~=102
                yticklabels ''
                xticklabels ''
            else
                %xticks([tim])
                %xticklabels({'S1','S2','S3','S4'})
            end

            if suc==1
                box on
                set(gca,'LineWidth',2)
                set(gca,'XColor','g')
                set(gca,'YColor','g')
            end
            d = d+1;
        end
        saveas(gcf, ['F:\BCICodeGroup\ERP\20230616', bands{b}.name,'ERPTarget',num2str(k),'.png']);
    end
end

