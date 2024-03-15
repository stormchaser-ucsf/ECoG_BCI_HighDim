function [Data,bins_per_mvmt,TrialData] = ...
    load_B3Data_RepresenatationalStruct_Fig1(bins_size)
%function [Data,bins_per_mvmt,TrialData] = ...
    %load_B3Data_RepresenatationalStruct_Fig1(bins_size)

root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
foldernames = {'20230111','20230118','20230119','20230125','20230126',...
    '20230201','20230203'};
cd(root_path)


files=[]; % these are the foldernanes within each day
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'ImaginedMvmtDAQ')
    D=dir(folderpath);
    for j=3:length(D)
        filepath=fullfile(folderpath,D(j).name,'Imagined');
        tmp=dir(filepath);
        files = [files;string(filepath)];
    end
end

ImaginedMvmt = {'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
    'Rotate Right Wrist','Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp',...
    'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
    'Rotate Left Wrist','Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
    'Squeeze Both Hands',...
    'Imagined Head Movement',...
    'Right Shoulder Shrug',...
    'Left Shoulder Shrug',...
    'Right Tricep','Left Tricep',...
    'Right Bicep','Left Bicep',...
    'Right Knee','Left Knee',...
    'Right Ankle','Left Ankle',...
    'Lips','Tongue'};

%no bicep or tricep
% ImaginedMvmt = {'Right Thumb','Right Index','Right Middle','Right Ring','Right Pinky',...
%     'Right Pinch Grasp','Right Tripod Grasp','Right Power Grasp','Rotate Right Wrist',...
%     'Left Thumb','Left Index','Left Middle','Left Ring','Left Pinky',...
%     'Left Pinch Grasp','Left Tripod Grasp','Left Power Grasp',...
%     'Rotate Left Wrist',...
%     'Squeeze Both Hands',...
%     'Right Shoulder Shrug',...
%     'Left Shoulder Shrug',...
%     'Imagined Head Movement',...
%     'Right Leg','Left Leg',...
%     'Lips','Tongue'};



Data={};
for i=1:length(ImaginedMvmt)
    Data{i}=zeros(0,0);
    bins_per_mvmt{i} = zeros(0,0);
end


for i=1:length(files)

    disp(i/length(files)*100)
    d=dir(files{i});
    len = length(d)-2;
    d=d(3:end);

    Data_tmp={};
    for ii=1:length(ImaginedMvmt)
        Data_tmp{ii}=zeros(0,0);
    end
    data_overall=[];
    cd(files{i})
    for jj=1:len
        %disp(d(jj).name)
        file_loaded=true;
        try
            load(d(jj).name)
        catch
            file_loaded=false;
            %disp(['file not loaded  ' d(jj).name])
        end
        if file_loaded
            features  = TrialData.SmoothedNeuralFeatures;
            %features  = TrialData.NeuralFeatures;
            kinax = TrialData.TaskState;
            kinax2 = find(kinax==2);
            kinax = find(kinax==3);
            %kinax=[kinax2 kinax];
            %kinax=[kinax2 kinax]; % for B3 apparently ERP start before go cue
            temp = cell2mat(features(kinax));
            temp2 = cell2mat(features(kinax2));
            %temp=temp(:,3:end); % ignore the first 600ms

            % baseline the data to state 2
            %m = mean(temp2,2);
            %s = std(temp2')';
            %temp = (temp-m)./s;

            % take from 400 to 2000ms
            temp = temp(:,bins_size); % 12 is better
            

            if size(temp,1)==1792

                % hg and delta and beta
                temp = temp([257:512 1025:1280 1537:1792],:);
                %temp = temp([1537:1792],:);% only hg

                % remove the bad channels 108, 113 118
                bad_ch = [108 113 118];
                good_ch = ones(size(temp,1),1);
                for ii=1:length(bad_ch)
                    %bad_ch_tmp = bad_ch(ii)*[1 2 3];
                    bad_ch_tmp = bad_ch(ii)+(256*[0 1 2]);
                    good_ch(bad_ch_tmp)=0;
                end
                temp = temp(logical(good_ch),:);


                if max(abs(temp(:))) < 8 % not needed when referencing to
                %state 1
                    data_overall = [data_overall;temp'];
                    for j=1:length(ImaginedMvmt)
                        if strcmp(ImaginedMvmt{j},TrialData.ImaginedAction)
                            tmp=Data_tmp{j};
                            tmp = [tmp temp];
                            Data_tmp{j} = tmp;

                            % store number of bins per movement
                            tempp = bins_per_mvmt{j};
                            tempp = [tempp size(temp,2)];
                            bins_per_mvmt{j} = tempp;
                            break
                        end
                    end
                end
            end
        end
    end

    m=mean(data_overall,1);
    s=std(data_overall,1);

    for j=1:length(Data_tmp)
        tmp=Data_tmp{j};
        if ~isempty(tmp)
            tmp = (tmp-m')./s'; % this is better
            Data_tmp{j}=tmp;

            % transfer to main file
            tmp=Data{j};
            tmp = [tmp Data_tmp{j}];
            Data{j} = tmp;
        end
    end
end
