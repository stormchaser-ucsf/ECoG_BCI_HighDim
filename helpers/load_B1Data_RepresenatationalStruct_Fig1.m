function [Data,bins_per_mvmt,TrialData] = load_B1Data_RepresenatationalStruct_Fig1(bins_size)
%function [Data,bins_per_mvmt] = load_B1Data_RepresenatationalStruct_Fig1(bins_size)



root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\';
foldernames = {'20211201','20211203','20211206','20211208','20211215','20211217',...
    '20220126','20220223','20220225'};
cd(root_path)

files=[]; % these are the foldernanes within each day
for i=1:length(foldernames)
    folderpath = fullfile(root_path, foldernames{i},'ImaginedMvmtDAQ')
    D=dir(folderpath);
    if i==3
        D = D([1:3 5:7 9:end]);
    elseif i==4
        D = D([1:3 5:end]);
    elseif i==6
        D = D([1:5 7:end]);
    end

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
    'Right Leg','Left Leg',...
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



Data={};bins_per_mvmt={};
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
        load(d(jj).name)
        features  = TrialData.SmoothedNeuralFeatures;
        %features  = TrialData.NeuralFeatures;
        kinax = TrialData.TaskState;
        kinax2 = find(kinax==2);
        kinax = find(kinax==3);
        temp = cell2mat(features(kinax));
        temp2 = cell2mat(features(kinax2));
        %temp=temp(:,3:end); % ignore the first 600ms

        % baseline the data to state 2
        m = mean(temp2,2);
        s = std(temp2')';
        %temp = (temp-m)./s;

        % take from 400 to 2000ms
        temp = temp(:,bins_size);
        %temp=temp(:,3:end); % ignore the first 600ms

        % hg and delta and beta
        %temp = temp([129:end],:);
        %temp = temp([129:256 513:640 769:end],:);
        temp = temp([ 769:end],:);% only hG
        %temp = temp([ 129:256],:);% only delta
        %temp = temp([ 513:640],:);% only beta

        %get smoothed delta hg and beta features
        %         new_temp=[];
        %         [xx yy] = size(TrialData.Params.ChMap);
        %         for k=1:size(temp,2)
        %             tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
        %             tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
        %             tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
        %             pooled_data=[];
        %             for i=1:2:xx
        %                 for j=1:2:yy
        %                     delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
        %                     beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
        %                     hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
        %                     pooled_data = [pooled_data; delta; beta ;hg];
        %                 end
        %             end
        %             new_temp= [new_temp pooled_data];
        %         end
        %         temp=new_temp;
        %         temp = temp([ 1:32 65:96 ],:);


        data_overall = [data_overall;temp'];



        for j=1:length(ImaginedMvmt)
            if strcmp(ImaginedMvmt{j},TrialData.ImaginedAction)
                tmp=Data_tmp{j};
                tmp = [tmp temp];
                Data_tmp{j} = tmp;

                % store number of bins per movement
                temp = bins_per_mvmt{j};
                temp = [temp size(tmp,2)];
                bins_per_mvmt{j} = temp;
                break
            end
        end
    end

    m=mean(data_overall,1);
    s=std(data_overall,1);

    for j=1:length(Data_tmp)
        tmp=Data_tmp{j};
        tmp = (tmp-m')./s';
        Data_tmp{j}=tmp;

        % transfer to main file
        tmp=Data{j};
        tmp = [tmp Data_tmp{j}];
        Data{j} = tmp;
    end
end
