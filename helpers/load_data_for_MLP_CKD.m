function [condn_data] = load_data_for_MLP_CKD(files)
%function [condn_data] = load_data_for_MLP_CKD(files)


D1=[];
D2=[];
D3=[];
D4=[];
D5=[];
D6=[];
D7=[];
for ii=1:length(files)
    disp(ii)
    file_loaded=1;
    try
        load(files{ii});
    catch
        file_loaded=0;
    end
    if file_loaded
        features  = TrialData.SmoothedNeuralFeatures;
        kinax = TrialData.TaskState;
        kinax = [find(kinax==3)];
        temp = cell2mat(features(kinax));
        
        % get the pooled data
        new_temp=[];
        [xx yy] = size(TrialData.Params.ChMap);
        for k=1:size(temp,2)
            tmp1 = temp(129:256,k);tmp1 = tmp1(TrialData.Params.ChMap);
            tmp2 = temp(513:640,k);tmp2 = tmp2(TrialData.Params.ChMap);
            tmp3 = temp(769:896,k);tmp3 = tmp3(TrialData.Params.ChMap);
            tmp3_smoothed=zeros(size(tmp3));
            tmp3_pooled=[];
            ix=1;
            iy=1;
            pooled_data=[];            
            for i=1:2:xx
                for j=1:2:yy
                    delta = (tmp1(i:i+1,j:j+1));delta=mean(delta(:));
                    beta = (tmp2(i:i+1,j:j+1));beta=mean(beta(:));
                    hg = (tmp3(i:i+1,j:j+1));hg=mean(hg(:));
                    pooled_data = [pooled_data; delta; beta ;hg];
                    tmp3_smoothed(i:i+1,j:j+1) = hg;
                end
            end
            new_temp= [new_temp pooled_data];
        end
        temp=new_temp;

        %if TrialData.TargetID  ==  TrialData.SelectedTargetID
        % temp=temp(:,end-5:end);

        %%% take only the first 2.5s or 13 bins
        %len_data = min(20,size(temp,2)); % this is for only imagined movement data
        %len_data = min(12,size(temp,2)); % this is the first 2.4s
        % temp = temp(:,1:len_data);

        %%%% get all neural if intended direction towards target (within
        %%%% 45deg bound)
        kin = TrialData.CursorState;
        kin = kin(1:3,kinax);
        target_pos = TrialData.TargetPosition;
        idx=[];
        angles=[];
        % get the ideal vector towards target
        pos2target = target_pos' - kin(:,1);
        chk=[];
        for i=1:size(kin,2)
            % compute the angle between ideal vector and intended vector.
            % this should be less than 45 degrees
            int_dir = target_pos' - kin(:,i);            
            ang = acosd((pos2target'*int_dir)/(norm(pos2target)*norm(int_dir)));            
%             if ang>90
%                 ang=180-ang;
%                 chk=[chk i];
%             end
%             if abs(ang)<45
%                 idx=[idx i];
%             end
            if ang<45 && ang>=0
                idx = [idx i];
            end
            angles(i)=ang;
        end
%         figure;
%         hold on
%         for i=1:size(kin,2)
%             if sum(idx==i) 
%                 col='r';
%             else
%                 col='k';
%             end
%             plot3(kin(1,i),kin(2,i),kin(3,i),'.','Color',col,'MarkerSize',20)
%         end
%         plot3(target_pos(1),target_pos(2),target_pos(3),'.r','MarkerSize',50)
%         plot3(kin(1,1),kin(2,1),kin(3,1),'.g','MarkerSize',50)
%         figure;stem(angles)
%         hline(30)

        temp = temp(:,idx);
        if TrialData.TargetID == 1
            D1 = [D1 temp];
        elseif TrialData.TargetID == 2
            D2 = [D2 temp];            
        elseif TrialData.TargetID == 3
            D3 = [D3 temp];
        elseif TrialData.TargetID == 4
            D4 = [D4 temp];
        elseif TrialData.TargetID == 5
            D5 = [D5 temp];
        elseif TrialData.TargetID == 6
            D6 = [D6 temp];
        elseif TrialData.TargetID == 7
            D7 = [D7 temp];
        end
        %end
    end
end


clear condn_data
idx = [1:96];
condn_data{1}=[D1(idx,:) ]'; % right thumb
condn_data{2}= [D2(idx,:)]'; % left leg
condn_data{3}=[D3(idx,:)]'; % left thumb
condn_data{4}=[D4(idx,:)]'; % head
condn_data{5}=[D5(idx,:)]'; % lips
condn_data{6}=[D6(idx,:)]'; % tongue
if size(D7,1)>0
    condn_data{7}=[D7(idx,:)]'; % both middle fingers
end

% 2norm
for i=1:length(condn_data)
    tmp = condn_data{i};
    for j=1:size(tmp,1)
        tmp(j,:) = tmp(j,:)./norm(tmp(j,:));
    end
    condn_data{i}=tmp;
end

end




