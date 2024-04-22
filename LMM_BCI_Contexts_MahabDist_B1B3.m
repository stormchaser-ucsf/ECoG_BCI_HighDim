%% (MAIN) RUNNING LMM ON MAHAB DISTANCES FOR B1 AND B3
% to show that there is or is not a systematic trend across days 


clc;clear

% load B1 data
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker')
a=load('mahab_dist_B1_latent');

% load B3 data 
cd('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3')
b=load('mahab_dist_b3_latent');

b1 = a.tmp;
b3 = b.tmp;

b1=b3;

% run LMM non parametric test, within subject but combining CL contexts
day_name = [(1:size(b1,1))';(1:size(b1,1))'];
mahab_dist = [b1(:,2);b1(:,3)];
subj = ones(size(mahab_dist));
mvmt_type = [ones(size(b1,1),1);2*ones(size(b1,1),1)];
data = table(subj,day_name,mahab_dist,mvmt_type);

% fit
glm = fitlme(data,'mahab_dist ~  day_name + (1|mvmt_type) ')
stat = glm.Coefficients.Estimate

% permutation testing 
pval=[];stat_boot=[];
for i=1:500
    disp(i)
    a = day_name(1:length(b1));
    b = day_name(length(b1)+1:end);
    a=a(randperm(numel(a)));
    b=b(randperm(numel(b)));
    day_name_tmp = [a;b];
    data_tmp = table(subj,day_name_tmp,mahab_dist,mvmt_type);
    glm_tmp = fitlme(data_tmp,'mahab_dist ~ 1 + day_name_tmp + (1|mvmt_type) ');
    stat_boot = [stat_boot glm_tmp.Coefficients.tStat];
end
figure;
hist(stat_boot(2,:));
vline(stat(2))
sum(stat(2)<=stat_boot(2,:))/500


% plotting 
tmp=b1;
figure;
num_days=size(tmp,1);
xlim([0 num_days+1])
hold on
x= [ ones(size(tmp(:,1),1),1) (1:length(tmp(:,1)))'];
% imag
plot(1:num_days,tmp(:,1),'.b','MarkerSize',20)
y = tmp(:,1);
[B,BINT,R,RINT,STATS1] = regress(y,x);
yhat = x*B;
plot(1:num_days,yhat,'b','LineWidth',1)
% online
%plot(1:num_days,tmp(:,2),'.k','MarkerSize',20)
plot(1:num_days,tmp(:,2),'.','Color','k','MarkerSize',20)
y = tmp(:,2);
[B,BINT,R,RINT,STATS2] = regress(y,x);
yhat = x*B;
%plot(1:num_days,yhat,'Color',[.65 .65 .65 .35],'LineWidth',1)
% batch
plot(1:num_days,tmp(:,3),'.r','MarkerSize',20)
y = tmp(:,3);
[B,BINT,R,RINT,STATS3] = regress(y,x);
yhat = x*B;
%plot(1:num_days,yhat,'Color',[1 0 0 .35],'LineWidth',1)
set(gcf,'Color','w')
set(gca,'LineWidth',1)
xticks([1:num_days])

bhat = glm.Coefficients(:,2).Estimate;
yhat = x*bhat;
plot(1:num_days,yhat,'m','LineWidth',2)


% run LMM non parametric test, one context at a time 
pval=[];stat_overall=[];
for context = 1:size(b1,2)
    day_name=[];
    mahab_dist=[];
    subj=[];

    % get B1 data
    day_name = [day_name;(1:size(b1,1))';11];
    mahab_dist = [mahab_dist;b1(:,context);NaN];
    subj = [subj;ones(size(b1,1),1);1];

    % get B3 data
    day_name = [day_name;(1:size(b3,1))'];
    mahab_dist = [mahab_dist;b3(:,context)];
    subj = [subj;2*ones(size(b3,1),1)];

    % collate
    data = table(day_name,mahab_dist,subj);

    % fit
    glm = fitlme(data,'mahab_dist ~ 1+(day_name) + (1|subj)');

    % run boot statistics 
    stat = glm.Coefficients.tStat(2);
    stat_overall(context)=stat;
    pval(context) = glm.Coefficients.pValue(2);
    stat_boot=[];
    parfor i=1:1000
        disp(i)
        aa = day_name(1:11);
        aa=aa(randperm(numel(aa)));
        bb = day_name(12:end);
        bb=bb(randperm(numel(bb)));
        day_name_tmp = [aa;bb];
        data_tmp = table(day_name_tmp,mahab_dist,subj);
        glm_tmp = fitglme(data_tmp,'mahab_dist ~ 1 + (day_name_tmp) + (1|subj)');
        stat_boot(i) = glm_tmp.Coefficients.tStat(2);
    end

    figure;
    hist(abs(stat_boot),20);
    vline(abs(stat),'r')
    pval(context)= sum(abs(stat_boot)>stat)/length(stat_boot);
end
pval

% boxplots comparing OL, CL1 and CL2 
figure;hold on
boxplot([b1;b3])
a = get(get(gca,'children'),'children');
for i=1:length(a)
    box1 = a(i);
    set(box1, 'Color', 'k');
end
x1= [1+ 0.1*randn(length(b1),1) 2+ 0.1*randn(length(b1),1) 3+ 0.1*randn(length(b1),1)];
h=scatter(x1,b1,'filled');
for i=1:3
    h(i).MarkerFaceColor = 'r';
    h(i).MarkerFaceAlpha = 0.5;
end
x1= [1+ 0.1*randn(length(b3),1) 2+ 0.1*randn(length(b3),1) 3+ 0.1*randn(length(b3),1)];
h=scatter(x1,b3,'filled');
for i=1:3
    h(i).MarkerFaceColor = 'b';
    h(i).MarkerFaceAlpha = 0.5;
end
xlim([0.5 3.5])
xticks(1:3)
xticklabels({'OL','CL1','CL2'})
yticks([0:10:80])
ylim([0 80])
set(gcf,'Color','w')
box off