load('X:\Active\zf_feeding\fmr1_outputFiles\outputFiles\output-prey_event_ranged_genotype.mat')
path2save = 'C:\Users\zhu.s\Data\QIMR\PDF\hunting\';
load('colorbrewer.mat')
ctarget = colorbrewer.seq.Greens{1,9};
cprey = colorbrewer.seq.Reds{1,9};
%%
score = d.event_score;
for n = 1:size(score,2)
    target_a = d.target_angle(n);
    prey_d = cell2mat(d.prey_dist(n));
    prey_a = cell2mat(d.prey_angle(n));
    %%% calculate paramecia location given fish eye mid point at (0,0);
    p_coord = [];
    for pn = 1:length(prey_d)
        if sign(target_a*prey_a(pn))>0
           p_coord(pn,1) = prey_d(pn)*cosd(abs(prey_a(pn)));
           p_coord(pn,2) = prey_d(pn)*sind(abs(prey_a(pn)));
        else
           p_coord(pn,1) = prey_d(pn)*cosd(-abs(prey_a(pn)));
           p_coord(pn,2) = prey_d(pn)*sind(-abs(prey_a(pn)));
        end
    end
    prey_coord{n} = p_coord;
end
target_coord(:,1) = d.target_dist.*cosd(d.target_angle);
target_coord(:,2) = d.target_dist.*sind(d.target_angle);
target_dist = d.target_dist;
%%
het_idx = strcmp(d.genotype,'HOM');
het_target_coord = target_coord(het_idx,:);
het_prey_coord = prey_coord(1,het_idx);
het_target_dist = target_dist(:,het_idx);
het_score = score(:,het_idx);
%%
imgscale = 62;
nlvl = 6;
%% plot target location
fh = figure();
ax = plot_densityContourf(het_target_coord,nlvl,imgscale,ctarget./255);
hold on
plot(het_target_coord(:,1)./imgscale,het_target_coord(:,2)./imgscale,'k.','MarkerSize',2)
title('all')
ax.XLim = [-5 5];ax.YLim = [-5 5];
ax.XLabel.String = 'mm';ax.YLabel.String = 'mm';
set(gcf, 'Color', 'w', 'Units', 'Centimeters', 'Position', [0 0 6 6],'PaperUnits', 'Centimeters','papersize',[6 6]);
filename = strcat(path2save,'FA','-target_all_range_hom');
print('-painters',filename,'-dpdf','-r0')

s = 0;
sidx = find(het_score(:)==s);
sub_prey_coord = het_target_coord(sidx,:);
D2P =sub_prey_coord;
fh = figure();
ax = plot_densityContourf(D2P,nlvl,imgscale,ctarget./255);
hold on
plot(D2P(:,1)./imgscale,D2P(:,2)./imgscale,'k.','MarkerSize',2)
title('abort')
ax.XLim = [-5 5];ax.YLim = [-5 5];
ax.XLabel.String = 'mm';ax.YLabel.String = 'mm';
set(gcf, 'Color', 'w', 'Units', 'Centimeters', 'Position', [0 0 6 6],'PaperUnits', 'Centimeters','papersize',[6 6]);
filename = strcat(path2save,'FA','-target_s0_range_hom');
print('-painters',filename,'-dpdf','-r0')

s = 1;
sidx = find(het_score(:)==s);
sub_prey_coord = het_target_coord(sidx,:);
D2P =sub_prey_coord;
fh = figure();
ax = plot_densityContourf(D2P,nlvl,imgscale,ctarget./255);
hold on
plot(D2P(:,1)./imgscale,D2P(:,2)./imgscale,'k.','MarkerSize',2)
title('miss')
ax.XLim = [-5 5];ax.YLim = [-5 5];
ax.XLabel.String = 'mm';ax.YLabel.String = 'mm';
set(gcf, 'Color', 'w', 'Units', 'Centimeters', 'Position', [0 0 6 6],'PaperUnits', 'Centimeters','papersize',[6 6]);
filename = strcat(path2save,'FA','-target_s1_range_hom');
print('-painters',filename,'-dpdf','-r0')

s = 2;
sidx = find(het_score(:)>=s);
sub_prey_coord = het_target_coord(sidx,:);
D2P =sub_prey_coord;
fh = figure();
ax = plot_densityContourf(D2P,nlvl,imgscale,ctarget./255);
hold on
plot(D2P(:,1)./imgscale,D2P(:,2)./imgscale,'k.','MarkerSize',2)
title('capture')
ax.XLim = [-5 5];ax.YLim = [-5 5];
ax.XLabel.String = 'mm';ax.YLabel.String = 'mm';
set(gcf, 'Color', 'w', 'Units', 'Centimeters', 'Position', [0 0 6 6],'PaperUnits', 'Centimeters','papersize',[6 6]);
filename = strcat(path2save,'FA','-target_s3_range_hom');
print('-painters',filename,'-dpdf','-r0')
%% plot prey location
D2P =cell2mat(het_prey_coord');
fh = figure();
ax = plot_densityContourf(D2P,nlvl,imgscale,cprey./255);
hold on
plot(D2P(:,1)./imgscale,D2P(:,2)./imgscale,'k.','MarkerSize',2)
title('all')
ax.XLim = [-5 5];ax.YLim = [-5 5];
ax.XLabel.String = 'mm';ax.YLabel.String = 'mm';
set(gcf, 'Color', 'w', 'Units', 'Centimeters', 'Position', [0 0 6 6],'PaperUnits', 'Centimeters','papersize',[6 6]);
filename = strcat(path2save,'FA','-prey_all_range_hom');
print('-painters',filename,'-dpdf','-r0')

s = 0;
sidx = find(het_score(:)==s);
sub_prey_coord = het_prey_coord(1,sidx);
D2P =cell2mat(sub_prey_coord');
fh = figure();
ax = plot_densityContourf(D2P,nlvl,imgscale,cprey./255);
hold on
plot(D2P(:,1)./imgscale,D2P(:,2)./imgscale,'k.','MarkerSize',2)
title('abort')
ax.XLim = [-5 5];ax.YLim = [-5 5];
ax.XLabel.String = 'mm';ax.YLabel.String = 'mm';
set(gcf, 'Color', 'w', 'Units', 'Centimeters', 'Position', [0 0 6 6],'PaperUnits', 'Centimeters','papersize',[6 6]);
filename = strcat(path2save,'FA','-prey_s0_range_hom');
print('-painters',filename,'-dpdf','-r0')

s = 1;
sidx = find(het_score(:)==s);
sub_prey_coord = het_prey_coord(1,sidx);
D2P =cell2mat(sub_prey_coord');
fh = figure();
ax = plot_densityContourf(D2P,nlvl,imgscale,cprey./255);
hold on
plot(D2P(:,1)./imgscale,D2P(:,2)./imgscale,'k.','MarkerSize',2)
title('miss')
ax.XLim = [-5 5];ax.YLim = [-5 5];
ax.XLabel.String = 'mm';ax.YLabel.String = 'mm';
set(gcf, 'Color', 'w', 'Units', 'Centimeters', 'Position', [0 0 6 6],'PaperUnits', 'Centimeters','papersize',[6 6]);
filename = strcat(path2save,'FA','-prey_s1_range_hom');
print('-painters',filename,'-dpdf','-r0')

s = 2;
sidx = find(het_score(:)>=s);
sub_prey_coord = het_prey_coord(1,sidx);
D2P =cell2mat(sub_prey_coord');
fh = figure();
ax = plot_densityContourf(D2P,nlvl,imgscale,cprey./255);
hold on
plot(D2P(:,1)./imgscale,D2P(:,2)./imgscale,'k.','MarkerSize',2)
title('capture')
ax.XLim = [-5 5];ax.YLim = [-5 5];
ax.XLabel.String = 'mm';ax.YLabel.String = 'mm';
set(gcf, 'Color', 'w', 'Units', 'Centimeters', 'Position', [0 0 6 6],'PaperUnits', 'Centimeters','papersize',[6 6]);
filename = strcat(path2save,'FA','-prey_s3_range_hom');
print('-painters',filename,'-dpdf','-r0')
%% plot prey count dist
s = 0;
sidx = find(het_score(:)==s);
sub_prey_coord = het_prey_coord(1,sidx);
PCount_0 = [];
for sn = 1:size(sub_prey_coord,2)
   pc = size(sub_prey_coord{1,sn},1);
   PCount_0(sn) = pc;
end
xi = -0.5:1:10.5;
[fz0,~] = histcounts(PCount_0,xi);

s = 1;
sidx = find(het_score(:)==s);
sub_prey_coord = het_prey_coord(1,sidx);
PCount_1 = [];
for sn = 1:size(sub_prey_coord,2)
   pc = size(sub_prey_coord{1,sn},1);
   PCount_1(sn) = pc;
end
xi = -0.5:1:10.5;
[fz1,~] = histcounts(PCount_1,xi);

s = 2;
sidx = find(het_score(:)>=s);
sub_prey_coord = het_prey_coord(1,sidx);
PCount_3 = [];
for sn = 1:size(sub_prey_coord,2)
   pc = size(sub_prey_coord{1,sn},1);
   PCount_3(sn) = pc;
end
xi = -0.5:1:10.5;
[fz3,~] = histcounts(PCount_3,xi);

figure();hold on
x = 0:1:10;
f0 = plot(x,fz0./length(PCount_0),'LineWidth',2,'Color',cprey(8,:)./255);
f1 = plot(x,fz1./length(PCount_1),'LineWidth',2,'Color',cprey(5,:)./255);
f3 = plot(x,fz3./length(PCount_3),'LineWidth',2,'Color',cprey(3,:)./255);
plot([mean(PCount_0),mean(PCount_0)],[0,1],':','Color',cprey(8,:)./255);
plot([mean(PCount_1),mean(PCount_1)],[0,1],':','Color',cprey(5,:)./255);
plot([mean(PCount_3),mean(PCount_3)],[0,1],':','Color',cprey(3,:)./255);
legend([f0,f1,f3],'abort','miss','capture');
legend('box','off')
set(gca,'FontSize',12,'box','off');
xlabel('# of paramecium')
ylabel('probability')
set(gcf, 'Color', 'w', 'Units', 'Centimeters', 'Position', [0 0 8 6],'PaperUnits', 'Centimeters','papersize',[8 6]);
filename = strcat(path2save,'FA','-prey_dist_range_hom');
print('-painters',filename,'-dpdf','-r0')

%% plot target distance dist
s = 0;
sidx = find(het_score(:)==s);
sub_target_dist = het_target_dist(1,sidx)./imgscale;
xi = [0:1:20];
fz0 = ksdensity(sub_target_dist,xi);

s = 1;
sidx = find(het_score(:)==s);
sub_target_dist = het_target_dist(1,sidx)./imgscale;
xi = [0:1:20];
fz1 = ksdensity(sub_target_dist,xi);

s = 2;
sidx = find(het_score(:)>=s);
sub_target_dist = het_target_dist(1,sidx)./imgscale;
xi = [0:1:20];
fz3 = ksdensity(sub_target_dist,xi);

figure();hold on
plot(xi,fz0,'LineWidth',2,'Color',ctarget(8,:)./255);
plot(xi,fz1,'LineWidth',2,'Color',ctarget(5,:)./255);
plot(xi,fz3,'LineWidth',2,'Color',ctarget(3,:)./255);
legend('abort','miss','capture');
legend('box','off')
set(gca,'FontSize',12,'box','off');
xlabel('detection distance (mm)')
ylabel('density')
set(gcf, 'Color', 'w', 'Units', 'Centimeters', 'Position', [0 0 8 6],'PaperUnits', 'Centimeters','papersize',[8 6]);
filename = strcat(path2save,'FA','-target_dist_range_hom');
print('-painters',filename,'-dpdf','-r0')
%%
function ax = plot_densityContourf(data2plot,nlvl,imgscale,clm)
x = data2plot(:,1);
y = data2plot(:,2);
upl = ceil(max([x(:);y(:)]));
lwl = ceil(min([x(:);y(:)]));
step = floor((upl-lwl)/50);
gridxy = lwl:step:upl;
[gridx,gridy] = meshgrid(gridxy,gridxy);
xi = [gridx(:),gridy(:)];
[f,~] = ksdensity(data2plot,xi);
%%
f2 = f./imgscale;   
levels = linspace(0,max(f2),nlvl);
mycolormap = clm;
X = gridx./imgscale;
Y = gridy./imgscale;
Z = reshape(f2,size(gridx));
contourf(X,Y,Z,levels);
colormap([1 1 1;mycolormap])
ax = gca;
ax.FontSize = 8;
ax.Box = 'off';
ax.YLimSpec = 'tight';
ax.XLimSpec = 'tight';
end
