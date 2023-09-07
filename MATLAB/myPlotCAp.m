function figure1 = myPlotCAp(net)

whitebg('white')

figure1 = figure;

% axes を作成
axes1 = axes('Parent',figure1);

axis equal

grid(axes1,'on');
% 残りの座標軸プロパティの設定
set(axes1,'DataAspectRatio',[1 1 1],'FontSize',24,'XTick',...
    [0.0 0.2 0.4 0.6 0.8 1.0],'YTick',[0.0 0.2 0.4 0.6 0.8 1.0]);
pause(0.01);

axes1.XMinorTick = 'on';
axes1.YMinorTick = 'on';


w = net.weight;
[N,D] = size(w);


% if D==2
%     plot(DATA(:,1),DATA(:,2),'cy.','MarkerSize',1);
% elseif D==3
%     scatter3(DATA(:,1),DATA(:,2),DATA(:,3),'cy.','MarkerFaceColor','cy');
% end

xlim([-0.05, 1.05])
ylim([-0.05, 1.05])
box on
hold on


% Change Node color based on LebelCluster
color = [
    [1 0 0]; 
    [0 1 0]; 
    [0 0 1]; 
%     [0 1 1]; 
    [1 0 1];
%     [1 1 0];
%     [0 0.4470 0.7410];
    [0.8500 0.3250 0.0980];
    [0.9290 0.6940 0.1250];
    [0.4940 0.1840 0.5560];
    [0.4660 0.6740 0.1880];
    [0.3010 0.7450 0.9330];
    [0.6350 0.0780 0.1840];
%     [1 1 1];
];
m = length(color);

for k = 1:N
    if D==2
%         plot(w(k,1),w(k,2),'.','Color',color(mod(label(1,k)-1,m)+1,:),'MarkerSize',35);
        plot(w(k,1),w(k,2),'.','Color',color(1,:),'MarkerSize',25);
    elseif D==3
%         plot3(w(k,1),w(k,2),w(k,3),'.','Color',color(mod(label(1,k)-1,m)+1,:),'MarkerSize',35);
        plot3(w(k,1),w(k,2),w(k,3),'.','Color',color(1,:),'MarkerSize',35);
    end
end




for i=1:N
    str = num2str(i); % Node Number
%     str = num2str(LebelCluster(1,i)); % Node Label
%     str = num2str(net.CountCluster(i)/max(TKBAnet.CountCluster));  % Norm Count Cluster
%     str = num2str(net.CountCluster(i));  % Count Cluster
%     str = num2str(net.adaptiveSig(1,i));
%     text(w(i,1)+0.01,w(i,2)+0.01, str,'Color','y','FontSize',14);
end


ytickformat('%.1f')
xtickformat('%.1f')

% axis equal
grid on
hold off
% axis([0 1 0 1]);
% axis([0 10 0 10]);
% title(figName,'fontsize',12);
% pause(0.01);

end