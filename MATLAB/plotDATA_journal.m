function figure1 = plotDATA_journal(DATA)


whitebg('white')

% figure を作成
figure1 = figure;

% axes を作成
axes1 = axes('Parent',figure1);

axis equal

% Axes の X 軸の範囲を保持するために以下のラインのコメントを解除
% xlim(axes1,[0 1]);
% Axes の Y 軸の範囲を保持するために以下のラインのコメントを解除
% ylim(axes1,[0 1]);
grid(axes1,'on');
% 残りの座標軸プロパティの設定
set(axes1,'DataAspectRatio',[1 1 1],'FontSize',24,'XTick',...
    [0.0 0.2 0.4 0.6 0.8 1.0],'YTick',[0.0 0.2 0.4 0.6 0.8 1.0]);
pause(0.01);

axes1.XMinorTick = 'on';
axes1.YMinorTick = 'on';

xlim([-0.05, 1.05])
ylim([-0.05, 1.05])
box on
hold on
X = DATA;

plot(X(:,1),X(:,2),'k.','MarkerSize',8);
% axis([0 1 0 1]);
ytickformat('%.1f')
xtickformat('%.1f')





hold off

end