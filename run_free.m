%% parameters

sampling_interval = 0.0005; % better be small
xrange = [-2:sampling_interval:2];
yrange = [-2:sampling_interval:2];
[X,Y] = meshgrid(xrange,yrange);

syms x y
%z = -1./sqrt(x.^2+y.^2+0.1);
z = -sqrt(8-x.^2-y.^2);
mf = matlabFunction(z);
Z = mf(X,Y);

start = [-0.5,2,mf(-0.5,2)];
direction = [0,-1,0];
step_forward = 0.2;
use_GPU = true;

%% main plotting process

% plot the surface
hold on
axis equal
%view(3); % 3D view
% use fewer points to improve performance
s = surf(X(1:10:end,1:10:end),Y(1:10:end,1:10:end),Z(1:10:end,1:10:end));
s.EdgeColor = 'none';

% plot the geodesics from different start points
route_set = [];
for i = -2:0.2:2
    % calc each geodesic
    start = [i,2,mf(i,2)];
    direction = [0,-1,0];
    route = geodesic_free(X,Y,Z,start,direction,step_forward,use_GPU);
    
    % plot all points on the geodesic with color
    % the color will repeat from black to red
    for j = 1:size(route,1)
        plot3(route(j,1),route(j,2),route(j,3),...
            '.',...
            'markersize',10,...
            'color',[(j-1)/10-fix((j-1)/10),0,0]);
        drawnow;
    end
end

hold off