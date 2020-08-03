function [route] = geodesic_free(X,Y,Z,start,direction,step_forward,use_GPU)
% To calculate an estimation of the geodesic of a given surface.
% input:
% X,Y -- matrix, [X,Y]=meshgrid(xrange,yrange)
% Z -- matrix, Z = z(X,Y)
% start -- vector, the start point [x,y,z(x,y)] on the surface
% direction -- vector, the start direction [dir_x,dir_y,dir_z]
% step_forward -- scalar, decides the distance between two points on the
%                 geodesic, usually smaller than surface's sampling interval
% use_GPU -- bool scalar, use GPU to compute if true
% output:
% route -- matrix, a col vector of all the 3D points on the geodesic
%
% Wanghao Xu, last edit time: 8/3/2020

ITER_MAX = 1000;

if nargin == 0
    xrange = [-2:0.01:2];
    yrange = [-2:0.01:2];
    [X,Y] = meshgrid(xrange,yrange);
    
    syms x y
    z = -1./sqrt(x.^2+y.^2+0.1); % surface equation
    mf = matlabFunction(z);
    Z = mf(X,Y);
    
    start = [-0.5,2,mf(-0.5,2)];
    direction = [0,-1,0];
    step_forward = 0.1;
    use_GPU = false;
end

% initialization
route = [];
cur_point = start;
cur_direction = direction/norm(direction);
iter = 0;
if use_GPU
    X = gpuArray(single(X));
    Y = gpuArray(single(Y));
    Z = gpuArray(single(Z));
end
fprintf('start iteration\n');
while iter < ITER_MAX
    route = [route;cur_point];
    cur_point = cur_point + step_forward*cur_direction;
    
    % check whether the current point is in the boundary
    if cur_point(1)>max(max(X)) || cur_point(2)>max(max(X))...
            || cur_point(1)<min(min(X)) || cur_point(2)<min(min(Y))
        %route = [route;cur_point];
        break;
    end
    
    % calculate the euclidean distance between cur point
    %  and other points on surface
    dX = X - cur_point(1);
    dY = Y - cur_point(2);
    dZ = Z - cur_point(3);
    distance = sqrt(dX.*dX + dY.*dY + dZ.*dZ);
    % select the closest point on surface as the new current point
    idx = find(distance==min(min(distance)));
    cur_point = [X(idx(1)),Y(idx(1)),Z(idx(1))]; % might be multiple idx, use idx(1)
    
    % calculate new direction vector
    cur_direction = cur_point - route(end,:);
    % deal with zero direction vector
    if ~any(cur_direction,'all')
        % zero direction vector usually implies the algo is stuck around the
        % extreme point or the step_forward is too small
        warning('zero direction vector appears, geodesic stop extending');
        break;
    end
    % normalize the non-zero direction vector
    cur_direction = cur_direction/norm(cur_direction);
    
    iter = iter + 1;
end

if use_GPU
    gather(route);
end
fprintf('calculation finished after %d iteration(s)\n',iter);

end