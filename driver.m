clear; close all;

%driver script for 1D isotropic phase field model
%this script does UQ but not load stepping

global U1
global P1

global L

global quad_pts
global grd
global dx

global grdy
global dy

global integration_tensor

global Nu
global Np
global num_params

global D
global ell
global G

global b

global pts
global pgrid

global history
global first
global tol

global xi
global rho

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%problem parameters

%write files?
files = 1;

%number of random input parameters
xi = 1;

%network parameters
U1 = 6; 
P1 = 12;

%stopping criteria
tol = 2E-8;

%geometric parameters
L = 1; 

%body force
b0 = 15;
b = @(x) ( b0 * sin(pi*(x(1,:)-L/2)) );

%length scale
ell = 0.1;

%size of integration element
sz = 1.5E-2;

%relative size of spatial grid to parameter grid
reduce = 8;

%elastic material properties
D = 1E1;

%marginal distribution of random parameter
marg = @(y) ( -6*y.*(y-1) );

%fracture energy
if xi == 0
    G = @(x) (1+0*x);
elseif xi == 1
    G = @(x) (  1 - 0.5 * exp( -200 * ( x(1,:) - L*x(2,:) ).^2 ) );
    rho = @(y) ( marg( y ) );
elseif xi == 2
    G = @(x) (  1 - 0.5 * x(3,:) .* exp( -200 * ( x(1,:) - L*x(2,:) ).^2 ) );
    rho = @(y) ( marg( y(1) ) .* marg( y(2) ) );
elseif xi == 3
    G = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plotting

%number of points for plotting
pts = 35; pgrid = linspace(0,L,pts);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%integration

%number of spatial quadrature points
quad_pts = fix(L/sz);

%1d spatial grid
grd = linspace(0,L,quad_pts);
dx = grd(2)-grd(1);

%1d parameter grid
eps = 5E-2;
grdy = linspace(eps,1-eps,fix(quad_pts/reduce));
dy = grdy(2)-grdy(1);

%vectorized integration
integration_tensor = grd(1:end-1)+dx/2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%construct neural network
[Nu,Np] = one_hidden(files);
num_params = Nu + Np;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%energy minimization

%optmization options
evals = 5E5;
options = optimoptions('fmincon', ...
            'OptimalityTolerance', 0, ...
            'GradObj' , 'on' , ...
            'Algorithm' , 'sqp' , ...
            'StepTolerance', 0, ...
            'OutputFcn' , @outfun , ...
            'MaxFunctionEvaluations', evals,...
            'MaxIterations', evals);

%initial guess for parameters
p0 = .1*randn(num_params,1);
 
%initialize history of objective
history = 0;
first = 1;

%optimization
param = fmincon( @energy , p0 , [] , [] , [] , [] , [] , [] , [] , options );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

