function [ obj , grad ] = energy( p )

%computes variational energy and its gradient for use in minimization

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

global obj
global history

global step
global steps

global xi
global rho

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%displacement parameters
p_u = p(1:Nu);

%phase field parameters
p_p = p(Nu+1:end);

%parameter for numerical stability
k = 1E-2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%strain energy density
bulk_u = @(x) ( 0.5 * D * ( ( ( P(x,p_p)-1 ).^2 + k ) .* Eps(x,p_u) ) * Eps(x,p_u)' );

%volumetric work
work_u = @(x) ( b(x) * U(x,p_u)' );

%fracture energy density
bulk_p = @(x) ( (0.5/ell) * ( sum(  G(x).*( P(x,p_p).^2 + ell^2 * flux(x,p_p).^2 ) )  ) );

%parameter gradient of strain energy density
bulk_u_grad_u = @(x) ( D * Eps_grad(x,p_u) * ( ( ( P(x,p_p)'-1 ).^2 + k ) .* Eps(x,p_u)' ) );

%parameter gradient of strain energy density
bulk_u_grad_p = @(x) ( D * P_grad(x,p_p) * ( ( P(x,p_p)' - 1 ) .* (Eps(x,p_u).^2)' ) );

%parameter gradient of volumetric work
work_u_grad = @(x) ( U_grad(x,p_u) * b(x)' );

%parameter gradient of fracture energy density
bulk_p_grad = @(x) ( ( P_grad(x,p_p)*( (G(x)/ell ) .*P (x,p_p) )' + flux_grad(x,p_p)*( (G(x)*ell) .* flux(x,p_p) )' ) );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%integrate to form energy and gradient

padu = zeros(Nu,1); padp = zeros(Np,1);
relax = 0.85;
obj = 0; grad = zeros(num_params,1);

%no random parameters
if xi == 0
    %build integration input
    inp = integration_tensor;

    obj = obj + dx * ( bulk_u(inp) - work_u(inp) + bulk_p(inp) );
    grad = grad + relax * dx * ( [ bulk_u_grad_u(inp) ; padp ] + [ padu ; bulk_u_grad_p(inp) ] - [ work_u_grad(inp) ; padp ] + [ padu ; bulk_p_grad(inp) ] );

%one random parameter
elseif xi == 1

    for i=1:length(grdy)
    
        %random sample for parameters
        ypt = grdy(i);
    
        %build integration input
        inp = zeros(xi+1,quad_pts-1);
        inp(1,:) = integration_tensor;
        inp(2:end,:) = ypt .* ones(xi,quad_pts-1);

        %expected energy and gradient
        obj = obj + dx * dy * rho(ypt) * ( bulk_u(inp) - work_u(inp) + bulk_p(inp) );
        grad = grad + relax * dx * dy * rho(ypt) * ( [ bulk_u_grad_u(inp) ; padp ] + [ padu ; bulk_u_grad_p(inp) ] - [ work_u_grad(inp) ; padp ] + [ padu ; bulk_p_grad(inp) ] );

    end

%two random parameters
elseif xi == 2
    
    for i=1:length(grdy)
        for j=1:length(grdy)

            %random sample for parameters
            ypt = [ grdy(i); grdy(j) ];

            %build integration input
            inp = zeros(xi+1,quad_pts-1);
            inp(1,:) = integration_tensor;
            inp(2:end,:) = ypt .* ones(xi,quad_pts-1);

            %expected energy and gradient
            obj = obj + dx * dy^2 * rho(ypt) * ( bulk_u(inp) - work_u(inp) + bulk_p(inp) );
            grad = grad + relax * dx * dy^2 * rho(ypt) * ( [ bulk_u_grad_u(inp) ; padp ] + [ padu ; bulk_u_grad_p(inp) ] - [ work_u_grad(inp) ; padp ] + [ padu ; bulk_p_grad(inp) ] );

        end
    end
    

%three random parameters
elseif xi == 3
    disp('god help you')




end

% disp( norm( [ padu ; bulk_u_grad_p(inp) ] + [ padu ; bulk_p_grad(inp) ]    )  )


if mod(length(history)+1,150) == 0

    

    %displacement parameters
    p_u = p(1:Nu);
    
    %phase field parameters
    p_p = p(Nu+1:end);
    
    %plot converged solution at each load step
    pf1 = zeros(pts,1); u1 = zeros(pts,1); g1 = zeros(pts,1);
    pf2 = zeros(pts,1); u2 = zeros(pts,1); g2 = zeros(pts,1);
    pf3 = zeros(pts,1); u3 = zeros(pts,1); g3 = zeros(pts,1);

    for k=1:pts
    
        pt = pgrid(k);
    
        arg1 = zeros(xi+1,1);
        arg1(1) = pt;

        pf1(k) = P( arg1 ,  p_p );
        u1(k) = U( arg1 , p_u );
        g1(k) = G( arg1 );

        %%%

        arg2 = ones(xi+1,1);
        arg2(1) = pt;

        pf2(k) = P( arg2 ,  p_p );
        u2(k) = U( arg2 , p_u );
        g2(k) = G( arg2 );

        %%%

        arg3 = 0.5*ones(xi+1,1);
        arg3(1) = pt;

        pf3(k) = P( arg3 ,  p_p );
        u3(k) = U( arg3 , p_u );
        g3(k) = G( arg3 );
        
    
    end
    
    subplot(3,3,1)
    plot( pgrid , u1 )
    xlabel('x1')
    title('Displacement')
    grid on
    drawnow
    
    subplot(3,3,2)
    plot( pgrid , pf1 )
    xlabel('x1')
    title('Phase Field')
    grid on
    drawnow

    subplot(3,3,3)
    plot( pgrid , g1 )
    xlabel('x1')
    title('Fracture Energy')
    grid on
    drawnow

    %%%

    subplot(3,3,4)
    plot( pgrid , u2 )
    xlabel('x1')
    title('Displacement')
    grid on
    drawnow
    
    subplot(3,3,5)
    plot( pgrid , pf2 )
    xlabel('x1')
    title('Phase Field')
    grid on
    drawnow

    subplot(3,3,6)
    plot( pgrid , g2 )
    xlabel('x1')
    title('Fracture Energy')
    grid on
    drawnow

    %%%

    subplot(3,3,7)
    plot( pgrid , u3 )
    xlabel('x1')
    title('Displacement')
    grid on
    drawnow
    
    subplot(3,3,8)
    plot( pgrid , pf3 )
    xlabel('x1')
    title('Phase Field')
    grid on
    drawnow

    subplot(3,3,9)
    plot( pgrid , g3 )
    xlabel('x1')
    title('Fracture Energy')
    grid on
    drawnow

    

end
