function [ num_U_params , num_P_params ] = one_hidden(iflag)

%construct separate one hidden layer neural networks for displacement and phase field
%approximations with given sizes

%build functions required for energy minimization and write them to files

global U1;
global P1;

global xi;

global L;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%number of nodes at input layer
inp = xi + 1;

%count parameters, 2d displacement and scalar phase field
num_U_layer1 = inp*U1 + U1;
num_U_output = U1;

%total displacement parameter count
num_U_params = num_U_layer1 + num_U_output; 

num_P_layer1 = inp*P1 + P1;
num_P_output = P1;

%total phase field parameter count
num_P_params = num_P_layer1 + num_P_output; 

%total number of parameters
num_params = num_U_params + num_P_params;

fprintf('number of displacement parameters: %d\n',num_U_params)
fprintf('number of phase field parameters: %d\n',num_P_params)
disp('--------------------')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%construct network

if iflag

    %spatial coordinate and stochastic variable
    x = sym( 'x' , [inp,1] );

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %displacement neural network
    
    %symbolic displacement parameter vector
    p_u = sym( 'p_u' , [ num_U_params , 1 ] ); assume(p_u,'real'); 
    
    %first layer
    w1_u = reshape( p_u( 1:(inp*U1) ) , [U1,inp] );
    b1_u = p_u( (inp*U1+1):inp*U1+U1 );
    
    %output layer
    w2_u =  p_u( (inp*U1+U1+1):(inp*U1+2*U1) );
    
    %define forward pass
    first = act( w1_u*x + b1_u );

    %zero displacement boundaries
    shape = sin( pi * x(1) / L ) * first;
    
    %output
    U = w2_u' * shape;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %phase field neural network

    %parameter for hyperbolic tangent enforcing phase field normalization
    p = 0.5;

    %symbolic displacement parameter vector
    p_p = sym( 'p_p' , [ num_P_params , 1 ] ); assume(p_p,'real'); 
    
    %first layer
    w1_p = reshape( p_p( 1:(inp*P1) ) , [P1,inp] );
    b1_p = p_p( (inp*P1+1):inp*P1+P1 );
    
    %output layer
    w2_p = p_p( (inp*P1+P1+1):(inp*P1+2*P1) ) ;
    
    %define forward pass
    first = act( w1_p*x + b1_p );

    %zero Neumann boundaries, enforced weakly
    shape = first;

    %output, enforcing phase field is in [0,1]
    P = 0.5 * ( 1 + tanh( p *  w2_p' * shape ) );
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %symbolic physics
    
    disp('begin symbolic calculations...')

    %displacement calculations

    %strain vector
    epsilon1 = diff( U , x(1) , 1 );
    eps = epsilon1; 

    %parameter gradient of strain vector
    eps_grad = sym( zeros( num_U_params , 1 ) );
    for i=1:num_U_params
        eps_grad(i) = diff( eps , p_u(i) , 1 );
    end

    %parameter gradient of displacement vector
    U_grad = sym( zeros( num_U_params , 1 ) );
    for i=1:num_U_params
        U_grad(i) = diff( U , p_u(i) , 1 );
    end

    %phase field calculations

    %spatial gradient of phase field
    flux1 = diff(  P , x(1) , 1 );
    flux = flux1; 

    %parameter gradient of flux vector
    flux_grad = sym( zeros( num_P_params , 1 ) );
    for i=1:num_P_params
        flux_grad(i) = diff( flux , p_p(i) , 1 );
    end

    %parameter gradient of phase field
    P_grad = sym( zeros( num_P_params , 1 ) );
    for i=1:num_P_params
       P_grad(i) = diff( P , p_p(i) , 1 );
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %convert to numerical functions
    
    disp('writing functions to files...')

    %displacement
    matlabFunction( U ,'vars' , { x , p_u } , 'File' , 'U' , 'Optimize' , true );

    %displacement parameter gradient
    matlabFunction( U_grad ,'vars' , { x , p_u } , 'File' , 'U_grad' , 'Optimize' , true );

    %strain
    matlabFunction( eps ,'vars' , { x , p_u } , 'File' , 'Eps' , 'Optimize' , true );

    %strain parameter gradient
    matlabFunction( eps_grad ,'vars' , { x , p_u } , 'File' , 'Eps_grad' , 'Optimize' , true );

    % % %

    %phase field
    matlabFunction( P ,'vars' , { x , p_p } , 'File' , 'P' , 'Optimize' , true );

    %phase field parameter gradient
    matlabFunction( P_grad ,'vars' , { x , p_p } , 'File' , 'P_grad' , 'Optimize' , true );

    %phase field flux
    matlabFunction( flux ,'vars' , { x , p_p } , 'File' , 'flux' , 'Optimize' , true );

    %phase field flux parameter gradient
    matlabFunction( flux_grad ,'vars' , { x , p_p } , 'File' , 'flux_grad' , 'Optimize' , true );

    disp('done')
    disp('--------------------')

end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%activation function

function vec = act( layer )
    vec = tanh(  layer );
end




