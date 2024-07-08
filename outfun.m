function stop = outfun( p ,  optimValues , state ) 

    %this function defines stopping criteria for the optimization

    global history
    global first
    global tol

    global obj
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %how many initial optimization steps to run?
    initial = 100;

    %how many historic values are used in checking for convergence?
    back = 50;

    %two calls to fmincon at each step, if it is the first -> continue
    if first == 1
        stop = false;
        first = 0;

    %do things at second call of fmincon
    elseif first == 0
        
        %let optimization run for a while
        if length(history) < initial
            stop = false;

        %start checking for convergence
        else

            %past step sizes
            diffs = history( (end-back):(end-1) ) - history( (end-back+1):end );

            %maximum step size in look-back region
            delta = max(abs(diffs));
            
            %check for convergence in objective and constraint
            if delta < tol
                fprintf('converged ater %d steps',length(history))
                stop = true;
  
            %no objective convergence
            else
                stop = false;
            end

        end
        
        %since it was just the second call, the next will be the first
        first = 1;

        %keep track of objective values at every step
        history(end+1) = obj;

    end
end