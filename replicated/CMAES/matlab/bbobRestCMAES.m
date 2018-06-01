function [x, restart] = bbobRestCMAES(FUN, ...
    DIM, ftarget, maxfunevals, maxrestarts, varargin)

    fevals = 0;
    % multistart such that ftarget is reached with reasonable prob.
    for restart = 0:maxrestarts,  % relaunch optimizer

        if restart > 0,
            strflag = strcat(stopflag, ',');
            strflag = strcat(strflag{:});
            strflag(end) = [];
            fgeneric('restart', ...
                sprintf('Indep. restart #%d, because of %s.', ...
                restart, strflag));
        end

        opt.DispFinal = 'off';
        opt.Restarts = '0';
        opt.MaxFunEvals = maxfunevals - fgeneric('evaluations');
        opt.StopFitness = ftarget;

        [x, fmin, counteval, stopflag, out, bestever, runs] = ...
            bipopcmaes(FUN, ['8 * rand(' num2str(DIM) ', 1) - 4'], 2, opt);
        fevals = fevals + counteval;

        if (fmin < ftarget) || (fevals >= maxfunevals), 
            break; 
        end
    end
end