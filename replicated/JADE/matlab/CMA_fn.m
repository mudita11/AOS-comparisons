% CMA-ES for non-linear function minimization
% See also http://www.bionik.tu-berlin.de/user/niko
%
% Updated by Vaclav Klems to be used with COCO
% Updated by Vaclav Klems to use restarts

function[restartReason] = CMA_fn(FUN, argumentPack, ftarget, maxfunevals)

   % unpack arguments
   DIM = argumentPack{1};
   setCMA = argumentPack{2};

   % init return values
   restartReason = 'not restarted';
   
   % Set dimension, fitness fct, stop criteria, start values...
   %   stopfitness = 1e-10; % stop criteria
   xmeanw = ones(DIM, 1);    % object parameter start point (weighted mean)
   sigma = 1.0;            % step size
   minsigma = 1e-15;       % minimum step size
   maxfunevals = min(300*(DIM+2)^2, maxfunevals);

   % Parameter setting: selection,
   lambda = 4 + floor(3*log(DIM)); 
   mu = floor(lambda/2);
   arweights = log((lambda+1)/2) - log(1:mu)'; % for recombination

   % parameter setting: adaptation
   cc = 4/(DIM+4); 
   ccov = 2/(DIM+2^0.5)^2;
   cs = 4/(DIM+4); 
   damp = 1/cs + 1;

   % Initialize dynamic strategy parameters and constants
   B = eye(DIM); 
   D = eye(DIM); 
   BD = B*D; 
   C = BD*transpose(BD);
   pc = zeros(DIM,1); 
   ps = zeros(DIM,1);
   cw = sum(arweights)/norm(arweights);
   chiN = DIM^0.5*(1-1/(4*DIM)+1/(21*DIM^2));
   
   % Setup restarts
   xbest = 1e10;
   stuckcount = 0;
   minvarcondition = setCMA.minvarcondition;          % minimum variance which don't restart the algorithm
   stuckcond_noImp = setCMA.stuckcond_noImp;          % maximum interations without improvement after which algorithm is restarted
   stuckcond_lowVar = setCMA.stuckcond_lowVar;        % max. iter without improvement + low diversity condition

   

   % Generation loop
   counteval = 0; 
   arfitness(1) = 2*abs(ftarget)+1;
   while arfitness(1) > ftarget && counteval < maxfunevals
      % Generate and evaluate lambda offspring
      for k=1:lambda
         % repeat the next two lines until arx(:,k) is feasible
         arz(:,k) = randn(DIM,1);
         arx(:,k) = xmeanw + sigma * (BD * arz(:,k)); % Eq.(13)
         arfitness(k) = feval(FUN, arx(:,k));
         counteval = counteval+1;
      end
      % Sort by fitness and compute weighted mean
      [arfitness, arindex] = sort(arfitness); % minimization
      xmeanw = arx(:,arindex(1:mu))*arweights/sum(arweights);
      zmeanw = arz(:,arindex(1:mu))*arweights/sum(arweights);
      % Adapt covariance matrix
      pc = (1-cc)*pc + (sqrt(cc*(2-cc))*cw) * (BD*zmeanw); % Eq.(14)
      C = (1-ccov)*C + ccov*pc*transpose(pc); % Eq.(15)
      % adapt sigma
      ps = (1-cs)*ps + (sqrt(cs*(2-cs))*cw) * (B*zmeanw); % Eq.(16)
      sigma = sigma * exp((norm(ps)-chiN)/chiN/damp); % Eq.(17)
      % Update B and D from C
      if mod(counteval/lambda, DIM/10) < 1
         C=triu(C)+transpose(triu(C,1)); % enforce symmetry
         [B,D] = eig(C);
         % limit condition of C to 1e14 + 1
         if max(diag(D)) > 1e14*min(diag(D))
            tmp = max(diag(D))/1e14 - min(diag(D));
            C = C + tmp*eye(DIM); D = D + tmp*eye(DIM);
         end
         D = diag(sqrt(diag(D))); % D contains standard deviations now
         BD = B*D; % for speed up only
         end % if mod
         % Adjust minimal step size
         if ( sigma*min(diag(D)) < minsigma ) ...
                | ( arfitness(1) == arfitness(min(mu+1,lambda)) ) ...
                | ( xmeanw == xmeanw + 0.2*sigma*BD(:,1+floor(mod(counteval/lambda,DIM))) );
         	sigma = 1.4*sigma;
         end
        
      % STOPPING CRITERIA
      % interim stopping (algorithm is stucked in local optima)
      if feval(FUN, 'fbest') == xbest
         stuckcount = stuckcount + 1;
         if stuckcount >= stuckcond_noImp
            restartReason = 'no improvement';
            break;
         end
      else
         stuckcount = 0;
         xbest = feval(FUN, 'fbest');
      end
      %interim stopping (too low diversity)   not tested no CMA-ES
      if (stuckcount > stuckcond_lowVar) && (sum(var(arx,1,2))/DIM < minvarcondition)
         restartReason = 'low variance';
         break;
      end
   end % while, end generation loop

   %disp([num2str(counteval) ': ' num2str(arfitness(1))]);
   %xmin = arx(:, arindex(1)); % return best point of last generation
