% Class AdaptiveEncoding implements adaptive encoding for optimalization
%
% See for more info:
% Hansen N., Adaptive Encoding for Optimization
% http://hal.inria.fr/docs/00/27/64/76/PDF/hansen-INRIA-RR-6518.pdf
%
% Implemented by Vaclav Klems; kvas@seznam.cz

classdef AdaptiveEncoding
   
   properties
      DIM         % orig. optimization problem param.: dimension of the 
      popsize     % orig. optimization problem param.: number of paralel solution candidates
      
      mu          % number of best solution candidates used for AE-update; usualy mu = floor(popsize/2)
      w_i         % recombination weights for mean m counting; w_i > 0 for i = 1:mu; sum(w_i) must equal to 1
      mu_w        % variance effective selection mass, equals to 1/sum(obj.w_i.^2)
      c_p         % learning rate for the evolution path p [1/sqrt(DIM); 2/(DIM+1)]
      c_1         % learning rate for rank-ONE-update (evolution path)
      c_mu        % learning rate for rank-MU-update (covariance matrix estimation)
      
      alp_p       % used for updating C
      alp_c       % used for counting c_1 and c_mu  
      alp_mu      % used for counting c_mu
      alp_0       % normalization factor when updating p (the direction is relevant and absolute size is disregarded)
      alp_i       % used for counting C_mu (rank-mu-update)
      
      m           % current mean
      m_old       % old mean
      
      p           % evolution path
      C           % covariance matrix
      B           % transformation matrix
      invB        % inversion of transformation matrix B
   end
   
   methods
      function obj = AdaptiveEncoding(DIM, popsize, mu, pop)
         % pop are column vector individuals - must be sorted belong fitness
         if nargin > 0
            % init basic params
            obj.alp_mu = 0.2;
            obj.alp_c = 1;
            obj.alp_p = 1;
                        
            % init algorithm parameters
            obj.DIM = DIM;
            obj.popsize = popsize;

            % init mu
            obj.mu = min(mu, popsize);    % can't use more then we really have         
            
               % set recombination weight
            obj.w_i = zeros(popsize,1);
            ln_sum = sum(log(1:mu));
            for i = 1:mu
               %ln_sum = ln_sum + log(i);
               obj.w_i(i) = (log(mu+1) - log(i)) / (mu*log(mu+1) - ln_sum);
            end
                       
            obj.mu_w = 1/sum(obj.w_i.^2);
            
            % init learning rates
            obj.c_p = 1/sqrt(DIM);
            obj.c_1 = (obj.alp_c*0.2) / ((DIM+1.3)^2+obj.mu_w); %speed
            obj.c_mu = (obj.alp_c * 0.2 *(obj.mu_w-2+(1/obj.mu_w))) / ((DIM+2)^2+obj.alp_mu*obj.mu_w) ; %speed

            % init variables
            obj.m = pop*obj.w_i;
            obj.p = zeros(DIM,1);
            obj.C = eye(DIM);
            obj.B = eye(DIM);
            obj.invB = eye(DIM);
            
            if ((obj.c_1 + obj.c_mu) > 1)
               fprintf('Setting AE failure - wrong values of c_1 and c_mu');
            end
            if (obj.c_p < 0)
               obj.c_p = 0;
            end
            if (obj.c_p > 1)
               obj.c_p = 1;
            end
         end
      end
      
      function obj = update(obj, pop)
         if sum(pop*obj.w_i == obj.m) == obj.DIM
            obj.m_old = zeros(obj.DIM,1);
         else
            obj.m_old = obj.m;
            obj.m = pop*obj.w_i;
         end
         
         obj.alp_0 = sqrt(obj.DIM)/norm(obj.invB*(obj.m - obj.m_old));

         obj.p = (1 - obj.c_p)*obj.p + sqrt(obj.c_p*(2-obj.c_p))*obj.alp_0*(obj.m-obj.m_old);
         
         obj.alp_i = zeros(size(pop,2),1);
         popminm_old = pop-repmat(obj.m_old,1,size(pop,2));  % pre-calculation
        
         % prepare obj.alp_i
            % conservative choise
   %          for i = 1:obj.mu
   %             obj.alp_i(i) = sqrt(obj.DIM)/norm(obj.invB*popminm_old(:,i));
   %          end

            % generaly recommanded choise
            beta = 2;
            auxiliary = obj.invB*popminm_old;
            l_i = zeros(obj.mu,1);
            for i = 1:obj.mu
               l_i(i) = norm(auxiliary(:,i));
            end
            medianl = median(l_i);
            for i = 1:obj.mu
               obj.alp_i(i) = max(l_i(i)/beta, medianl);
            end
        
         
         C_mu = zeros(obj.DIM,obj.DIM);
         for i = 1:obj.mu
            C_mu = C_mu + obj.w_i(i) * obj.alp_i(i)^2 * (popminm_old(:,i) * popminm_old(:,i)');
         end
         
         % obj.alp_p = sqrt(obj.DIM)/norm(obj.p)
         obj.C = (1-obj.c_1-obj.c_mu)*obj.C + obj.c_1*obj.alp_p*obj.p*obj.p' + obj.c_mu*C_mu;

         % enforce symmetry
         obj.C = tril(obj.C,0)+tril(obj.C,-1)';
         %obj.C = 0.5 * (obj.C + obj.C');

         %[obj.B,D,U] = svd(obj.C);
         [obj.B,D] = eig(obj.C);

         % limit condition of C to 1e14 + 1
         if max(diag(D)) > 1e14*min(diag(D))
             tmp = max(diag(D))/1e14 - min(diag(D));
             obj.C = obj.C + tmp*eye(obj.DIM); D = D + tmp*eye(obj.DIM);
         end

         % optionally normalize D
         D = diag(sqrt(diag(D)));
        
         obj.B = obj.B * D;
         
         obj.invB = inv(obj.B);
         %obj.invB = obj.B^-1;

         %showAE(obj,pop)
      end
      
      function pop = encode(obj,pop)
         if(~isempty(pop))
            pop = obj.invB * pop;
         end
      end
      
      function pop = decode(obj,pop)
         if(~isempty(pop))
            pop = obj.B * pop;
         end
      end
      
   end
end