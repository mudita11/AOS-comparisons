% launch Matlab, then type 
%
%   help bipopcmaes
%
% and read the last few lines to understand how to reproduce a BBOB-2009 result. 
%
% The following code replicates the complete experiment: 

datapath = 'bipopcmaes-bbob-replication';
Foptions.algName = 'bipopcmaes'; 
Foptions.comments = 'replication of BBOB-2009 results'; 

for dim = [2,3,5,10,20,40]  % small dimensions first, for CPU reasons
  for ifun = 1:24
    for iinstance = [1 1 1 2 2 2 3 3 3 4 4 4 5 5 5]  % first 5 function instances, three times 

      opts.stopfit = fgeneric('initialize', ifun, iinstance, datapath, Foptions); 

      bipopcmaes('fgeneric', ['8 * rand(' num2str(dim) ', 1) - 4'], 2, opts);

      fgeneric('finalize');
      disp(['*** f' num2str(ifun) ', instance ' num2str(iinstance) ', done ***']);
    end
  end
end

