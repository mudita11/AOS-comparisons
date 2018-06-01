warning off; addpath('/home/posik/P/code/MATLAB/toolbox'); warning on;

clc; clear;
more off;  % in octave pagination is on by default
newdiary();

%% Get the path to the bbob directory
% bbobpath = pwd;
% bbobdirname = 'bbob-local';
% ind = max(strfind(bbobpath,[filesep bbobdirname filesep]));
% bbobpath(ind+length(bbobdirname)+1:end) = [];
bbobpath = getBBOBRootPath('bbob-local');

%% Set the parameters of maxfunevals
Fopt.algName = 'CMA-ES multistart';  % different name for each experiment
Fopt.comments = 'BBOB 2012 settings, run with BIPOP-CMA-ES code';
maxfunevals = '5e4*dim';
minfunevals = 'dim+2';
maxrestarts = 1e4;

%% Define folders for results, for results of the particular algorithm and its variant
resultspath = 'results';
algopath = 'CMA-ES_Multistart';
variantpath = maxfunevals; 
variantpath = strrep(variantpath, '*dim', 'D');

% addpath([bbobpath '\matlab']);  % should point to fgeneric.m etc.
%% Path of data, different folder for each experiment
datapath = fullfile(bbobpath, resultspath, algopath, variantpath, ''); 
% datapath = '.';

%% Copy the m-files to the results
copypath = fullfile(datapath,'matlab','');
mkdir(copypath);
copyfile('*.m', copypath);

t0 = clock;
rand('state', sum(100 * t0));

dims = [2,3,5,10,20,40];
funcs = benchmarks('FunctionIndices');
instances = [1:5 21:30];

startdim = 2;
startifun = 1;
startinstance = 1;

% try

for dim = dims,  % small dimensions first, for CPU reasons
    if dim < startdim, continue; end
    for ifun = funcs,
        if (dim == startdim && ifun < startifun), continue; end
        for instance = instances,  
            if (dim == startdim && ifun == startifun && instance < startinstance),
                continue;
            end

            ftarget = fgeneric('initialize', ifun, instance, datapath, Fopt);

            [xmin, restarts] = ... 
                bbobRestCMAES('fgeneric', dim, ftarget, eval(maxfunevals), maxrestarts);

            fprintfts(['  f%d in %d-D, instance %d: FEs=%d with %d restarts,' ...
                ' fbest-ftarget=%.4e, elapsed time [h]: %.2f\n'], ...
                ifun, dim, instance, ...
                fgeneric('evaluations'), ...
                restarts, ...
                fgeneric('fbest') - fgeneric('ftarget'), ...
                etime(clock, t0)/60/60);
            
            fgeneric('finalize');
        end
        fprintfts('%s\n', ['      date and time: ' num2str(clock, ' %.0f')]);
    end
    fprintfts('---- dimension %d-D done ----\n', dim);
end

% catch ME,
%     
%     diary off;
%     fgeneric('finalize');
%     rethrow(lasterror);
%     
% end
