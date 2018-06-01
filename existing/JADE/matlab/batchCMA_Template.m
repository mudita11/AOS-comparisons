%% SETTING CMA-ES - Template
clear all;

%-------- setExperiment --------------------
setExperiment.datapath = fullfile('C:', 'DP-Experiments', 'CMA_Template'); % path with the experiment data
setExperiment.name = 'CMA Template';     % name the experiment e.g.: 'DEAE with new feature'
setExperiment.comment = 'CMA 1e5DIMEF 80i60v';
setExperiment.maxfunevals = '1000 * dim'; %  e.g.: '10000 * dim'

setExperiment.dim = [2,3,5,10,20,40]; % e.g.: [2,3,5,10,20,40]
setExperiment.benchmark_fun = 'benchmarks(''FunctionIndices'')'; %e.g.: 'benchmarks(''FunctionIndices'')' e.g.: 'benchmarksnoisy(''FunctionIndices'')'
setExperiment.useFunctions = 'all';% e.g. 'all' or for noiseless:[1-24] or for noisy[101-130];
setExperiment.instance = [1:5 21:30];     % #instances = #trials e.g.: [1:15] for BBOB 2012: [1:5 21:30]

%-------- setDEAE --------------------------
setCMA.minvarcondition =  1e-10;    %e.g.: 1e-10   minimum variance which don't restart the algorithm
setCMA.stuckcond_noImp = 80;    %e.g.:50  maximum interations without improvement after which algorithm is restarted
setCMA.stuckcond_lowVar = 60;    %e.g: 30; 


try
   experimentCMA_fn(setExperiment, setCMA);
catch exception
   fprintf('The experiment: "%s" has FAILED:\n\n',setExperiment.name);
   disp(exception.message);
   
   
   filename_1 = fullfile(setExperiment.datapath, '\00_FAILURE.txt');
   file_1 = fopen(filename_1,'w');
   fprintf(file_1,'The experiment:  "%s" has FAILED:\n\n',setExperiment.name);
   fprintf(file_1,'%s\n',exception.message);
   fclose(file_1);
   
   filename_2 = fullfile(setExperiment.datapath, '\log.txt');
   file_2 = fopen(filename_2,'a');
   fprintf(file_2,'The experiment:  "%s" has FAILED:\n\n',setExperiment.name);
   fprintf(file_2,'%s\n',exception.message);
   fclose(file_2);
end