# RecPM-AOS-comparison

Adaptive Operator Selection within Differential Evolution

Algorithms are tested on BBOB test suite on dimension 20.

Please cite these results/algorithms as follows:


Notations: 

DE-RecPM-AOS or RecPM: Recursive Probability Matching Adaptive Operator Selector within Differential Evolution

DE-F-AUC(-R) or FAUC(-R): Bandit with Area Under the curve within Differential Evolution without replication(best replication from thesis: Adaptive Operator Selection for Optimisation)

PM-AdapSS-DE(-R) or AdapSS(-R): Probability Matching Adaptive Operator Selector within Differential Evolution without replication(best replication from thesis: Fialho, Álvaro. Adaptive operator selection for optimization. Diss. Université Paris Sud-Paris XI, 2010.)

Any above mentioned algorithm can have three version, notations described below (Please note in following "Algo" can be replaced by any algorithm above): 

Algo1: Parameters of AOS and DE are tuned using IRACE.

Algo2: Parameters of only AOS method are tuned; Parameter values for DE algorithm: CR (Crossover rate) = 1.0, F (Mutation rate) = 0.5, NP (Population size) = 200

Algo3: Parameter values for DE algorithm: CR (Crossover rate) = 1.0, F (Mutation rate) = 0.5, NP (Population size) = 200; Parameter values for AOS method: alpha (Adaptation rate) / gamma (Discount factor) = 0.6, p_min (Minimum probability attainable by any operator) = 0.0, W (Window size) = 50, C (Scaling factor) = 0.5

CMAES: Covariance Matrix Adaptation Evolution Strategy

JADE: Adaptive Differential Evolution With Optional External Archive


Folder: Replicated
Shows comparisons of various algorithms: CMAES, JADE, DE-RecPM-AOS and replicated versions of DE-F-AUC and PM-AdapSS-DE denoted as DE-F-AUC-R and PM-AdapSS-DE-R respectively. Data to generate results (graphs and tables) for CMAES and JADE algorithms is taken from coco website (http://coco.gforge.inria.fr/doku.php?id=algorithms-bbob). 

Three algorithms with AOS method (DE-RecPM-AOS, DE-F-AUC and PM-AdapSS-DE) are tuned. 


Folder: Existing

Shows comparisons of various algorithms: CMAES, JADE, DE-RecPM-AOS, DE-F-AUC and PM-AdapSS-DE. Data for CMAES, JADE and latter two algorithms is taken from coco website (http://coco.gforge.inria.fr/doku.php?id=algorithms-bbob).   

Three algorithms with AOS method (DE-RecPM-AOS, DE-F-AUC and PM-AdapSS-DE) are tuned. 

