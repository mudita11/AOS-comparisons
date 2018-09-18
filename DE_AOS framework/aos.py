import numpy as np
from scipy.stats import rankdata
import math
from collections import Counter

class AOS(object):
    def __init__(self, popsize, F1, F, u, X, f_min, x_min, best_so_far, best_so_far1, n_ops, window_size= None):
        self.n_ops = n_ops
        self.window_size = window_size
        # The initial list of operators (randomly permuted)
        self.op_init_list = list(np.random.permutation(n_ops))
        self.popsize = int(popsize)
        self.F1 = F1
        self.F = F
        self.u = u
        self.X = X
        self.f_min = f_min
        self.x_min = x_min
        self.best_so_far = best_so_far
        self.best_so_far1 = best_so_far1
        
        self.p = np.zeros(self.n_ops)
        self.Q = np.zeros(self.n_ops)
        self.reward = np.zeros(int(self.n_ops))
        
        self.n = np.zeros(int(self.popsize))       # calculates relatives fitness improvement for each strictly improved offspring
        self.sr_strict=np.zeros(int(self.n_ops))   # counts number of offsprings strictly better than their parents
        self.sr_equals=np.zeros(int(self.n_ops))   # counts number of offsprings having same fitness as their parents
        self.sum_n = np.zeros(int(self.n_ops))
        
        self.opu = [4 for i in range(int(popsize))]; self.opu=np.array(self.opu)
        self.choice = np.zeros(int(self.n_ops))
        self.window_delta_f = np.zeros(int(self.window_size))
        self.operator = np.zeros(int(self.window_size), dtype='int')
        self.rank = np.zeros(int(self.window_size))
        self.area = np.zeros(int(self.n_ops))
    
    
    def AOSUpdate(self):
        self.n = np.zeros(int(self.popsize))
        self.sr_strict = np.zeros(int(self.n_ops))
        self.sr_equals = np.zeros(int(self.n_ops))
        self.sum_n = np.zeros(int(self.n_ops))
    
        for i in range(self.popsize):
            if self.F1[i] <= self.F[i]:
                if self.F1[i] < self.F[i]:
                    self.sr_strict[self.opu[i]] += 1
                    self.n[i] = (self.best_so_far / self.F1[i]) * math.fabs(self.F[i] - self.F1[i])
                    self.sum_n[self.opu[i]] += self.n[i]
                else:
                    self.sr_equals[self.opu[i]] += 1
             
                delta_f = self.F1[i]
                if np.any(self.window_delta_f == np.inf):
                    for value in range(self.window_size-1,-1,-1):
                        if self.window_delta_f[value] == np.inf:
                            self.window_delta_f[value] = delta_f
                            self.operator[value] = self.opu[i]
                            break
                else:
                    for nn in range(self.window_size-1,-1,-1):
                        if self.operator[nn] == self.opu[i]:
                            for nn1 in range(nn,0,-1):
                                self.window_delta_f[nn1] = self.window_delta_f[nn1-1]
                                self.operator[nn1] = self.operator[nn1-1];
                            self.window_delta_f[0] = delta_f;
                            self.operator[0] = self.opu[i];
                            break
                        elif nn == 0 and self.operator[nn] != self.opu[i]:
                            if delta_f < np.max(self.window_delta_f):
                                zy = np.argmax(self.window_delta_f)
                                self.window_delta_f[zy] = delta_f;
                                self.operator[zy] = self.opu[i];
                self.X[i][:] = self.u[i][:]
                self.F[i] = self.F1[i]
        
        self.Reward()
        self.Quality()
        self.Probability()

def normalize_matrix(x):
    return x / np.sum(x, axis=1)[:, None]

def calc_delta_r (decay, W, ndcg):
    if decay == 0:
        return np.ones(window_size)
    r = np.array(range(W), dtype='float')
    if ndcg:
        r += 1
        delta_r = ((2 ** (W - r)) - 1) / np.log(1 + r)
    else:
        delta_r = (decay ** r) * (W - r)
    return delta_r

def AUC(operators, rank, op, decay = 2, ndcg = True):
    assert len(operators) == len(rank)
    W = len(operators)
    delta_r_vector = calc_delta_r(decay, W, ndcg)
    x, y, area = 0, 0, 0
    r = 0
    while r < W:
        delta_r = delta_r_vector[r]
        # number of rewards equal to reward ranked r given by op
        tiesY = np.count_nonzero(rank[operators == op] == rank[r])
        # number of rewards equal to reward ranked r given by others
        tiesX = np.count_nonzero(rank[operators != op] == rank[r])
        assert tiesY >= 0
        assert tiesX >= 0
        if (tiesX + tiesY) > 0 :
            delta_r = np.sum(delta_r_vector[r : r + tiesX + tiesY]) / (tiesX + tiesY)
            x += tiesX * delta_r
            area += (y * tiesX * delta_r) + (0.5 * delta_r * delta_r * tiesX * tiesY)
            y += tiesY * delta_r
            r += tiesX + tiesY
        elif operators[r] == op:
            y += delta_r
            r += 1
        else:
            x += delta_r
            area += y * delta_r
            r += 1
    return area

def upper_confidence_bound (n_ops, window_op_sorted, C, quality):
    N = np.zeros(n_ops)
    # the number of times each operator appears in the sliding window
    op, count = np.unique(window_op_sorted, return_counts=True)
    N[op] = count
    ucb = quality + C * np.sqrt( 2 * np.log(np.sum(N))/N)
    # Infinite quality means division by zero, give zero quality.
    ucb[np.isinf(ucb)] = 0
    return ucb

class F_AUC(AOS):

    def __init__(self, popsize, F1, F, u, X, f_min, x_min, best_so_far, best_so_far1, n_ops, alpha, p_min, scaling_factor, window_size):
        super(F_AUC,self).__init__(popsize, F1, F, u, X, f_min, x_min, best_so_far, best_so_far1, n_ops, window_size)
        self.Q.fill(np.inf)
        self.opu.fill(np.inf)
        self.window_delta_f.fill(np.inf)
        self.operator.fill(-1)
        self.scaling_factor = scaling_factor

    def Selection(self):
        if self.op_init_list:
            SI = self.op_init_list.pop()
        else:
            SI = np.argmax(self.choice)
        return SI
    
    def Reward(self):
        self.rank = rankdata(self.window_delta_f.round(1), method = 'min')
        # Sort operators and rank in increasing rank
        order = self.rank.argsort()
        self.window_op_sorted = self.operator[order]
        self.rank = self.rank[order]
        self.rank = self.rank[self.window_op_sorted >= 0]
        self.window_op_sorted = self.window_op_sorted[self.window_op_sorted >= 0]
        
        for op in range(self.n_ops):
            self.reward[op] = AUC(self.window_op_sorted, self.rank, op)

    def Quality(self):
        self.Q[:] = self.reward
        cat = np.sum(self.Q)
        if cat != 0:
            self.Q = self.Q / cat # credit = normalized area

    def Probability(self):
        self.choice = upper_confidence_bound (self.n_ops, self.window_op_sorted, self.scaling_factor, self.Q)

class PM_AdapSS(AOS):
    def __init__(self, popsize, F1, F, u, X, f_min, x_min, best_so_far, best_so_far1, n_ops, alpha, p_min, scaling_factor, window_size):
        super(PM_AdapSS,self).__init__(popsize, F1, F, u, X, f_min, x_min, best_so_far, best_so_far1, n_ops, window_size)
        self.p[:] = 1.0 / len(self.p)
        self.Q[:] = 1.0
        self.alpha = alpha
        self.p_min = p_min

    def Selection(self):
        if self.op_init_list:
            SI = self.op_init_list.pop()
        else:
            assert np.allclose(np.sum(self.p), 1.0)
            assert np.all(self.p >= 0.0)
            SI = np.random.choice(len(self.p), p = self.p)
        return SI

    def Reward(self):
        self.reward[:] = 0
        for i in range(self.n_ops):
            if self.sr_strict[i] > 0:
                self.reward[i] = self.sum_n[i] / self.sr_strict[i]
    
    def Quality(self):
        self.Q += self.alpha * (self.reward - self.Q)
        self.Q = self.Q - np.max(self.Q)
        self.Q = np.exp(self.Q)
        self.Q = self.Q / np.sum(self.Q)

    def Probability(self):
        assert np.sum(self.Q) > 0
        assert len(self.Q) == len(self.p)
        if np.sum(self.Q) != 0:
            self.p = self.p_min + (1 - len(self.Q) * self.p_min) * (self.Q / np.sum(self.Q))
        self.p = self.p / np.sum(self.p)

class Rec_PM(PM_AdapSS):
    def __init__(self, popsize, F1, F, u, X, f_min, x_min, best_so_far, best_so_far1, n_ops, alpha, p_min, scaling_factor, window_size):
        super(Rec_PM,self).__init__(popsize, F1, F, u, X, f_min, x_min, best_so_far, best_so_far1, n_ops, alpha, p_min, scaling_factor, window_size)
        self.tran_matrix = np.random.rand(n_ops, n_ops)
        self.tran_matrix = normalize_matrix(self.tran_matrix)

    def Selection(self):
        SI = PM_AdapSS.Selection(self)
        return SI
    
    def Reward(self):
        coun = np.zeros(int(self.n_ops))
        c = Counter(self.opu)
        coun=[c[v] for v in range(self.n_ops)]
        self.reward *= 0.5
        for i in range(self.n_ops):
            if coun[i] != 0:
                self.reward[i] += np.array(self.sr_strict[i] + self.sr_equals[i]) / np.array(coun[i]);

    def Quality(self):
        self.Q = np.matmul(np.linalg.pinv(np.array((1 - self.alpha * self.tran_matrix))), np.array(self.reward))
        self.Q = self.Q - np.max(self.Q)
        self.Q = np.exp(self.Q)
        self.Q = self.Q / np.sum(self.Q)

    def Probability(self):
        PM_AdapSS.Probability(self)
        for i in range(self.n_ops):
            for j in range(self.n_ops):
                self.tran_matrix[i][j] = self.p[j] + self.p[i]
        self.tran_matrix = normalize_matrix(self.tran_matrix)

