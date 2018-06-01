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
        
        self.n = np.zeros(int(self.popsize))       #calculates relatives fitness improvement for each strictly improved offspring
        self.sr_strict=np.zeros(int(self.n_ops))   #counts number of offsprings strictly better than their parents
        self.sr_equals=np.zeros(int(self.n_ops))   #counts number of offsprings having same fitness as their parents
        self.sum_n = np.zeros(int(self.n_ops))
        
        self.opu = [4 for i in range(int(popsize))]; self.opu=np.array(self.opu)
        self.choice = np.zeros(int(self.n_ops))
        self.window_delta_f = np.zeros(int(self.window_size))
        self.operator = np.zeros(int(self.window_size), dtype='int')
        self.rank = np.zeros(int(self.window_size))
        self.area = np.zeros(int(self.n_ops))
        
    # FIXME: This function is mixing up the update by DE and the update of
    # AOS. Please put back the DE part in DE and just tell the AOS method what
    # DE has changed. Take a look at the SciPy implementation of DE:
    # https://github.com/scipy/scipy/blob/master/scipy/optimize/_differentialevolution.py
    
    def AOSUpdate(self):
        #print("AOS update")
        self.n = np.zeros(int(self.popsize))
        self.sr_strict = np.zeros(int(self.n_ops))
        self.sr_equals = np.zeros(int(self.n_ops))
        self.sum_n = np.zeros(int(self.n_ops))
        # F1 is children
        # F is target parent
        for i in range(self.popsize):
            if self.F1[i] <= self.F[i]:
                if self.F1[i] < self.F[i]:
                    self.sr_strict[self.opu[i]] += 1
                    self.n[i] = (self.best_so_far / self.F1[i]) * math.fabs(self.F[i] - self.F1[i])
                    self.sum_n[self.opu[i]] += self.n[i]
                else:
                    self.sr_equals[self.opu[i]] += 1
                # PMAdapSS but originally proposed in Y.-S. Ong and A.J. Keane.
                # Meta-Lamarckian learning in memetic algorithms. IEEE Trans.
                # on Evol. Comput.,8(2):99-110, Apr 2004.
             
                delta_f = self.F1[i]#math.fabs(F1[i]-F[i])
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
                            #print("opu[i],Inside",opu[i])
                            break
                        elif nn == 0 and self.operator[nn] != self.opu[i]:
                            if delta_f < np.max(self.window_delta_f):
                                zy = np.argmax(self.window_delta_f) # argmin gives position of value in list that is minimum
                                self.window_delta_f[zy] = delta_f;
                                self.operator[zy] = self.opu[i];
                self.X[i][:] = self.u[i][:]
                self.F[i] = self.F1[i]
        
        self.Reward()
        self.Quality()
        self.Probability()

def normalize_matrix(x):
    return x / np.sum(x, axis=1)[:, None]
    

def test_AUC():
    # From Fialho's PhD (Table 5.1)
    operators = np.array([1,1,3,1,1,2,3,4,1,1,2,3,4,4,3]) - 1
    rank = np.array([1,2,3,3,5,6,7,7,7,10,11,12,13,14,15])
    ops = np.unique(operators)
    areas = np.zeros(len(ops))
    
    for op in ops:
        areas[op] = AUC(operators, rank, op, decay = 0, ndcg = False)
    areas /= np.sum(areas)
    # Fig 5.3a
    assert np.allclose(areas, np.array([ 0.56875,  0.15   ,  0.19375,  0.0875 ]))

    for op in ops:
        areas[op] = AUC(operators, rank, op, decay = 1, ndcg = False)
    areas /= np.sum(areas)
    # Fig 5.3b
    assert np.allclose(areas, np.array([ 0.64188168,  0.10074439,  0.21327027,  0.04410365]))

    for op in ops:
        areas[op] = AUC(operators, rank, op, decay = 0.5, ndcg = False)
    areas /= np.sum(areas)
    areas = np.round(areas,4)
    # Fig 5.3c (Fialho's Thesis says this is for decay = 0.4)
    assert np.allclose(areas, np.array([ 0.9216,  0.0011,  0.0772,  0.0001]))

def calc_delta_r (decay, W, ndcg):
    if decay == 0:
        return np.ones(window_size)
    # This formula is also wrong in Fialho's PhD.
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
            # This is wrong in Fialho's PhD (Algorithm 5.2)
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
        #print("\nx = %g  y = %g  tiesX = %g  tiesY = %g  area = %g  delta_r = %g" % (x, y, tiesX, tiesY, area, delta_r))
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
    # for i in range(self.n_ops):
    #         # N[i]=0;
    #         for it in range(self.W):
    #             if self.window_op_sorted[it]==i:
    #                 self.N[i]=self.N[i]+1;  #the number of times each operator appears in the sliding window
    #     #for ss in range(n_ops):
    #         #N_sum=N_sum+N[ss]
    #     # FIXME: How can N_z be zero???
    #     for z in range(self.n_ops):
    #         if self.N[z]!=0:
    #             self.choice[z]=self.Q[z]+self.C * math.sqrt((2*math.log(np.sum(self.N)))/(self.N[z]))


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
        # window_delta_f1 = list(self.window_delta_f)
        # a,b = zip(*sorted(zip(self.window_delta_f1, self.operator1)))#,reverse=True))
        # window_delta_f1=list(a)
        # self.operator1=list(b); #print(window_delta_f1,operator1)
        #Application of one n_ops might affect the others
        #CreditAssignment.GetReward(i)
        # MANUEL: This is very suspicious to round to 1???
        # rank according to window_delta_f (rounded to 1 decimal place),
        # with ties being assigned the minimum rank
        self.rank = rankdata(self.window_delta_f.round(1), method = 'min')
        # Sort operators and rank in increasing rank
        order = self.rank.argsort()
        self.window_op_sorted = self.operator[order]
        self.rank = self.rank[order]
        # Remove everything that has no operator
        self.rank = self.rank[self.window_op_sorted >= 0]
        self.window_op_sorted = self.window_op_sorted[self.window_op_sorted >= 0]
        
        #self.rank[0] = 1
        # for i in range(1,W):
        #     if round(self.window_delta_f1[i],1)==round(self.window_delta_f1[i-1],1):
        #         self.rank[i]=self.rank[i-1]
        #     else:
        #         self.rank[i]=i+1
        for op in range(self.n_ops):
            self.reward[op] = AUC(self.window_op_sorted, self.rank, op)
        #print("rank,Q,operator1,window_delta_f1,rank,SI",rank,Q,operator1,window_delta_f1,rank,SI)

    # def mudita_AUC(self,operator1,rank,ii):
    #     x1=0;y1=0;area=0;delta_r1=0;rr=0;rr1=0;
    #     for r in range(1,self.W+1): # r:rank-position
    #         tiesX=0;tiesY=0;
    #         #delta_r1=math.pow(DD,r)*(W-r) # calculate weight of rank position in the area
    #         delta_r1=(math.pow(2,self.W-r)-1)/(math.log(1+r)) #adapted_NDCG
    #         i=0;
    #         while i<self.W:
    #             if self.window_op_sorted[i]==ii: #number of rewards equal to reward ranked r given by op-ii
    #                 rr=self.rank[i]
    #                 flag=False;
    #                 for dd in range(i+1,self.W):
    #                     if self.rank[dd]==rr:
    #                         tiesY=tiesY+1
    #                         xyz=dd;
    #                         i=xyz;
    #                         flag=True
    #                 if not flag:
    #                     i=i+1
    #             else:
    #                 i=i+1;

    #         i=0;
    #         while i<self.W:
    #             if self.window_op_sorted[i]!=ii:  # rewards equal to reward ranked r given by others
    #                 rr1=self.rank[r-1];
    #                 flag=False;
    #                 for dd1 in range(i+1,self.W):
    #                     if self.rank[dd1]==rr1:
    #                         tiesX=tiesX+1;
    #                         xyz1=dd1
    #                         i=xyz1;
    #                         flag=True;
    #                 if not flag:
    #                     i=i+1
    #             else:
    #                 i=i+1;

    #         #print("tiesX,tiesY :",tiesX,tiesY)
    #         if tiesX+tiesY>0:
    #             for s in range(r+1,r+tiesX+tiesY):
    #                 #delta_r1=delta_r1+(math.pow(DD,s)*(W-s))/(tiesX+tiesY) #sum weights of tied ranks, divided by number of ties
    #                 delta_r1=delta_r1+((math.pow(2,self.W-r)-1)/(math.log(1+r)))/(tiesX+tiesY)
    #             #x1=x1+(tiesX*delta_r)
    #             area=area+(y1*tiesX*delta_r1);  #print("area",area);#sum the rectangle below
    #             y1=y1+(tiesY*delta_r1);
    #             area=area+(0.5*math.pow(delta_r1,2)*tiesX*tiesY); #print("area1",area); #sum the triangle below slanted line
    #             r=r+tiesX+tiesY-1;
    #         elif self.window_op_sorted[r-1]==ii: #if op generated r, vertical segment
    #             y1=y1+delta_r1
    #         else: #if another operator generated r, horizontal segment
    #             #x1=x1+delta_r
    #             area=area+(y1*delta_r1)
    #     return area

    def Quality(self):
        self.Q[:] = self.reward
        cat = np.sum(self.Q)
        if cat != 0:
            #print("n_ops,Q[ii],cat,Q[ii]/cat",n_ops,Q[ii],cat,Q[ii]/cat);
            self.Q = self.Q / cat # credit = normalized area

    def Probability(self):
        self.choice = upper_confidence_bound (self.n_ops, self.window_op_sorted, self.scaling_factor, self.Q)
        # self.N = np.zeros(self.n_ops)
        # for i in range(self.n_ops):
        #     # N[i]=0;
        #     for it in range(self.W):
        #         if self.window_op_sorted[it]==i:
        #             self.N[i]=self.N[i]+1;  #the number of times each operator appears in the sliding window
        # #for ss in range(n_ops):
        #     #N_sum=N_sum+N[ss]
        # for z in range(self.n_ops):
        #     if self.N[z]!=0:
        #         self.choice[z]=self.Q[z]+self.C * math.sqrt((2*math.log(np.sum(self.N)))/(self.N[z]))

class PM_AdapSS(AOS):
    def __init__(self, popsize, F1, F, u, X, f_min, x_min, best_so_far, best_so_far1, n_ops, alpha, p_min, scaling_factor, window_size):
        super(PM_AdapSS,self).__init__(popsize, F1, F, u, X, f_min, x_min, best_so_far, best_so_far1, n_ops, window_size)
        self.p[:] = 1.0 / len(self.p)
        self.Q[:] = 1.0
        self.alpha = alpha
        self.p_min = p_min
        #print("PM_AdapSS: alpha = %g  p_min = %g\n" % (self.alpha, self.p_min))
        #print("Finished PM-init")

    def Selection(self):
        #print("PM selection")
        if self.op_init_list:
            SI = self.op_init_list.pop()
        else:
            #print(self.p)
            assert np.allclose(np.sum(self.p), 1.0)
            assert np.all(self.p >= 0.0)
            #SI = np.argmax(np.random.multinomial(1, self.p, size=1))
            SI = np.random.choice(len(self.p), p = self.p)
            #print(SI)
        return SI

    def Reward(self):
        #print("PM reward")
        # Average Absolute Reward
        self.reward[:] = 0
        for i in range(self.n_ops):
            if self.sr_strict[i] > 0:
                self.reward[i] = self.sum_n[i] / self.sr_strict[i]
            
        # for i in range(self.n_ops):
        #     if i==SI and self.n_improvements!=0:
        #         self.reward[i]=np.sum(self.n)/self.n_improvements;
        #     elif i==SI and self.n_improvements==0:
        #         self.reward[i]=0
        #     else:
        #         self.reward[i]=0
    
    def Quality(self):
        # FIXME: update only probability of selected operator.
        # MUDITA: In case of parent wise selection, the reward of each operator is seleted, thus Q od each operator is estimated
        self.Q += self.alpha * (self.reward - self.Q)
        #print("Q: ",self.Q)
        self.Q = self.Q - np.max(self.Q)
        #print("Q1: ",self.Q)
        self.Q = np.exp(self.Q)
        #print("Q2: ",self.Q)
        self.Q = self.Q / np.sum(self.Q)
        #print("Q3: ",self.Q)
        #print("Finished calculating Quality")


    def Probability(self):
        #print("PM probability")
        assert np.sum(self.Q) > 0
        assert len(self.Q) == len(self.p)
        if np.sum(self.Q) != 0:
            self.p = self.p_min + (1 - len(self.Q) * self.p_min) * (self.Q / np.sum(self.Q))
        # Normalize them
        self.p = self.p / np.sum(self.p)
        #print("p in pm: ",self.p)


class Rec_PM(PM_AdapSS):
    def __init__(self, popsize, F1, F, u, X, f_min, x_min, best_so_far, best_so_far1, n_ops, alpha, p_min, scaling_factor, window_size):
        #print("Rec-PM init")
        super(Rec_PM,self).__init__(popsize, F1, F, u, X, f_min, x_min, best_so_far, best_so_far1, n_ops, alpha, p_min, scaling_factor, window_size)
        #self.p[:] = 1.0 / len(self.p)
        #self.Q[:] = 1.0
        #print("PM_AdapSS: alpha = %g  p_min = %g\n" % (self.alpha, self.p_min))
        self.tran_matrix = np.random.rand(n_ops, n_ops)
        #self.tran_matrix=[[1 for j in range(int(self.n_ops))] for i in range(int(self.n_ops))]
        self.tran_matrix = normalize_matrix(self.tran_matrix)
        #print("Finished rec_PM-init")

    def Selection(self):
        #print("Rec-PM selection")
        SI = PM_AdapSS.Selection(self)
        return SI
    
    def Reward(self):
        #print("Rec-PM reward")
        #print("opu in reward",self.opu)
        coun = np.zeros(int(self.n_ops));#print("coun in recpm reward",coun)
        c = Counter(self.opu)
        #print("c",c)
        coun=[c[v] for v in range(self.n_ops)]
        self.reward *= 0.5
        for i in range(self.n_ops):
            if coun[i] != 0:
                self.reward[i] += np.array(self.sr_strict[i] + self.sr_equals[i]) / np.array(coun[i]);
        #print("End of reward calculation: ",self.reward)

    def Quality(self):
        #print("Rec-PM Quality")
        self.Q = np.matmul(np.linalg.pinv(np.array((1 - self.alpha * self.tran_matrix))), np.array(self.reward))
        #print("Q: ",self.Q)
        self.Q = self.Q - np.max(self.Q)
        #print("Q1: ",self.Q)
        self.Q = np.exp(self.Q)
        #print("Q2: ",self.Q)
        self.Q = self.Q / np.sum(self.Q)
        #print("Q3: ",self.Q)
        #print("Finished calculating Quality")

    def Probability(self):
        #print("Rec-PM Probability")
        # last_p = np.copy(self.p)
        PM_AdapSS.Probability(self)
        for i in range(self.n_ops):
            for j in range(self.n_ops):
                self.tran_matrix[i][j] = self.p[j] + self.p[i]
        self.tran_matrix = normalize_matrix(self.tran_matrix)
        #print("Finished calculating Probability")
        #print("probability: ",self.p)


