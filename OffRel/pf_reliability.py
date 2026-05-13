import numpy as np
from scipy.stats import norm
from scipy.stats import weibull_min
import math
from scipy.special import gamma, gammaincc
from scipy.optimize import minimize
import py_fatigue as pf
from py_fatigue import SNCurve


class Reliability:

    """
    Reliability analysis of fatigue failure using Monte Carlo Simulation (MCS).

    This class estimates the probability of fatigue failure over the service life
    of a structure using Miner’s rule, a Weibull-distributed long-term stress
    range model, and an S-N curve represented by a py_fatigue SNCurve object.

    The Weibull scale parameter q may be obtained using fatigue_load class following different approaches, such as:
        - design-to-the-limit fatigue assessment,
        - long-term monitoring measurements,
        - DLC simulation-based fatigue load estimation.

    Parameters
    ----------
    config : dict
        Dictionary containing all input parameters required for the reliability
        analysis.

        Required keys
        -------------
        T : int
            Number of years or time steps considered in the analysis.

        ncycles : float
            Number of stress cycles per year or per time step.

        q_cov : float
            Coefficient of variation of the Weibull scale parameter q.

        q_mean : float
            Mean Weibull scale parameter of the long-term stress range distribution.

        h : float
            Weibull shape parameter of the long-term stress range distribution.

        sn : py_fatigue.SNCurve
            S-N curve object defining the fatigue resistance model.
            Both single-slope and two-slope S-N curves are supported.

        samp : int
            Number of Monte Carlo samples.

        delta_mean : float
            Mean value of the lognormal model uncertainty factor applied
            to accumulated fatigue damage.

        delta_cov : float
            Coefficient of variation of the lognormal model uncertainty
            factor applied to accumulated fatigue damage.

        Additional required keys depending on S-N curve type
        ----------------------------------------------------

        For single-slope S-N curves:
            loga_std : float
                Standard deviation of log(a).

        For two-slope S-N curves:
            loga1_std : float
                Standard deviation of log(a1) for the first S-N branch.

            loga2_std : float
                Standard deviation of log(a2) for the second S-N branch.

        Optional keys
        -------------
        time_variant_q : bool, optional
            If True, q is treated as a time-variant stochastic variable.
            Default is False.

        epsilonq_std : float, optional
            Standard deviation of the multiplicative uncertainty factor
            applied to q when time_variant_q is True.

    Attributes
    ----------
    sn : py_fatigue.SNCurve
        S-N curve used in the fatigue damage calculation.

    q_mean : float
        Mean Weibull scale parameter of the long-term stress range distribution.

    damage : ndarray
        Simulated accumulated fatigue damage for all time steps and samples.

    dd : ndarray
        Limit-state variable defined as fatigue resistance minus accumulated
        fatigue damage.

    qq : ndarray
        Simulated Weibull scale parameter q for all time steps and samples.

    Methods
    -------
    run_mcs(seed=None)
        Runs the Monte Carlo Simulation and returns the probability of fatigue
        failure at each time step.

    transition_models(n_dstates=20, n_qstates=30)
        Builds discrete transition probability models for damage and,
        optionally, the Weibull scale parameter q.

    observation_models(obs_params)
        Builds inspection and monitoring observation models for Bayesian
        updating or decision analysis.
    """

    def __init__(self, config = {"time_variant_q": False, "epsilonq_std": None}):
        self.T = config["T"]
        self.ncycles = config["ncycles"]
        self.q_cov = config["q_cov"]
        self.h = config["h"]
        self.sn = config["sn"]          # py_fatigue.SNCurve object
        self.samp = int(config["samp"])

        self.q_mean = config.get("q_mean", None)
        self.time_variant_q = config.get("time_variant_q", False)
        self.epsilonq_std = config.get("epsilonq_std", None)

        # S-N uncertainty inputs
        if self.sn.slope.shape[0] == 1:
            self.loga_std = config.get("loga_std", 0.2)
            self.m = self.sn.slope[0]
            self.loga = self.sn.intercept[0]

        elif self.sn.slope.shape[0] == 2:
            self.loga1_std = config.get("loga1_std", 0.2)
            self.loga2_std = config.get("loga2_std", 0.2)
            self.m1, self.m2 = self.sn.slope
            self.loga1, self.loga2 = self.sn.intercept            
            

        else:
            raise ValueError("SNCurve must have either one or two slopes.")

        # Damage model uncertainty inputs
        self.delta_mean = config.get("delta_mean", 0)
        self.delta_cov = config.get("delta_cov", 0.3)


    
    def run_mcs(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        """
        Run Monte Carlo Simulation to estimate the pf and reliability index.
        """
        if self.sn.slope.shape[0] == 1:
            la_mean = self.loga + 2 * self.loga_std
            la_samples = np.random.normal(la_mean, self.loga_std, self.samp)

            sn_samples = {
                "a": 10**la_samples,
            }

        elif self.sn.slope.shape[0] == 2:
            la1_mean = self.loga1 + 2 * self.loga1_std
            la2_mean = self.loga2 + 2 * self.loga2_std

            cov_matrix = np.array([[1, 1], [1, 1]])
            z_samples = np.random.multivariate_normal([0, 0], cov_matrix, self.samp)
            u_samples = norm.cdf(z_samples, 0, 1)

            la1_samples = norm.ppf(u_samples[:, 0], la1_mean, self.loga1_std)
            la2_samples = norm.ppf(u_samples[:, 1], la2_mean, self.loga2_std)

            sn_samples = {
                "a1": 10**la1_samples,
                "a2": 10**la2_samples,
                "S1": 10**((la1_samples - la2_samples) / (self.m1 - self.m2)),
            }
            

        # Accumulated damage as lognormal distribution
        delta_std = np.sqrt(np.log(self.delta_cov**2 + 1))
        delta_samples = np.random.lognormal(self.delta_mean, delta_std, self.samp)

        self.damage = np.zeros((self.T+1, self.samp))
        self.dd = np.zeros((self.T+1, self.samp))
        self.dd[0,:] = delta_samples

        # q_std = self.q_cov * self.q_mean
        # q0 = np.random.normal(self.q_mean, q_std, self.samp)
        # Remove negative samples
        # while (q0<0).sum() > 0:
        #     q0[q0 < 0] = np.random.normal(self.q_mean, q_std, np.sum(q0 < 0))
        # if self.time_variant_q is True:
            # epsilonq = np.random.normal(1, self.epsilonq_std, nsamples)
        self.q_mean = {'q': [self.q_mean], 't_end': [self.T+1]} # design scale parameter

        q = np.zeros((len(self.q_mean['q']), self.samp))
        for i in range(len(self.q_mean['q'])):
            qi = np.random.normal(self.q_mean['q'][i], 
                                  self.q_mean['q'][i]*self.q_cov, self.samp)
            while (qi<0).sum() > 0:
                qi[qi < 0] = np.random.normal(self.q_mean['q'][i], 
                                              self.q_mean['q'][i]*self.q_cov, np.sum(qi < 0))
            q[i,:] = qi
        t_end = self.q_mean['t_end']
        durations = np.diff([0] + t_end)
        q = np.repeat(q, durations, axis=0)

        self.qq = np.zeros((self.T+1, self.samp))
        self.qq[0,:] = q[0,:]
        for t in range(self.T):
            if self.time_variant_q is True:
                if t+1 not in t_end:
                    epsilonq = np.random.normal(1, self.epsilonq_std, self.samp)
                    q_next = self.qq[t, :]*epsilonq # uncertainty of q increases
                    while (q_next<0).sum() > 0:
                        q_next[q_next<0] = self.qq[t, :][q_next<0]*np.random.normal(1, self.epsilonq_std, (q_next<0).sum())
                    self.qq[t+1,:] = q_next
                else:
                    self.qq[t+1,:] = q[t+1, :]
            else:
                self.qq[t+1,:] = q[t+1, :]   

            #     epsilonq = np.random.normal(1, self.epsilonq_std, self.samp)
            #     qt = q0*epsilonq # uncertainty of q increases
            #     while (qt<0).sum() > 0:
            #         qt[qt<0] = q0[qt<0]*np.random.normal(1, self.epsilonq_std , (qt<0).sum())
            #     q0 = qt
            #     self.qq[t+1,:] = qt
            # else:
            #     self.qq[t+1,:] = q0


        for t in range(self.T):
            q0 = self.qq[t,:]
            print(t+1, np.mean(q0), np.std(q0))

            damage_increment = self._fatigue_damage_increment(q0=q0,sn_samples=sn_samples)

          
            self.damage[t + 1, :] = self.damage[t, :] + damage_increment

            self.dd[t+1,:] = delta_samples - self.damage[t+1,:] # limit state variable
        
        return np.sum(self.dd<=0, axis=1)/self.samp # pf at each time step 
    

    def _fatigue_damage_increment(self, q0, sn_samples):
        """
        Calculate fatigue damage increment for one time step.

        Supports both single-slope and two-slope S-N curves.
        """

        if self.sn.slope.shape[0] == 1:
            a_samples = sn_samples["a"]

            damage_increment = (
                self.ncycles
                / a_samples
                * q0**self.m
                * gamma(1 + self.m / self.h)
            )

        elif self.sn.slope.shape[0] == 2:
            a1_samples = sn_samples["a1"]
            a2_samples = sn_samples["a2"]
            S1_samples = sn_samples["S1"]

            G1 = gamma(1 + self.m1 / self.h) * gammaincc(
                1 + self.m1 / self.h,
                (S1_samples / q0) ** self.h,
            )

            G2 = gamma(1 + self.m2 / self.h) * (
                1 - gammaincc(
                    1 + self.m2 / self.h,
                    (S1_samples / q0) ** self.h,
                )
            )

            damage_increment = self.ncycles * (
                q0**self.m1 / a1_samples * G1
                + q0**self.m2 / a2_samples * G2
            )

        else:
            raise ValueError("SNCurve must have either one or two slopes.")

        return damage_increment
    



    def transition_models(self, n_dstates=20, n_qstates = 30):
        if self.time_variant_q is True:
            self.d_interv = -1e20
            self.d_interv = np.append(self.d_interv, np.linspace(0, 3, n_dstates-1))
            self.d_interv = np.append(self.d_interv, 1e20)

            self.q_interv = np.linspace(0, 43.5, n_qstates)
            self.q_interv[0] = 1e-20
            self.q_interv = np.append(self.q_interv, 1e20)

            det_rates = self.T+1
            nsamples = self.dd.shape[-1]  
            H, _, _ = np.histogram2d(self.dd[0,:], self.qq[0,:], [self.d_interv, self.q_interv])
            self.b0 = (H/nsamples).reshape(-1) # d is the outer loop

            self.T0 = np.zeros((det_rates, n_dstates*n_qstates, n_dstates*n_qstates))
            for i in range(det_rates-1):
                #print(i)
                D = self.dd[i,:] # Samples a at det. rate i
                D_ = self.dd[i+1,:] # Samples a at det. rate i+1
                Q = self.qq[i,:] # Samples q at det. rate i, 
                Q_ = self.qq[i+1,:] # Samples q at det. rate i+1]

                for j in reversed(range(n_dstates)):
                    countd = (D>=self.d_interv[j]) &  (D<self.d_interv[j+1])
                    for k in reversed(range(n_qstates)):
                        countq =(Q>self.q_interv[k]) &  (Q<self.q_interv[k+1])
                        Dnext = D_[countd & countq]
                        Qnext = Q_[countd & countq]
                        if (countd & countq).sum() < 1:
                            self.T0[i,j*n_qstates+k,j*n_qstates+k]=1
                            #print('no sample')
                        else:
                            H, _, _ = np.histogram2d(Dnext, Qnext , [self.d_interv, self.q_interv])
                            #print(len(Dnext)-sum(H.reshape(-1)))
                            self.T0[i,j*n_qstates+k,:] = (H/(countd & countq).sum()).reshape(-1)
            self.T0[-1,] = self.T0[-2,].copy()
            
            return self.d_interv, self.q_interv
            
        else:
            self.d_interv = -1e20
            self.d_interv = np.append(self.d_interv, np.linspace(0, 3, n_dstates-1))
            self.d_interv = np.append(self.d_interv, 1e20)
            
            det_rates = self.T+1
            nsamples = self.dd.shape[-1]   
            H, _ = np.histogram(self.dd[0,:], self.d_interv)
            self.b0 = H/nsamples
            
            self.T0 = np.zeros((det_rates, n_dstates, n_dstates))
            for i in range(det_rates-1):
                D = self.dd[i,:] # Samples a at det. rate i
                D_ = self.dd[i+1,:] # Samples a at det. rate i+1
                for j in reversed(range(n_dstates)):
                    countd = (D>=self.d_interv[j]) &  (D<self.d_interv[j+1])
                    Dnext = D_[countd]
                    if countd.sum() < 1:
                        self.T0[i,j,j]=1
                    else:
                        H, _ = np.histogram(Dnext, self.d_interv) 
                        self.T0[i,j,:] = H/countd.sum()
            self.T0[-1,] = self.T0[-2,]
                
            
            return self.d_interv





    def observation_models(self, obs_params = {"inspect": True, 
                                               "beta0": 7.3704, "beta1": 2.092, "sigma_epsilon": 4.189, "det_thres": 5.4898,
                                               "monitor": True,
                                               "error_std": 0.5}):
        inspect = obs_params["inspect"] if obs_params.get("inspect") is not None\
            else True      
        beta0 = obs_params["beta0"] if obs_params.get("beta0") is not None\
            else 7.3704
        beta1 = obs_params["beta1"] if obs_params.get("beta1") is not None\
            else 2.092
        sigma_epsilon = obs_params["sigma_epsilon"] if obs_params.get("sigma_epsilon") is not None\
            else 4.189
        det_thres = obs_params["det_thres"] if obs_params.get("det_thres") is not None\
            else 5.4898
        monitor = obs_params["monitor"] if obs_params.get("monitor") is not None\
            else False
        error_std = obs_params["error_std"] if obs_params["monitor"] is True\
            else None
        
        n_dstates = len(self.d_interv)-1 
        if self.time_variant_q is True:
            n_qstates = len(self.q_interv)-1
        
        if inspect is False and monitor is False:
            dobs = np.zeros((n_dstates, 2))
            dobs[:,0] = 1
            dobs[:,1] = 0
            if self.time_variant_q is True:             
                dobs = np.repeat(dobs,n_qstates,axis=0) 
        
        if inspect is True: # inspection model
            dobs = np.zeros((n_dstates, 2))
            d_ref = (self.d_interv[0:-1]+self.d_interv[1:])/2
            d_ref[-1] = d_ref[-2]+0.1
            dobs[:,0] = 1-stats.norm.cdf((det_thres-beta0-beta1*np.log(d_ref))/sigma_epsilon)
            dobs[:,1] = 1-dobs[:,0] 
            if self.time_variant_q is True:             
                dobs = np.repeat(dobs,n_qstates,axis=0)

        if monitor is True: # monitoring model
            q_ref = -1e100
            q_ref = np.append(q_ref, self.q_interv[1:])
            q_ref[-1] = 1e100
            q_ref = np.tile(q_ref,(100,1)).T
            qobs_std = np.ones((100,))*error_std           
            qobs = np.zeros((n_qstates, n_qstates))
            for i in range(n_qstates):
                qobs_mean = np.linspace(self.q_interv[i],self.q_interv[i+1],100)
                qobs_cdf = norm.cdf(q_ref, qobs_mean, qobs_std).T
                qobs_cdf[:,-1] = 1 # to make sure the probabilities sum to one
                qobs_pdf = np.diff(qobs_cdf)/100
                qobs[i,:] += np.sum(qobs_pdf, axis=0) 
            if self.time_variant_q is True:
                qobs = np.tile(qobs,(n_dstates,1))
        
        if inspect is True and monitor is True: # joint probability of inspection and monitoring
            ins_monitor = np.concatenate((qobs.T*dobs[:,0],qobs.T*dobs[:,1]),axis=0).T
            monitor = np.zeros(ins_monitor.shape)
            monitor[:,0:n_qstates] = qobs
            ins = np.zeros(ins_monitor.shape)
            ins[:,[0, n_qstates]] = dobs
            no_obs = np.zeros(ins_monitor.shape)
            no_obs[:,0] =1
            self.O = {"ins_monitor": ins_monitor,"monitor": monitor, "ins": ins, "no_obs": no_obs}
        elif inspect is True:
            ins = dobs
            no_ins = np.zeros(ins.shape)
            no_ins[:,0] =1
            self.O = {"ins": ins, "no_ins": no_ins}
        elif monitor is True:            
            monitor = qobs
            no_monitor = np.zeros(monitor.shape)
            no_monitor[:,0] =1
            self.O = {"monitor": monitor, "no_monitor": no_monitor}
        else:
            self.O = {"no_obs": dobs}
                       
        return self.O
        

        
        

    


    