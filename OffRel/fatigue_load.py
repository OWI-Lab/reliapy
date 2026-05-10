# -*- coding: utf-8 -*-
"""Fit a Weibull distribution to fatigue cycle data from time series, 
rainflow histograms, or CycleCount objects.

This module is part of the OffRel package, which provides utilities 
for estimating fatigue load distributions in offshore support structures.

Classes
-------
LongtermLoad
    Estimate long-term Weibull scale parameter from DLC simulations or design-to-the-limit.
WeibullFit
    Static methods to fit Weibull distributions from various data formats.

Dependencies
------------
- py_fatigue for cycle counting
- scipy for optimization and statistical functions

Examples
--------
A minimal example using a toy stress time series:

>>> import numpy as np
>>> import pandas as pd
>>> from OffRel.fatigue_load import WeibullFit
>>> import py_fatigue as pf

>>> # Simple stress signal (deterministic)
>>> time = np.arange(6)
>>> stress = np.array([0, 5, -5, 10, -10, 0])
>>> df = pd.DataFrame({"time": time, "stress": stress})

>>> # From time series
>>> round(WeibullFit.from_timeseries(df, k=0.8, scale_init=2), 3)
np.float64(11.167)

>>> # From cycle count
>>> CC = pf.CycleCount.from_timeseries(time=time, data=stress)
>>> round(WeibullFit.from_cyclecount(CC, k=0.8, scale_init=2), 3)
np.float64(11.167)
"""

import numpy as np
import pandas as pd
from scipy.stats import weibull_min
from scipy.special import gamma, gammaincc
from scipy.optimize import minimize, brentq
import py_fatigue as pf
from py_fatigue import SNCurve
from py_fatigue.cycle_count.cycle_count import pbar_sum
import matplotlib.pyplot as plt

class LongtermLoad:
    """
    Class to estimate long-term load distribution parameters (Weibull scale q) from DLC simulations or design to the limit.
    """

    def __init__(self, sn: SNCurve, t_years: int, mean_period: float, fdf: float):
        """Initialize LongtermLoad with an S-N curve, design lifetime, mean period, and fatigue design factor.

        Parameters
        ----------
        sn : SNCurve
            S-N curve object (from py_fatigue).
        t_years : int
            Design lifetime in years.
        mean_period : float
            Mean period of the stress cycles in seconds.
        fdf : float
            Fatigue design factor.  
        """
        self.sn = sn
        self.t_years = t_years
        self.mean_period = mean_period
        self.fdf = fdf

    def fatigue_damage_weibull(self, n_cycles: float, q: float, h: float = 0.8) -> float:
        """
        Compute fatigue damage for a Weibull-distributed stress range 
        Parameters
        ----------
        n_cycles : float
            Number of stress cycles to calculate damage as Number of Years * Cycles per Year; cycles per year = 3600*24*365/mean period.    
        q : float
            Weibull scale parameter.
        h : float
            Weibull shape parameter (default 0.8).
        Returns
        -------
        damage : float or ndarray
            Fatigue damage (Miner's sum).
        """

        # --- Single slope case ---
        if self.sn.slope.shape[0] == 1:
            m = self.sn.slope[0]
            log_a = self.sn.intercept[0]
            a = 10**log_a
            damage = n_cycles / a * (q**m) * gamma(1 + m /h)
            return damage
        
        # --- Two slope case ---
        elif self.sn.slope.shape[0] == 2:
            m1, m2 = self.sn.slope
            log_a1, log_a2 = self.sn.intercept
            a1, a2 = 10**log_a1, 10**log_a2
            s_knee = self.sn.get_knee_stress()[0]

            x = (s_knee / q) ** h

            # Upper incomplete gamma
            g1 = gamma(1 + m1 / h) * gammaincc(1 + m1 / h, x)
            # Lower incomplete gamma
            g2 = gamma(1 + m2 / h) * (1 - gammaincc(1 + m2 / h, x))

            damage = n_cycles * ((q**m1) / a1 * g1 + (q**m2) / a2 * g2)
            return damage
        else:
            raise ValueError("SNCurve must have either one or two slopes.")
    
    def from_design_to_the_limit(self, h: float = 0.8) -> float:
        """Estimate Weibull scale parameter q from design to the limit approach.
        Parameters
        ----------
        h : float, optional
            Weibull shape parameter (default 0.8).

        Returns
        -------
        q_estimated : float
            Estimated Weibull scale parameter for the long-term load distribution.
        """
        
        # Calculate number of cycles in design lifetime
        n_cycles = self.t_years * 3600 * 24 * 365 / self.mean_period

        # Damage limit for design to the limit approach (D_cr = 1 / fdf)
        D_cr = 1 / self.fdf

        def residual(q):
            return self.fatigue_damage_weibull(n_cycles=n_cycles, q=q, h=h) - D_cr

        q_estimated = brentq(residual, a=1e-3, b=1e3)
        return q_estimated
        
    def from_dlc_simulations(self, dlc_data):
        """Estimate Weibull scale parameter q from DLC simulation data.
        Parameters
        ----------
        dlc_data : list of dict
            List of DLC simulation results, where each dict contains 'time' and 'stress' arrays.
        Returns
        -------
        q_estimated : float
            Estimated Weibull scale parameter for the long-term load distribution.
        """
        pass  # Placeholder for future implementation using DLC simulation data
    

class WeibullFit:
    """
    Static methods to fit Weibull distributions from various data formats.
    """
    @staticmethod
    def from_cyclecount(
        CC,
        bin_edges_stress: np.ndarray = None,
        k: float = None,
        scale_init: float = 5,
        SCF: float = None,
        plot: str = None 
    ) -> float:
        """
        Fit Weibull distribution from CycleCount object(s).

        Parameters
        ----------
        CC : pf.CycleCount or list of pf.CycleCount
            Cycle count object(s) from py_fatigue.
        bin_edges_stress : np.ndarray, optional
            Bin edges for stress ranges. Defaults to np.append(np.arange(0, 30, 1), np.inf).
        k : float, optional
            Weibull shape parameter. Defaults to 0.8.
        scale_init : float, optional
            Initial guess for Weibull scale parameter. Defaults to 5.
        SCF : float, optional
            Stress concentration factor to scale stresses. Defaults to 1.
        plot : str, optional
            If 'observed', 'fitted', or 'both', generates a plot of the observed and fitted PDFs.

        Returns
        -------
        fitted_scale : float
            Estimated Weibull scale parameter.
        
        """
        if bin_edges_stress is None:
            bin_edges_stress = np.append(np.arange(0, 30, 1), np.inf)
        if k is None:
            k = 0.8
        if SCF is None:
            SCF = 1.0

        # Aggregate cycle counts
        if isinstance(CC, pf.CycleCount):
            a = CC * SCF
        else:
            a = pbar_sum(CC) * SCF

        # Build DataFrame of stress ranges and cycle counts
        df = pd.DataFrame(
            zip(a.stress_range, a.count_cycle), columns=["stress_range", "count_cycle"]
        )

        # Normalize to probability density
        df["pdf"] = df["count_cycle"] / df["count_cycle"].sum()

        # Bin stress ranges
        df["stress_bin"] = pd.cut(
            df["stress_range"],
            bins=bin_edges_stress,
            right=False,
            include_lowest=True,
        )

        # Aggregate PDF into bins
        observed_pdf = (
            df.pivot_table(values="pdf", index="stress_bin", aggfunc="sum", observed=False)
            .fillna(0)["pdf"]
            .values
        )

        # Objective: minimize squared error between model and observed PDFs
        def objective(scale):
            cdf = weibull_min.cdf(bin_edges_stress, c=k, scale=scale[0])
            model_pdf = np.diff(cdf)
            return np.sum((model_pdf - observed_pdf) ** 2)

        result = minimize(objective, x0=[scale_init], bounds=[(1e-5, 500)])


        # --- Plotting logic ---
        if plot is not None:
            fitted_scale = result.x[0]
            plot_edges = bin_edges_stress[np.isfinite(bin_edges_stress)]
            n_plot = len(plot_edges) - 1
            bin_centers = (plot_edges[:-1] + plot_edges[1:]) / 2
            bin_widths = np.diff(plot_edges)

            if plot in ['fitted', 'both']:
                cdf_fitted = weibull_min.cdf(plot_edges, c=k, scale=fitted_scale)
                fitted_pdf = np.diff(cdf_fitted)

            plt.figure(figsize=(16 / 2.54, 7 / 2.54))
            if plot in ['observed', 'both']:
                plt.bar(bin_centers, observed_pdf[:n_plot], width=bin_widths,
                        alpha=0.5, label='Observed PDF', color='gray', edgecolor='k')
            if plot in ['fitted', 'both']:
                plt.plot(bin_centers, fitted_pdf, 'r-o', lw=2, label='Fitted PDF')
            plt.xlabel('Stress range')
            plt.ylabel('Probability density')
            plt.title('Weibull Fit of Stress Ranges')
            plt.legend()
            plt.grid(True)
            plt.show()

        return result.x[0]

    @staticmethod
    def from_rainflow(
        rf_input,
        bin_edges_stress: np.ndarray = None,
        k: float = None,
        scale_init: float = 5,
        SCF: float = None,
        plot: str = None 
    ) -> float:
        """
        Fit Weibull distribution from rainflow output(s) usually processed timeseries data stored in database.

        Parameters
        ----------
        rf_input : dict, list of dict, or pandas.Series
            Rainflow histogram(s). Can be:
              - A single dict
              - A list of dicts
              - A pandas Series with datetime index and dict values
        bin_edges_stress : np.ndarray, optional
            Bin edges for stress ranges.
        k : float, optional
            Weibull shape parameter. Defaults to 0.8.
        scale_init : float, optional
            Initial guess for Weibull scale parameter. Defaults to 5.
        SCF : float, optional
            Stress concentration factor to scale stresses. Defaults to 1.
        plot : str, optional
            If 'observed', 'fitted', or 'both', generates a plot of the observed and fitted PDFs.

        Returns
        -------
        fitted_scale : float
            Estimated Weibull scale parameter.
        """
        if isinstance(rf_input, dict):
            CC = pf.CycleCount.from_rainflow(rf_input)
        elif isinstance(rf_input, list):
            CC = [pf.CycleCount.from_rainflow(rf) for rf in rf_input]
        elif isinstance(rf_input, pd.Series):
            CC = [pf.CycleCount.from_rainflow(rf) for rf in rf_input.tolist()]
        else:
            raise ValueError("Unsupported input type for from_rainflow.")

        return WeibullFit.from_cyclecount(CC, bin_edges_stress, k, scale_init, SCF, plot=plot)

    @staticmethod
    def from_timeseries(
        df_timeseries: pd.DataFrame,
        bin_edges_stress: np.ndarray = None,
        k: float = None,
        scale_init: float = 5,
        time_col: str = "time",
        stress_col: str = "stress",
        SCF: float = None,
        plot: str = None 
    ) -> float:
        """
        Fit Weibull distribution from raw time series (time vs stress).

        Parameters
        ----------
        df_timeseries : pandas.DataFrame
            DataFrame with at least two columns: time and stress.
        bin_edges_stress : np.ndarray, optional
            Bin edges for stress ranges.
        k : float, optional
            Weibull shape parameter.
        scale_init : float, optional
            Initial guess for Weibull scale parameter. Defaults to 5.
        time_col : str, optional
            Column name for time values.
        stress_col : str, optional
            Column name for stress values.
        SCF : float, optional
            Stress concentration factor to scale stresses. Defaults to 1.
        plot : str, optional
            If 'observed', 'fitted', or 'both', generates a plot of the observed and fitted PDFs.

        Returns
        -------
        fitted_scale : float
            Estimated Weibull scale parameter.
        """
        if not isinstance(df_timeseries, pd.DataFrame):
            raise ValueError("Input must be a DataFrame with time and stress columns.")

        time = df_timeseries[time_col].to_numpy()
        stress = df_timeseries[stress_col].to_numpy()

        CC = pf.CycleCount.from_timeseries(time = time, data=stress)

        return WeibullFit.from_cyclecount(CC, bin_edges_stress, k, scale_init, SCF, plot=plot)

