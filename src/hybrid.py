import numpy as np
from typing import Optional, Tuple
from numpy.linalg import inv
from numpy.linalg import pinv
import matplotlib.pyplot as plt
import pandas as pd

class LSTM_SSM:
    """State-space models for modeling baselines:
    Define model matrices [A,Q,F] and initial hidden states zB and SzB

    Attributes:
        Define
    """

    def __init__(
        self,
        neural_network,
        baseline: str,
        zB: Optional[np.ndarray] = None,
        SzB: Optional[np.ndarray] = None,
    ) -> None:
        self.net = neural_network
        self.baseline = baseline
        self.z = np.concatenate((zB, np.array([0])), axis=0).reshape(-1,1)
        self.Sz = np.diag(np.concatenate((SzB, np.array([0])), axis=0))
        self.nb_hs = len(self.z)
        self.define_matrices()

    def __call__(
        self, mu_x: np.ndarray, var_x: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        # lstm forward
        m_pred_lstm, v_pred_lstm = self.net(mu_x, var_x)
        m_pred_lstm = np.array([m_pred_lstm[0]])  # check with Ha why?
        v_pred_lstm = np.array([v_pred_lstm[0]])  # check with Ha why?
        # hybrid forward
        m_pred, v_pred  = self.forward(mu_lstm=m_pred_lstm, var_lstm=v_pred_lstm)

        return m_pred, v_pred, m_pred_lstm, v_pred_lstm

    def forward(
        self, mu_lstm: np.ndarray, var_lstm: np.ndarray,
    ):
        # Prediction step:
        z_prior  = self.A @ self.z
        Sz_prior = self.A @ self.Sz @ self.A.T + self.Q

        # Replace lstm prediction (mu_lstm and var_lstm) to the hidden state
        z_prior[-1,-1]  = mu_lstm[0]
        Sz_prior[-1,-1] = var_lstm[0]

        # Predicted mean and var
        m_pred = self.F @ z_prior
        var_pred = self.F @ Sz_prior @ self.F.T

        # save the priors for hidden states
        self.mu_priors.append(z_prior)
        self.cov_priors.append(Sz_prior)
        self.mu_y_pred.append(m_pred)
        self.var_y_pred.append(var_pred)
        self.z  = z_prior
        self.Sz = Sz_prior

        return m_pred, var_pred

    def backward(
        self,
        mu_obs: Optional[np.ndarray] = None,
        var_obs: Optional[np.ndarray] = None,
    ):
        # load variables
        z_prior  = self.mu_priors[-1]
        Sz_prior = self.cov_priors[-1]
        y_pred   = self.mu_y_pred[-1]
        Sy_pred  = self.var_y_pred[-1]
        var_lstm = Sz_prior[-1,-1]

        if ~np.isnan(mu_obs):
            #
            cov_zy =  Sz_prior @ self.F.T
            var_y = Sy_pred + var_obs
            # delta for mean z and var Sz
            cov_= cov_zy/var_y
            delta_mean =  cov_ * (mu_obs - y_pred)
            delta_var  = - cov_ @ cov_zy.T
            # update mean for mean z and var Sz
            z_posterior = z_prior + delta_mean
            Sz_posterior = Sz_prior + delta_var
            # detla for mean and var to update LSTM (parameters in net)
            delta_mean_lstm = delta_mean[-1,-1]/var_lstm
            delta_var_lstm  = delta_var[-1,-1]/var_lstm**2

            # # update lstm network
            self.net.input_delta_z_buffer.delta_mu = np.array([delta_mean_lstm]).flatten()
            self.net.input_delta_z_buffer.delta_var = np.array([delta_var_lstm]).flatten()
            self.net.backward()
            self.net.step()
        else:
            z_posterior  = z_prior
            Sz_posterior = Sz_prior

        # save
        self.z  = z_posterior
        self.Sz = Sz_posterior
        self.mu_posteriors.append(z_posterior)
        self.cov_posteriors.append(Sz_posterior)

    def smoother(self):
        nb_obs = len(self.mu_priors)
        nb_hs = self.nb_hs
        mu_smoothed  = [None] * nb_obs
        cov_smoothed = [None] * nb_obs
        mu_smoothed[-1] = self.mu_posteriors[-1]
        cov_smoothed[-1] = self.cov_posteriors[-1]
        A = self.A

        for i in range(nb_obs-2,-1,-1):
            J = self.cov_posteriors[i] @ A.T \
                @ pinv(self.cov_priors[i+1],rcond=1e-12)
            mu_smoothed[i] = self.mu_posteriors[i] \
                + J @ (mu_smoothed[i+1] - self.mu_priors[i+1])
            cov_ = self.cov_posteriors[i] + \
                J @ (cov_smoothed[i+1] - self.cov_priors[i+1]) @ J.T
            cov_smoothed[i] = cov_

        self.mu_smoothed  = mu_smoothed
        self.cov_smoothed = cov_smoothed

    def init_ssm_hs(self):
        self.mu_y_pred  = list()
        self.var_y_pred = list()
        self.mu_priors  = list()
        self.cov_priors = list()
        self.mu_posteriors  = list()
        self.cov_posteriors = list()
        if hasattr(self,'mu_smoothed'):
            self.z  = self.mu_smoothed[0]
            Sz_ = self.cov_smoothed[0]
            Sz_ = np.diag(np.diag(Sz_))
            self.Sz = Sz_

    def define_matrices(self):
        if self.baseline == 'level':
            self.A = np.diag([[1][0]])
            self.Q = np.zeros((2,2))
            self.F = np.array([1,1]).reshape(1, -1)
        elif self.baseline == 'trend':
            self.A = np.array([[1,1,0],[0,1,0], [0,0,0]])
            self.Q = np.zeros((3,3))
            self.F = np.array([1,0,1]).reshape(1, -1)
        elif self.baseline == 'acceleration':
            self.A = np.array([[1,1,0.5,0],[0,1,1,0],[0,0,1,0],[0,0,0,0]])
            self.Q = np.zeros((4,4))
            self.F = np.array([1,0,0,1]).reshape(1, -1)


def process_input_ssm(
    mu_x: np.ndarray,
    mu_preds_lstm: list,
    var_preds_lstm: list,
    input_seq_len: int,
    num_features: int,
    ):
    mu_preds_lstm = np.array(mu_preds_lstm)
    var_preds_lstm = np.array(var_preds_lstm)
    var_x = np.zeros(mu_x.shape)
    nb_replace = min(len(mu_preds_lstm), input_seq_len)

    if nb_replace > 0:
        mu_x[-nb_replace*num_features::num_features]  =  mu_preds_lstm[-nb_replace:]
        var_x[-nb_replace*num_features::num_features] =  var_preds_lstm[-nb_replace:]
    return mu_x, var_x

class PredictionViz:
    """Visualization of prediction
    Attributes:
        task_name: Name of the task such as autoencoder
        data_name: Name of dataset such as Boston housing or toy example
        figsize: Size of figure
        fontsize: Font size for letter in the figure
        lw: linewidth
        ms: Marker size
        ndiv_x: Number of divisions for x-direction
        ndiv_y: Number of division for y-direciton
    """

    def __init__(
        self,
        task_name: str,
        data_name: str,
        figsize: tuple = (12, 12),
        fontsize: int = 28,
        lw: int = 3,
        ms: int = 10,
        ndiv_x: int = 4,
        ndiv_y: int = 4,
    ) -> None:
        self.task_name = task_name
        self.data_name = data_name
        self.figsize = figsize
        self.fontsize = fontsize
        self.lw = lw
        self.ms = ms
        self.ndiv_x = ndiv_x
        self.ndiv_y = ndiv_y

    def load_dataset(self, file_path: str, header: bool = False) -> np.ndarray:
        """Load dataset (*.csv)
        Args:
            file_path: File path to the data file
            header: Ignore hearder ?

        """

        # Load image data from *.csv file
        if header:
            df = pd.read_csv(file_path, skiprows=1, delimiter=",", header=None)
        else:
            df = pd.read_csv(file_path, skiprows=0, delimiter=",", header=None)

        return df[0].values

    def plot_predictions(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        sy_pred: np.ndarray,
        std_factor: int,
        x_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        sy_test: Optional[np.ndarray] = None,
        label: str = "diag",
        title: Optional[str] = None,
        eq: Optional[str] = None,
        x_eq: Optional[float] = None,
        y_eq: Optional[float] = None,
        time_series: bool = False,
        save_folder: Optional[str] = None,
    ) -> None:
        """Compare prediciton distribution with theorical distribution

        x_train: Input train data
        y_train: Output train data
        x_test: Input test data
        y_test: Output test data
        y_pred: Prediciton of network
        sy_pred: Standard deviation of the prediction
        std_factor: Standard deviation factor
        sy_test: Output test's theorical standard deviation
        label: Name of file
        title: Figure title
        eq: Math equation for data
        x_eq: x-coordinate for eq
        y_eq: y-coordinate for eq

        """

        # Get max and min values
        if sy_test is not None:
            std_y = max(sy_test)
        else:
            std_y = 0

        if x_train is not None:
            max_y = np.maximum(max(y_test), max(y_train)) + std_y
            min_y = np.minimum(min(y_test), min(y_train)) - std_y
            max_x = np.maximum(max(x_test), max(x_train))
            min_x = np.minimum(min(x_test), min(x_train))
        else:
            max_y = max(y_test) + std_y
            min_y = min(y_test) - std_y
            max_x = max(x_test)
            min_x = min(x_test)

        # Plot figure
        plt.figure(figsize=self.figsize)
        ax = plt.axes()
        ax.set_title(title, fontsize=1.1 * self.fontsize, fontweight="bold")
        if eq is not None:
            ax.text(x_eq, y_eq, eq, color="k", fontsize=self.fontsize)
        ax.plot(x_test, y_pred, "r", lw=self.lw, label=r"$\mathbb{E}[Y^{'}]$")
        ax.plot(x_test, y_test, "k", lw=self.lw, label=r"$y_{true}$")

        ax.fill_between(
            x_test,
            y_pred - std_factor * sy_pred,
            y_pred + std_factor * sy_pred,
            facecolor="red",
            alpha=0.3,
            label=r"$\mathbb{{E}}[Y^{{'}}]\pm{}\sigma$".format(std_factor),
        )
        if sy_test is not None:
            ax.fill_between(
                x_test,
                y_test - std_factor * sy_test,
                y_test + std_factor * sy_test,
                facecolor="blue",
                alpha=0.3,
                label=r"$y_{{test}}\pm{}\sigma$".format(std_factor),
            )
        if x_train is not None:
            if time_series:
                marker = ""
                line_style = "-"
            else:
                marker = "o"
                line_style = ""
            ax.plot(
                x_train,
                y_train,
                "b",
                marker=marker,
                mfc="none",
                lw=self.lw,
                ms=0.2 * self.ms,
                linestyle=line_style,
                label=r"$y_{train}$",
            )

        ax.set_xlabel(r"$x$", fontsize=self.fontsize)
        ax.set_ylabel(r"$y$", fontsize=self.fontsize)
        if time_series:
            x_ticks = pd.date_range(min_x, max_x, periods=self.ndiv_x).values
        else:
            x_ticks = np.linspace(min_x, max_x, self.ndiv_x)
        y_ticks = np.linspace(min_y, max_y, self.ndiv_y)
        ax.set_yticks(y_ticks)
        ax.set_xticks(x_ticks)
        ax.tick_params(
            axis="both", which="both", direction="inout", labelsize=self.fontsize
        )
        ax.legend(
            loc="upper right",
            edgecolor="black",
            fontsize=1 * self.fontsize,
            ncol=1,
            framealpha=0.3,
            frameon=False,
        )

        ax.set_ylim([min_y, max_y])
        ax.set_xlim([min_x, max_x])

        # Save figure
        if save_folder is None:
            plt.show()
        else:
            saving_path = f"saved_results/pred_{label}_{self.data_name}.png"
            plt.savefig(saving_path, bbox_inches="tight")
            plt.close()
