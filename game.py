import numpy as np
import matplotlib.pyplot as plt
import nashpy as nash
from scipy.signal import hilbert


class Play:
    """
    Run fictitious play on a 2‑player zero‑sum game A (row vs. col).

    Attributes:
      A                     : (m×n) payoff matrix for the row player
      iterations            : number of FP rounds
      game                  : nashpy.Game(A, -A)
      x_eq, y_eq            : equilibrium mixed strategies
      x_hat                 : (T×m) empirical row mixes
      y_hat                 : (T×n) empirical col mixes
      x_error               : (T,)  L∞ errors for row mixes
      y_error               : (T,)  L∞ errors for col mixes
      x_convergence_iter    : int, first k where x_error[k] ≤ ε
      y_convergence_iter    : int, first k where y_error[k] ≤ ε
    """

    def __init__(self, A, iterations):
        self.A = np.asarray(A, dtype=float)
        self.iterations = int(iterations)
        # build and check zero‑sum game
        self.game = nash.Game(self.A, -self.A)
        assert self.game.zero_sum, "Matrix must define a zero‑sum game"

        # compute mixed equilibrium
        eqs = list(self.game.support_enumeration())

        if not eqs:
            raise ValueError("No equilibrium found via support enumeration")
        self.x_eq, self.y_eq = eqs[0]
        # print(f"Equilibrium mixes: x* = {self.x_eq}, y* = {self.y_eq}")

        # placeholders
        self.x_hat = None
        self.y_hat = None
        self.x_error = None
        self.y_error = None
        self.error = None
        self.x_convergence_iter = None
        self.y_convergence_iter = None

    def run(self):
        """
        Run fictitious play for self.iterations rounds.
        Fills self.x_hat and self.y_hat arrays.
        """
        history = list(
            self.game.fictitious_play(iterations=self.iterations)
        )
        # history[k] = (row_counts, col_counts), with k=0…T
        row_counts = np.stack([r for r, _ in history], axis=0)[1:]
        col_counts = np.stack([c for _, c in history], axis=0)[1:]
        self.x_hat = row_counts / row_counts.sum(axis=1, keepdims=True)
        self.y_hat = col_counts / col_counts.sum(axis=1, keepdims=True)
        
    def calc_error(self):
        """
        Compute the convergence exponent alpha by fitting
            envelope(t) ≈ C * t^(−alpha)
        to the Hilbert‐transform envelope of the combined error
            e(t) = max{|x_hat[t,0]-x_eq[0]|, |y_hat[t,0]-y_eq[0]|}.
        We discard the first `burn_in` and last `burn_out` points
        to avoid transients and edge artifacts.
        """
        if self.x_hat is None or self.y_hat is None:
            raise RuntimeError("Must call .run() before .calc_error()")

        # 1) coordinate‐wise error
        ex = np.abs(self.x_hat[:, 0] - self.x_eq[0])
        ey = np.abs(self.y_hat[:, 0] - self.y_eq[0])

        # 2) combine into single error time‐series
        e = np.maximum(ex, ey)
        self.x_error = ex
        self.y_error = ey
        self.error   = e

    def compute_spectrum(self, which="x"):
        """
        Compute the one‐sided Fourier amplitude spectrum of
        either x_error or y_error.
        Returns (freqs, amps) where
          freqs = np.fft.rfftfreq(T, d=1)
          amps  = |FFT[e_error]| / T
        """
        if which == "x":
            err = self.x_error
        elif which == "y":
            err = self.y_error
        else:
            raise ValueError("which must be 'x' or 'y'")
        T = len(err)
        # do real‐FFT
        fftvals = np.fft.rfft(err)
        freqs   = np.fft.rfftfreq(T, d=1.0)   # sampling step=1 iteration
        amps    = np.abs(fftvals) / T          # normalize amplitude
        return freqs, amps

    def dominant_freq(self, which="x", fmin=1e-6):
        """
        Find the frequency (and amplitude) of the largest peak in
        the spectrum of x_error or y_error, ignoring the zero‐freq.
        Returns (f_dom, A_dom).
        """
        freqs, amps = self.compute_spectrum(which)
        # ignore the DC component at freqs[0]=0:
        idx = np.argmax(amps[1:]) + 1
        return freqs[idx], amps[idx]

    def compute_envelope(self, which="x"):
        """
        Compute the analytic signal and its amplitude‐envelope
        via the Hilbert transform:
          envelope[k] = | hilbert(err)[k] |
        """
        if which == "x":
            err = self.x_error
        else:
            err = self.y_error
        analytic = hilbert(err)
        envelope = np.abs(analytic)
        return envelope
    
    # def calc_error(self, epsilon, window_size=100):
    #     """
    #     Given ε > 0 and a required window_size w, compute the L∞
    #     errors e_x, e_y, then find the first k such that
    #       e_x[k:k+w] < ε   (and similarly for e_y).
    #     Prints the two convergence points and
    #     asserts that both exist within self.iterations.
    #     """
    #     if self.x_hat is None or self.y_hat is None:
    #         raise RuntimeError("Must call .run() before .calc_error()")

    #     # 1) compute L∞ errors per iteration
    #     self.x_error = np.abs(self.x_hat - self.x_eq).max(axis=1)
    #     self.y_error = np.abs(self.y_hat - self.y_eq).max(axis=1)

    #     # 2) helper to find first window of length w
    #     def first_hitting(err, ε, w):
    #         for k in range(len(err) - w + 1):
    #             if np.all(err[k : k + w] < ε):
    #                 return k + 1         # 1‑based
    #         return None

    #     # 3) find row‑player convergence
    #     kx = first_hitting(self.x_error, epsilon, window_size)
    #     assert kx is not None, (
    #         f"Row‐player never stayed below {epsilon} "
    #         f"for {window_size} consecutive steps"
    #     )
    #     self.x_convergence_iter = kx
    #     # print(f"Row errors hit ε={epsilon:g} for {window_size} steps from k={kx}")

        # # 4) find col‑player convergence
        # ky = first_hitting(self.y_error, epsilon, window_size)
        # assert ky is not None, (
        #     f"Col‐player never stayed below {epsilon} "
        #     f"for {window_size} consecutive steps"
        # )
        # self.y_convergence_iter = ky
        # # print(f"Col errors hit ε={epsilon:g} for {window_size} steps from k={ky}")


    def plot_error(self, epsilon):
        """
        Plot the per–player convergence errors on a semilog scale.
        Requires that .run() and .calc_error(epsilon) have been called.
        """
        if self.x_error is None or self.y_error is None:
            # automatically compute errors if missing
            self.calc_error(epsilon)

        T = len(self.x_error)
        t = np.arange(1, T + 1)
        kx = self.x_convergence_iter
        ky = self.y_convergence_iter

        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        # Row‐player error panel
        ax[0].semilogy(t, self.x_error, color="red", label=r"$e_x^k$")
        ax[0].axhline(epsilon, color="gray", linestyle=":", label=r"$\varepsilon$")
        if kx:
            ax[0].axvline(kx, color="red", linestyle="--",
                          label=rf"$k^x_{{\rm conv}}={kx}$")
        ax[0].set_title(r"Row‐player error $e_x^k$")
        ax[0].set_xlabel(r"Iteration $k$")
        ax[0].set_ylabel(r"Error $\|\hat x^k - x^*\|_\infty$")
        ax[0].legend(loc="upper right")

        # Column‐player error panel
        ax[1].semilogy(t, self.y_error, color="blue", label=r"$e_y^k$")
        ax[1].axhline(epsilon, color="gray", linestyle=":", label=r"$\varepsilon$")
        if ky:
            ax[1].axvline(ky, color="blue", linestyle="--",
                          label=rf"$k^y_{{\rm conv}}={ky}$")
        ax[1].set_title(r"Col‐player error $e_y^k$")
        ax[1].set_xlabel(r"Iteration $k$")
        ax[1].legend(loc="upper right")

        plt.suptitle("Per‑Player Convergence Errors in Fictitious Play")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def plot_trajectories(self):
        """
        Plot the empirical strategy trajectories x_hat and y_hat against
        their equilibrium mixes x_eq and y_eq. Assumes .run() has been called.
        """
        if self.x_hat is None or self.y_hat is None:
            raise RuntimeError("Must call .run() before plotting trajectories")

        T, m = self.x_hat.shape
        _, n = self.y_hat.shape
        t = np.arange(1, T + 1)
        x1, x2 = self.x_eq if m == 2 else (None, None)
        y1, y2 = self.y_eq if n == 2 else (None, None)

        fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        # Row‐player trajectories
        ax[0].plot(t, self.x_hat[:, 0], color="red",  label=r"$\hat x^k_1$")
        ax[0].plot(t, self.x_hat[:, 1], color="blue", label=r"$\hat x^k_2$")
        if m == 2:
            ax[0].axhline(x1, color="red", linestyle="--", label=r"$x^*_1$")
            ax[0].axhline(x2, color="blue", linestyle="--", label=r"$x^*_2$")
            ax[0].set_ylim(0, 1)
        ax[0].set_title(
            rf"Row FP vs Nash ($x^*=[{x1:.3f},\,{x2:.3f}]$)" if m == 2 else "Row FP"
        )
        ax[0].set_xlabel(r"Iteration $k$")
        ax[0].set_ylabel(r"Probability")
        ax[0].legend(loc="upper right")

        # Column‐player trajectories
        ax[1].plot(t, self.y_hat[:, 0], color="green",  label=r"$\hat y^k_1$")
        ax[1].plot(t, self.y_hat[:, 1], color="purple", label=r"$\hat y^k_2$")
        if n == 2:
            ax[1].axhline(y1, color="green", linestyle="--", label=r"$y^*_1$")
            ax[1].axhline(y2, color="purple", linestyle="--", label=r"$y^*_2$")
            ax[1].set_ylim(0, 1)
        ax[1].set_title(
            rf"Column FP vs Nash ($y^*=[{y1:.3f},\,{y2:.3f}]$)" if n == 2 else "Column FP"
        )
        ax[1].set_xlabel(r"Iteration $k$")
        ax[1].legend(loc="upper right")

        plt.suptitle(r"Convergence of Fictitious Play in a $2\times 2$ Zero‐Sum Game")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
