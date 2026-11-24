"""
Newton-CG for regularized binary logistic regression.

This is deliberately an optimization example rather than a complete machine-
learning pipeline.  It does not create a train/test split or compute a
misclassification rate.  The emphasis is on the nested iteration:

    outer Newton iteration
        inner conjugate-gradient iteration
        Armijo backtracking line search

The first component of w is the bias because the first column of X is one.
The bias is not regularized.
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification


class Newton_CG_Logistic_Regression:
    def __init__(self, y, X, lam, armijo_c=0.1, backtrack_rho=0.5):
        self.y = np.asarray(y, dtype=float)
        self.X = np.asarray(X, dtype=float)
        self.lam = float(lam)
        self.armijo_c = float(armijo_c)
        self.backtrack_rho = float(backtrack_rho)

        self.N, self.D = self.X.shape

        if self.y.shape != (self.N,):
            raise ValueError("y must contain one label for each row of X")
        if not (0.0 < self.armijo_c < 1.0):
            raise ValueError("armijo_c must lie strictly between zero and one")
        if not (0.0 < self.backtrack_rho < 1.0):
            raise ValueError("backtrack_rho must lie strictly between zero and one")

    @staticmethod
    def log_partition(u):
        """log(1 + exp(u)), evaluated without overflow."""
        return np.logaddexp(0.0, u)

    @staticmethod
    def sigmoid(u):
        """Stable sigmoid without clipping the mathematical probability."""
        probability = np.empty_like(u, dtype=float)
        positive = u >= 0.0

        probability[positive] = 1.0 / (1.0 + np.exp(-u[positive]))

        exp_u = np.exp(u[~positive])
        probability[~positive] = exp_u / (1.0 + exp_u)
        return probability

    @staticmethod
    def without_bias(v):
        """Return a copy whose bias component is zero."""
        v_regularized = v.copy()
        v_regularized[0] = 0.0
        return v_regularized

    def objective(self, w):
        """
        E(w) = sum_n [log(1 + exp(x_n^T w)) - y_n x_n^T w]
               + (lambda/2) ||w_without_bias||^2.
        """
        u = self.X @ w
        data_term = np.sum(self.log_partition(u) - self.y * u)

        w_regularized = self.without_bias(w)
        regularizer = 0.5 * self.lam * (w_regularized @ w_regularized)
        return data_term + regularizer

    def gradient_and_curvature(self, w):
        """
        Return the gradient and the diagonal of R in

            g = X^T(p-y) + lambda Cw,
            H = X^T R X + lambda C.
        """
        u = self.X @ w
        probability = self.sigmoid(u)
        curvature = probability * (1.0 - probability)

        w_regularized = self.without_bias(w)
        gradient = self.X.T @ (probability - self.y)
        gradient += self.lam * w_regularized
        return gradient, curvature

    def hessian_vector_product(self, v, curvature):
        """Compute Hv without constructing the Hessian matrix H."""
        Xv = self.X @ v
        Hv = self.X.T @ (curvature * Xv)

        v_regularized = self.without_bias(v)
        Hv += self.lam * v_regularized
        return Hv

    @staticmethod
    def conjugate_gradient(A_mv, b, max_iterations, relative_tolerance=1.0e-3):
        """Approximately solve Ax=b using hand-written conjugate gradient."""
        x = np.zeros_like(b)
        residual = b.copy()  # b - A @ 0
        direction = residual.copy()

        residual_squared = residual @ residual
        initial_residual_norm = np.sqrt(residual_squared)

        if initial_residual_norm == 0.0:
            information = {
                "iterations": 0,
                "residual_norm": 0.0,
                "converged": True,
                "positive_curvature": True,
            }
            return x, information

        converged = False
        positive_curvature = True
        iterations_used = 0

        for iteration in range(1, max_iterations + 1):
            A_direction = A_mv(direction)
            directional_curvature = direction @ A_direction

            # For a positive-definite matrix, this quantity must be positive.
            if directional_curvature <= 0.0:
                positive_curvature = False
                break

            alpha_cg = residual_squared / directional_curvature
            x += alpha_cg * direction
            residual -= alpha_cg * A_direction

            new_residual_squared = residual @ residual
            residual_norm = np.sqrt(new_residual_squared)
            iterations_used = iteration

            if residual_norm <= relative_tolerance * initial_residual_norm:
                converged = True
                residual_squared = new_residual_squared
                break

            beta_cg = new_residual_squared / residual_squared
            direction = residual + beta_cg * direction
            residual_squared = new_residual_squared

        information = {
            "iterations": iterations_used,
            "residual_norm": np.sqrt(residual_squared),
            "converged": converged,
            "positive_curvature": positive_curvature,
        }
        return x, information

    def armijo_backtracking(
        self,
        w,
        direction,
        objective_at_w,
        gradient,
        alpha_initial=1.0,
        alpha_minimum=1.0e-12,
    ):
        """
        Find alpha such that

          E(w + alpha*d) <= E(w) + c*alpha*g^T d.
        """
        slope = gradient @ direction
        if slope >= 0.0:
            raise ValueError("Armijo requires a descent direction: g^T d < 0")

        alpha = alpha_initial
        number_of_backtracks = 0

        while alpha >= alpha_minimum:
            trial_objective = self.objective(w + alpha * direction)
            armijo_bound = objective_at_w + self.armijo_c * alpha * slope
            armijo_margin = armijo_bound - trial_objective

            print(
                f"        trial alpha={alpha:.3e}  "
                f"Armijo margin={armijo_margin:.3e}"
            )

            if armijo_margin >= 0.0:
                return alpha, number_of_backtracks, True

            alpha *= self.backtrack_rho
            number_of_backtracks += 1

        return 0.0, number_of_backtracks, False


def newton_cg_optimize(
    solver,
    w_initial,
    maximum_outer_iterations=40,
    maximum_cg_iterations=100,
    cg_relative_tolerance=1.0e-3,
    gradient_relative_tolerance=1.0e-6,
):
    """Minimize the regularized logistic objective using Newton-CG."""
    w = w_initial.copy()
    objective_history = [solver.objective(w)]

    initial_gradient, _ = solver.gradient_and_curvature(w)
    initial_gradient_norm = np.linalg.norm(initial_gradient)
    gradient_scale = max(1.0, initial_gradient_norm)

    print(
        f"outer={0:2d}  E_LR={objective_history[-1]:.6e}  "
        f"||g||={initial_gradient_norm:.3e}"
    )

    for outer_iteration in range(1, maximum_outer_iterations + 1):
        objective_at_w = objective_history[-1]
        gradient, curvature = solver.gradient_and_curvature(w)
        gradient_norm = np.linalg.norm(gradient)

        if gradient_norm <= gradient_relative_tolerance * gradient_scale:
            print("Outer stopping condition satisfied: the gradient is small.")
            break

        # The matrix in the Newton system is represented only by its action.
        H_mv = lambda v: solver.hessian_vector_product(v, curvature)
        right_hand_side = -gradient

        newton_direction, cg_information = solver.conjugate_gradient(
            H_mv,
            right_hand_side,
            max_iterations=maximum_cg_iterations,
            relative_tolerance=cg_relative_tolerance,
        )

        slope = gradient @ newton_direction

        # An inexact solve need not always provide a descent direction.
        if slope >= 0.0:
            print("        CG direction was not a descent direction; using -g.")
            newton_direction = -gradient
            slope = -(gradient @ gradient)

        alpha, backtracks, line_search_succeeded = solver.armijo_backtracking(
            w,
            newton_direction,
            objective_at_w,
            gradient,
            alpha_initial=1.0,
        )

        if not line_search_succeeded:
            print("Armijo line search failed. Optimization stopped.")
            break

        # Danger, Will Robinson: this is the state update.
        w = w + alpha * newton_direction
        new_objective = solver.objective(w)
        objective_history.append(new_objective)
        new_gradient, _ = solver.gradient_and_curvature(w)
        new_gradient_norm = np.linalg.norm(new_gradient)

        print(
            f"outer={outer_iteration:2d}  "
            f"E_LR={new_objective:.6e}  "
            f"||g_new||={new_gradient_norm:.3e}  "
            f"CG iters={cg_information['iterations']:2d}  "
            f"CG residual={cg_information['residual_norm']:.3e}  "
            f"gTd={slope:.3e}  "
            f"alpha={alpha:.3e}  "
            f"backtracks={backtracks}"
        )

    return w, objective_history


def make_optimization_example():
    """Construct a deliberately difficult binary logistic problem."""
    X, y = make_classification(
        n_samples=1500,
        n_features=100,
        n_informative=10,
        n_redundant=70,
        n_repeated=10,
        n_clusters_per_class=2,
        class_sep=1.0,
        flip_y=0.03,
        random_state=7,
    )

    # Unequal feature scales make truncated CG work harder.
    X = X * np.logspace(-2, 2, X.shape[1])

    # The first column corresponds to the bias w[0].
    X = np.column_stack([np.ones(X.shape[0]), X])

    # Begin confidently in approximately the wrong direction.
    mean_difference = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
    mean_difference[0] = 0.0
    w_initial = -10.0 * mean_difference / np.linalg.norm(mean_difference)

    return X, y, w_initial


if __name__ == "__main__":
    X, y, w_initial = make_optimization_example()

    solver = Newton_CG_Logistic_Regression(
        y=y,
        X=X,
        lam=1.0e-03,
        armijo_c=0.1,
        backtrack_rho=0.5,
    )

    w, objective_history = newton_cg_optimize(
        solver,
        w_initial,
        maximum_outer_iterations=40,
        maximum_cg_iterations=100,
        cg_relative_tolerance=1.0e-3,
        gradient_relative_tolerance=1.0e-6,
    )

    print("final w:", w)

    # No train/test split or classification error is computed here.  The plot
    # shows only the behavior of the optimization algorithm.
    plt.plot(range(len(objective_history)), objective_history, marker="o")
    plt.xlabel("outer iteration")
    plt.ylabel(r"$E_{LR}(w)$")
    plt.title("Newton-CG with Armijo backtracking")
    plt.tight_layout()
    # plt.savefig("Newton_CG_Algorithm_Binary_LR.pdf")
    plt.show()
