# GNN Example: E. coli Chemotaxis
# GNN Version: 1.2
# 2-D gradient climb toward a nutrient source whose sensing sharpens with proximity

## GNNSection
ActInfContinuous

## GNNVersionAndFlags
GNN v1.2

## ModelName
E coli Chemotaxis

## ModelAnnotation
A continuous-state Active Inference gradient climber rendered as a native
linear-Gaussian state-space model (LGSSM) with a state-dependent observation
noise R(x). The bacterium reads a chemoattractant field whose signal-to-noise
ratio improves near the source and degrades with distance from it —
chemotaxis is information limited, and the information available to the cell
depends on where in the gradient it sits (Mattingly, Kamino, Machta & Emonts,
"Escherichia coli chemotaxis is information limited", Nature Physics 17,
1426–1431, 2021).
- Hidden state x = (x, y): the cell's planar position in the dish.
- Observation y: a noisy position estimate the cell infers from the local
  attractant readings; a `quadratic_beacon` family centred on the source at
  (3.0, 1.0) gives the sharpest sensing at the source (r_min = 0.1) and noise
  growing quadratically with distance (k = 0.5).
- Goal: the source itself, goal_mean = (3.0, 1.0). The proportional gain
  control_gain = 0.25 is what the parity/jax backends use; the cpomdp backend
  runs an enumerated expected-free-energy search over its swim directions.
The dynamics carry a mild drift-cancelling damping (F = 0.95 I) so a cell that
stops swimming diffuses back toward the origin. All numbers — positions,
gains, noise floors, the damping — are illustrative and are not fitted to the
cited measurements.

## StateSpaceBlock
# Continuous latent state x = (x, y) position in the dish
x[2,1,type=float]        # continuous latent state
# Continuous observation y = noisy position estimate from the attractant field
y[2,1,type=float]        # continuous observation
# Swim input added to the state each step
u[2,1,type=float]        # control input
# State transition matrix (mild damping toward the origin)
F[2,2,type=float]        # state transition
# Observation matrix (identity readout of the position)
H[2,2,type=float]        # observation matrix
# Process-noise covariance (run-and-tumble jitter)
Q[2,2,type=float]        # process-noise covariance
# Nominal observation-noise covariance (scaled by the sensor family)
R[2,2,type=float]        # observation-noise covariance
# State-dependent observation-noise family and its parameters
R_x_family[1,type=string] # quadratic_beacon
R_x_params[4,type=float]  # [cx, cy, r_min, k]
# Gaussian prior over the starting position
prior_mean[2,type=float] # prior mean over the initial latent state
prior_cov[2,2,type=float]# prior covariance over the initial latent state
# The nutrient source
goal_mean[2,type=float]  # preferred state (goal)
# Scalar proportional control gain (parity / non-cpomdp backends)
control_gain[1,type=float] # scalar proportional gain
# Time index
t[1,type=int]            # discrete time step

## Connections
prior_mean>x
F>x
x>y
H>y
Q>x
R>y
R_x_family>y
R_x_params>y
u>x
goal_mean>u
control_gain>u

## InitialParameterization
# F: mild damping — without swimming the cell drifts back toward the origin.
F={
  (0.95, 0.0),
  (0.0, 0.95)
}

# H: identity readout of the position.
H={
  (1.0, 0.0),
  (0.0, 1.0)
}

# Q: run-and-tumble jitter per step.
Q={
  (0.04, 0.0),
  (0.0, 0.04)
}

# R: nominal sensing noise far from the source; the family scales it by s(x).
R={
  (0.3, 0.0),
  (0.0, 0.3)
}

# quadratic_beacon centred on the source: sharpest sensing at (3.0, 1.0),
# noise floor 0.1 x R there, growing as 0.5 x distance^2.
R_x_family=quadratic_beacon
R_x_params={(3.0, 1.0, 0.1, 0.5)}

# Gaussian prior over the starting position: near the origin, uncertain.
prior_mean={(0.0, 0.0)}
prior_cov={
  (0.5, 0.0),
  (0.0, 0.5)
}

# goal_mean: the nutrient source.
goal_mean={(3.0, 1.0)}

# control_gain: proportional gain used by the parity and non-cpomdp backends.
control_gain={(0.25)}

## Equations
# Generative model (linear-Gaussian state-space with state-dependent sensing):
#   x_1 ~ N(prior_mean, prior_cov)
#   x_t = F x_{t-1} + u_{t-1} + N(0, Q)
#   y_t = H x_t + N(0, R(x_t)),  R(x) = R * (r_min + k * |x - c|^2)
# cpomdp control: u_t = first action of argmin_pi G(pi), G = pragmatic - epistemic.
# Parity / other backends: u_t = control_gain * (goal_mean - mu_t).

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=25

## ActInfOntologyAnnotation
F=StateTransitionMatrix
H=ObservationMatrix
Q=ProcessNoiseCovariance
R=ObservationNoiseCovariance
R_x_family=StateDependentObservationNoise
R_x_params=StateDependentObservationNoise
prior_mean=PriorMean
prior_cov=PriorCovariance
goal_mean=PreferredState
control_gain=ControlGain
x=ContinuousHiddenState
y=ContinuousObservation
u=ControlInput
t=Time

## ModelParameters
num_timesteps: 25
dt: 0.1
random_seed: 42
num_states: 2
num_observations: 2
cpomdp_control_mode: efe
cpomdp_action_scale: 0.3
cpomdp_horizon: 1
cpomdp_goal_precision: 1.0

## Footer
E. coli Chemotaxis v1.2 - native linear-Gaussian GNN model with a state-dependent sensor.
Gradient climb toward a source that is easiest to sense up close; numbers illustrative.

## Signature
Cryptographic signature goes here
