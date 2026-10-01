# GNN Example: Curious Rocket
# GNN Version: 1.2
# 2-D thrusters-only rocket steering to a landing pad under a state-dependent sensor: one beacon (sharp sensing) and one blind spot (dead sensing)

## GNNSection
ActInfContinuous

## GNNVersionAndFlags
GNN v1.2

## ModelName
Curious Rocket

## ModelAnnotation
A continuous-state Active Inference rocket rendered as a native linear-Gaussian
state-space model (LGSSM) with a state-dependent observation noise R(x):
- Hidden state x = (x, y): the rocket's planar position. Thrusters are the only
  dynamics: the control input u is added to the state each step (F = I).
- Observation y: a noisy position fix whose precision depends on where the
  rocket is. A beacon at (1.5, 0.5) makes the fix sharp nearby (r_min = 0.05,
  growing quadratically with distance, k = 0.8); a blind spot at (-1.0, 1.5)
  saturates the noise at 25x the nominal R inside a disc of radius 0.6.
- Goal: the landing pad goal_mean = (2.0, 2.0). The proportional gain
  control_gain = 0.3 is what the parity/jax backends use; the cpomdp backend
  steers by an enumerated expected-free-energy (EFE) search instead.

Collapse and revival. Under a fixed linear-Gaussian sensor the epistemic term of
EFE is the same for every policy — nothing the rocket does changes how much it
will learn — so every backend that keeps the nominal R (JAX, NumPyro, PyTorch,
Stan, RxInfer.jl) can only trade goal distance against control effort. With
R(x) declared, cpomdp's `epistemic_by_policy_t0` differs across the nine
compass-rose policies: moves toward the beacon carry more expected
information gain (pull), moves into the blind spot carry less (push), and the
selected policy is the argmin of pragmatic minus epistemic. The goal
precision sets the balance, exactly as in cpomdp's own examples: at
cpomdp_goal_precision 1.0 the rocket is over-curious and parks at the beacon;
at 2.0 (the value declared here) it detours past the beacon and then commits
to the pad; at 8.0 it flies straight and never reads the beacon closely.
Run both backends on this file (`--frameworks jax,cpomdp`) to see the empty
`efe_history` on jax next to the per-policy record on cpomdp. All numbers are
illustrative.

## StateSpaceBlock
# Continuous latent state x = (x, y) position
x[2,1,type=float]        # continuous latent state
# Continuous observation y = noisy position fix
y[2,1,type=float]        # continuous observation
# Thruster input added to the state each step
u[2,1,type=float]        # control input
# State transition matrix (position persists; thrusters move it)
F[2,2,type=float]        # state transition
# Observation matrix (identity readout of the position)
H[2,2,type=float]        # observation matrix
# Process-noise covariance (thruster jitter)
Q[2,2,type=float]        # process-noise covariance
# Nominal observation-noise covariance (scaled by the sensor family)
R[2,2,type=float]        # observation-noise covariance
# State-dependent observation-noise family and its parameters
R_x_family[1,type=string] # beacon_and_blind_spot
R_x_params[8,type=float]  # [bx, by, r_min, k, sx, sy, radius, r_dead]
# Gaussian prior over the launch position
prior_mean[2,type=float] # prior mean over the initial latent state
prior_cov[2,2,type=float]# prior covariance over the initial latent state
# The landing pad
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
# F: thrusters only — the position persists and every move comes from u.
F={
  (1.0, 0.0),
  (0.0, 1.0)
}

# H: identity readout — the fix reads the 2D position directly.
H={
  (1.0, 0.0),
  (0.0, 1.0)
}

# Q: thruster jitter per step.
Q={
  (0.02, 0.0),
  (0.0, 0.02)
}

# R: nominal fix noise; the sensor family scales it by s(x).
R={
  (0.2, 0.0),
  (0.0, 0.2)
}

# One beacon (sharp sensing, pull) and one blind spot (dead sensing, push):
# beacon at (1.5, 0.5) with floor 0.05 and quadratic growth 0.8; blind spot at
# (-1.0, 1.5), radius 0.6, saturating at 25x the nominal R.
R_x_family=beacon_and_blind_spot
R_x_params={(1.5, 0.5, 0.05, 0.8, -1.0, 1.5, 0.6, 25.0)}

# Gaussian prior over the launch position: the origin, fairly certain.
prior_mean={(0.0, 0.0)}
prior_cov={
  (0.3, 0.0),
  (0.0, 0.3)
}

# goal_mean: the landing pad.
goal_mean={(2.0, 2.0)}

# control_gain: proportional gain used by the parity and non-cpomdp backends.
control_gain={(0.3)}

## Equations
# Generative model (linear-Gaussian state-space with state-dependent sensing):
#   x_1 ~ N(prior_mean, prior_cov)
#   x_t = F x_{t-1} + u_{t-1} + N(0, Q)
#   y_t = H x_t + N(0, R(x_t)),  R(x) = R * s(x)
#   s(x) = (r_min + k * |x - b|^2) * (1 + (r_dead - 1) * exp(-|x - c|^2 / radius^2))
# cpomdp control: u_t = first action of argmin_pi G(pi), G = pragmatic - epistemic,
#   over the compass-rose action set; epistemic = 1/2 (ln det S - ln det R(mu+)).
# Parity / other backends: u_t = control_gain * (goal_mean - mu_t).

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=30

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
num_timesteps: 30
dt: 0.1
random_seed: 42
num_states: 2
num_observations: 2
cpomdp_control_mode: efe
cpomdp_action_scale: 0.4
cpomdp_horizon: 1
cpomdp_goal_precision: 2.0

## Footer
Curious Rocket v1.2 - native linear-Gaussian GNN model with a state-dependent sensor.
Thrusters-only 2D rocket: the beacon pulls, the blind spot pushes, the pad is the goal.

## Signature
Cryptographic signature goes here
