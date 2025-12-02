import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# let Q be the configuration space for the system of observed particles
def compute_differentials(data: np.ndarray, max_time_delta=2.0):
    """
    Compute arc length differentials along spatial dimensions.
    data should be a 2 or 3d array with shape (states, nodes, dims)
    where
    states: records over time
    nodes: also considered to be tracked points, particles, observers, players, etc. (depends on context)
    dims: space-time dimensions (x1, x2, ..., xt) where xt is time dimension
    Note: for a non-relativistic context, all nodes share the same time dimension and it will be the last dimension. Time will be constant across nodes at each state.
    For relativistic contexts, each node may have its own time measurement.
    The time dimension is assumed to be the last dimension if included.
    Returns:
        dt: time differentials between states, shape (states, nodes)
        dQ: spatial differentials between states, shape (states, nodes, dims-1)
        delQ: spatial variation between nodes, shape (states, nodes, dims-1)
    """
    # Compute differentials along the states (time) dimension
    dt = np.diff(data[:, :, -1], axis=0)  # time differences between states
    dQ = np.diff(data[:, :, :-1], axis=0) # the difference along time at constant node
    delQ = np.diff(data[:, :, :-1], axis=1) # the variation between nodes at constant time
    
    # pad dt to match dQ.shape[0] with zeros at the beginning
    dt = np.vstack([np.zeros((1, data.shape[1])), dt]) + 1e-8  # avoid division by zero
    # pad first record with zeros
    dQ = np.vstack([np.zeros((1, data.shape[1], data.shape[2]-1)), dQ]) # shoreline_evolution or temporal_displacement
    # pad first node with zeros
    delQ = np.hstack([np.zeros((data.shape[0], 1, data.shape[2]-1)), delQ]) # shoreline_geometry or spatial_variation

    # Any calculation that depends on time differentials dt must ensure that dt does not have zero values (This should not be an issue unless records repeat).
    # Any calculation that depends on time differentials dt must ensure that the discrete time_delta values are within a user-defined threshold.
    # This function still needs a control on the distance for a max time_delta to avoid large jumps in position over long time intervals
    dt_mask = dt[:,0] > max_time_delta
    dt[dt_mask,:] = np.nan  # set large time deltas to NaN to avoid erroneous velocity calculations
    
    return dt, dQ, delQ

def compute_arc_lengths(dQ, delQ):
    """Compute arc lengths from spatial differentials."""
    dS2 = np.sum(dQ**2, axis=2)  # squared spatial differences in time (temporal metric component g_tt of the full metric tensor g)
    dS = np.sqrt(dS2 + 1e-8)  # arc length of each particle's movement through space between time steps

    delS2 = np.sum(delQ**2, axis=2) # squared arc length elements between nodes (spatial metric component, g_ss)
    delS = np.sqrt(delS2 + 1e-8)  # actual arc length distance between adjacent shoreline points
    
    return dS, delS


def compute_worldline_arc_length(dS):
    """
    Compute cumulative path length traveled by each particle through space.
    
    Args:
        dS: Arc length of particle motion between time steps, shape (states, nodes)
    
    Returns:
        tau: Worldline arc length coordinate, shape (states, nodes)
    """
    tau = np.nancumsum(dS, axis=0)
    return tau


def compute_fxy(delS):
    X = delS[:,:,0]
    Y = delS[:,:,1]
    fyx = Y / X # fyx = (y_j+1 - y_j) / (x_j+1 - x_j)
    delXdelY = np.sqrt(1 + fyx**2)
    return fyx


def compute_arc_length_coordinate(delS):
    """
    Compute arc length coordinate along shoreline.
    
    Args:
        delS: Arc length elements between nodes, shape (states, nodes)
    
    Returns:
        s: Arc length coordinate (cumulative distance from start), shape (states, nodes)
    """
    # s = np.cumsum(delS, axis=1)
    s = np.nancumsum(delS, axis=1)  # in-place cumulative sum to handle NaNs
    # np.nansum(delS, axis=1, out=s[:,-1])  # ensure last value is total arc length
    return s


def compute_tangent_vectors(delQ, delS):
    """Compute tangent vectors along shoreline from spatial variation delQ and arc lengths delS."""
    T = delQ / delS[:, :, np.newaxis]  # shape (states, nodes, dims-1) [tangent vectors along shoreline] Direction along the shoreline curve (spatial geometry) (shoreline tangent)
    return T 


def compute_worldline_tangents(dQ, dS):
    """Compute worldline tangent vectors from spatial differentials dQ and arc lengths dS."""
    dQdS = dQ / dS[:, :, np.newaxis]  # shape (states, nodes, dims-1) [directional vectors of motion] (worldline tangents)
    return dQdS



def compute_velocity_vectors(dQ, dt, max_time_delta=2.0):
    """Compute velocity vectors from spatial differentials dQ and time differentials dt."""
    Velocity = dQ / dt[:, :, np.newaxis]  # shape (states, nodes, dims-1) Direction of particle motion through space (temporal evolution)
    return Velocity


def compute_speed(Velocity):
    """Compute speed (magnitude of velocity vectors)."""
    speed = np.linalg.norm(Velocity, axis=2)  # shape (states, nodes)
    return speed


def compute_normal_vectors(T):
    """Compute normal vectors to shoreline from tangent vectors T."""
    # assuming 2D space (dims-1 = 2)
    N = np.zeros_like(T)
    N[:,:,0] = -T[:,:,1]
    N[:,:,1] = T[:,:,0]
    return N


def compute_velocity_components(T, N, Velocity):
    """
    Compute tangential and normal velocity vectors.
    
    Args:
        T: Tangent vectors along shoreline, shape (states, nodes, dims-1)
        N: Normal vectors to shoreline, shape (states, nodes, dims-1) 
        Velocity: Velocity vectors, shape (states, nodes, dims-1)
    
    Returns:
        V_t: Tangential velocity vectors = (V·T)T, shape (states, nodes, dims-1)
        V_n: Normal velocity vectors = (V·N)N, shape (states, nodes, dims-1)
        V_t_mag: Magnitude of tangential component = |V·T|, shape (states, nodes)
        V_n_mag: Magnitude of normal component = |V·N|, shape (states, nodes)
    
    Notes:
        - V_t and V_n are vector projections preserving directional information
        - V_t_mag and V_n_mag are scalar magnitudes for energy calculations
        - The tangential component V_t_mag equals the mixed metric component g_st
        - Vector components are useful for force projections, scalars for energy analysis
    """
    V_dot_T = np.sum(Velocity * T, axis=2, keepdims=True)  # shape: (states, nodes, 1)
    V_dot_N = np.sum(Velocity * N, axis=2, keepdims=True)  # shape: (states, nodes, 1)
    
    V_t = V_dot_T * T  # Tangential velocity vectors
    V_n = V_dot_N * N  # Normal velocity vectors
    
    V_t_mag = np.squeeze(V_dot_T)  # |V_t| 
    V_n_mag = np.squeeze(V_dot_N)  # |V_n|
    
    return V_t, V_n, V_t_mag, V_n_mag


# shoreline_strain_rate, spatial_arc_rate
def compute_arc_change_rates(delS, dt):
    """Compute rates of change of arc lengths over time."""
    # delS has shape (states, nodes)
    # dt has shape (states, nodes)  
    # Take finite difference of delS along time axis (axis=0)
    d_delS_dt = np.diff(delS, axis=0) / dt[:-1, :]  # shape: (states-1, nodes)
    # Pad to match original shape
    d_delS_dt = np.vstack([np.zeros((1, delS.shape[1])), d_delS_dt])
    # notationally, this is ∂(delS)/∂t, which is the same as del(∂S/∂t)
    # but is this truly a variation given this are observed shoreline points, can we really play at it being a variation between nodes?
    return d_delS_dt  # This is the true shoreline_strain_rate


def compute_shoreline_strain_acceleration(delS, dt):
    """Compute acceleration of shoreline segment length changes."""
    # First derivative: how segment lengths change
    d_delS_dt = np.diff(delS, axis=0) / dt[:-1, :]  # shape: (states-1, nodes)
    
    # Second derivative: how the rate of change accelerates  
    d2_delS_dt2 = np.diff(d_delS_dt, axis=0) / dt[:-2, :]  # shape: (states-2, nodes)
    
    # Pad both to match original shape
    d_delS_dt = np.vstack([np.zeros((1, delS.shape[1])), d_delS_dt])
    d2_delS_dt2 = np.vstack([np.zeros((2, delS.shape[1])), d2_delS_dt2])
    
    return d_delS_dt, d2_delS_dt2


def compute_velocity_gradient_tensor(Velocity, delS, dt):
    """
    Compute the velocity gradient tensor ∇v and its decomposition.
    
    The velocity gradient tensor ∇v describes how the velocity field varies 
    spatially and can be decomposed into strain rate (deformation) and 
    vorticity (rotation) components.
    
    Args:
        Velocity: Velocity vector field, shape (states, nodes, dims-1)
        delS: Arc lengths between nodes along shoreline, shape (states, nodes)  
        dt: Time differentials, shape (states, nodes)
    
    Returns:
        grad_v: Velocity gradient tensor ∇v, shape (states, nodes, dims-1, dims-1)
        strain_rate: Symmetric strain rate tensor, shape (states, nodes, dims-1, dims-1)
        vorticity: Antisymmetric vorticity tensor, shape (states, nodes, dims-1, dims-1)
        vorticity_vector: Vorticity vector (in 3D), shape (states, nodes, dims-1)
    
    Notes:
        - ∇v = ∂v_i/∂x_j describes how velocity changes in space
        - Strain rate tensor = ½(∇v + ∇vᵀ) represents deformation
        - Vorticity tensor = ½(∇v - ∇vᵀ) represents rotation
        - For 2D, vorticity vector has one non-zero component (out-of-plane)
    """
    # Compute spatial derivatives of velocity along shoreline (s-direction)
    dV_ds = np.diff(Velocity, axis=1) / delS[:, 1:, np.newaxis]  # ∂v/∂s
    dV_ds = np.pad(dV_ds, ((0,0), (1,0), (0,0)), mode='edge')
    
    # Compute time derivatives of velocity  
    dV_dt = np.diff(Velocity, axis=0) / dt[:-1, :, np.newaxis]  # ∂v/∂t
    dV_dt = np.pad(dV_dt, ((1,0), (0,0), (0,0)), mode='edge')
    
    # Build velocity gradient tensor (for 2D: [∂v_x/∂s, ∂v_x/∂t; ∂v_y/∂s, ∂v_y/∂t])
    grad_v = np.zeros((Velocity.shape[0], Velocity.shape[1], 2, 2))
    grad_v[:,:,:,0] = dV_ds  # ∂v/∂s components
    grad_v[:,:,:,1] = dV_dt  # ∂v/∂t components
    
    # Decompose into strain rate and vorticity
    strain_rate = 0.5 * (grad_v + np.transpose(grad_v, (0,1,3,2)))  # ½(∇v + ∇vᵀ)
    vorticity = 0.5 * (grad_v - np.transpose(grad_v, (0,1,3,2)))    # ½(∇v - ∇vᵀ)
    
    # Vorticity vector (scalar in 2D, stored as vector for consistency)
    vorticity_vector = np.zeros_like(Velocity)
    vorticity_vector[:,:,0] = vorticity[:,:,1,0] - vorticity[:,:,0,1]  # ω_z
    
    return grad_v, strain_rate, vorticity, vorticity_vector


def compute_curvature(T, delS, s):
    """
    Compute curvature κ = |dT/dS| along the shoreline.
    
    Curvature measures how rapidly the shoreline tangent vector changes 
    with respect to arc length, representing the bending of the curve.
    
    Args:
        T: Tangent vectors along shoreline, shape (states, nodes, dims-1)
        delS: Arc length elements between nodes, shape (states, nodes)
        s: Cumulative arc length from start, shape (states, nodes)
    
    Returns:
        kappa: Curvature values, shape (states, nodes)
        dT_dS: Curvature vectors (derivative of T), shape (states, nodes, dims-1)
    
    Notes:
        - κ = |dT/dS| measures the rate of change of tangent direction
        - High curvature = sharp bends in shoreline
        - Low curvature = straight shoreline segments
        - Uses central differences for interior points, forward/backward for boundaries
    """
    kappa = np.zeros_like(delS)
    dT_dS = np.zeros_like(T)
    
    for i in range(1, T.shape[1]-1):  # Interior points
        dT_dS[:, i] = (T[:, i+1] - T[:, i-1]) / (s[:, i+1] - s[:, i-1])[:, np.newaxis]
        kappa[:, i] = np.linalg.norm(dT_dS[:, i], axis=1)
    
    # Boundaries
    kappa[:, 0] = np.linalg.norm(T[:, 1] - T[:, 0], axis=1) / (s[:, 1] - s[:, 0])
    kappa[:, -1] = np.linalg.norm(T[:, -1] - T[:, -2], axis=1) / (s[:, -1] - s[:, -2])
    
    # Boundary dT_dS approximations
    dT_dS[:, 0] = (T[:, 1] - T[:, 0]) / (s[:, 1] - s[:, 0])[:, np.newaxis]
    dT_dS[:, -1] = (T[:, -1] - T[:, -2]) / (s[:, -1] - s[:, -2])[:, np.newaxis]
    
    return kappa, dT_dS


def compute_worldline_curvature(dQdS, tau):
    """
    Compute curvature of particle worldlines through space.
    
    Args:
        dQdS: Worldline tangent vectors (dQ/dS), shape (states, nodes, dims-1)  
        tau: Worldline arc length coordinate, shape (states, nodes)
    
    Returns:
        kappa_tau: Worldline curvature, shape (states, nodes)
        dT_dtau: Curvature vectors of worldlines, shape (states, nodes, dims-1)
    """
    kappa_tau = np.zeros_like(tau)
    dT_dtau = np.zeros_like(dQdS)
    
    # Interior time steps (central differences)
    for i in range(1, dQdS.shape[0]-1):
        dT_dtau[i] = (dQdS[i+1] - dQdS[i-1]) / (tau[i+1] - tau[i-1])[:, np.newaxis]
        kappa_tau[i] = np.linalg.norm(dT_dtau[i], axis=1)
    
    # Boundaries (forward/backward differences)
    kappa_tau[0] = np.linalg.norm(dQdS[1] - dQdS[0], axis=1) / (tau[1] - tau[0])
    kappa_tau[-1] = np.linalg.norm(dQdS[-1] - dQdS[-2], axis=1) / (tau[-1] - tau[-2])
    
    dT_dtau[0] = (dQdS[1] - dQdS[0]) / (tau[1] - tau[0])[:, np.newaxis]
    dT_dtau[-1] = (dQdS[-1] - dQdS[-2]) / (tau[-1] - tau[-2])[:, np.newaxis]
    
    return kappa_tau, dT_dtau


def compute_tangential_acceleration(V_t_mag, dt):
    """
    Compute tangential acceleration a_T = dv/dt along the shoreline.
    
    Args:
        V_t_mag: Tangential speed |V_t|, shape (states, nodes)
        dt: Time differentials, shape (states, nodes)
    
    Returns:
        a_T: Tangential acceleration, shape (states, nodes)
    """
    a_T = np.diff(V_t_mag, axis=0) / dt[:-1, :]  # dv/dt
    a_T = np.pad(a_T, ((1, 0), (0, 0)), mode='edge')  # Pad first timestep
    return a_T


def compute_normal_acceleration(kappa, V_t_mag):
    """
    Compute normal acceleration a_N = κv² from curvature and speed.
    
    Args:
        kappa: Curvature values, shape (states, nodes)
        V_t_mag: Tangential speed |V_t|, shape (states, nodes)
    
    Returns:
        a_N: Normal acceleration, shape (states, nodes)
    """
    a_N = kappa * V_t_mag**2  # κv²
    return a_N


def compute_geometric_acceleration(a_T, a_N, T, N):
    """
    Reconstruct acceleration vector from tangential and normal components.
    
    Args:
        a_T: Tangential acceleration, shape (states, nodes)
        a_N: Normal acceleration, shape (states, nodes)  
        T: Tangent vectors, shape (states, nodes, dims-1)
        N: Normal vectors, shape (states, nodes, dims-1)
    
    Returns:
        A_geometric: Reconstructed acceleration vectors, shape (states, nodes, dims-1)
        a_mag: Magnitude of geometric acceleration, shape (states, nodes)
    """
    A_geometric = (a_T[:, :, np.newaxis] * T) + (a_N[:, :, np.newaxis] * N)
    a_mag = np.sqrt(a_T**2 + a_N**2)
    return A_geometric, a_mag


def compute_jacobian(T, Velocity):
    """Compute Jacobian matrix from tangent vectors T and Velocity."""
    # the inputs T, Vel of the Jacobian makes the tangent bundle of the shoreline worldsheet
    J = np.zeros((T.shape[0], T.shape[1], 2, 2))
    J[:,:,:,0] = T  # ∂γ/∂s = shoreline tangent (spatial derivative)
    J[:,:,:,1] = Velocity  # ∂γ/∂t = velocity (temporal derivative)
    return J


def compute_covariant_basis(T, Velocity):
    """Compute covariant basis vectors from tangent vectors T and Velocity."""
    covariant_s = T  # ∂γ/∂s 
    covariant_t = Velocity  # ∂γ/∂t
    return covariant_s, covariant_t


def compute_covariant_basis(J):
    """Compute covariant basis vectors from Jacobian matrix J."""
    covariant_s = J[:,:,:,0]  # ∂γ/∂s but this is just T
    covariant_t = J[:,:,:,1]  # ∂γ/∂t but this is just Velocity
    return covariant_s, covariant_t


def compute_inverse_jacobian(J):
    """Compute inverse Jacobian matrix."""
    det_J = np.linalg.det(J)
    valid_mask = np.abs(det_J) > 1e-6
    J_inv = np.zeros_like(J)
    J_inv[valid_mask] = np.linalg.inv(J[valid_mask])
    return J_inv


def compute_contravariant_basis(J_inv):
    """Compute contravariant basis vectors from inverse Jacobian matrix J_inv."""
    contravariant_s = J_inv[:,:,:,0]  # ∇s (normal to constant-time curves)
    contravariant_t = J_inv[:,:,:,1]  # ∇t (normal to constant-arc-length curves)
    return contravariant_s, contravariant_t


def compute_acceleration(Velocity, dt):
    """Compute acceleration vectors from Velocity and time differentials dt."""
    Acceleration = np.diff(Velocity, axis=0) / dt[1:,:,np.newaxis]  # shape (states-1, nodes, dims-1)
    # pad first record with zeros
    Acceleration = np.vstack([np.zeros((1, Velocity.shape[1], Velocity.shape[2])), Acceleration])
    return Acceleration


def compute_mixed_partial_derivatives(Velocity, delS):
    """
    Compute mixed partial derivatives ∂²/∂s∂t of position.
    
    Measures how velocity patterns vary along the shoreline.
    
    Args:
        Velocity: Velocity vectors, shape (states, nodes, dims-1)
        delS: Arc length elements, shape (states, nodes)
    
    Returns:
        d2x_dsdt: ∂²x/∂s∂t, shape (states, nodes)
        d2y_dsdt: ∂²y/∂s∂t, shape (states, nodes) 
        mixed_partials: Mixed partial vectors, shape (states, nodes, dims-1)
    """
    # Use finite differences along s-axis for velocity components
    dVx_ds = np.diff(Velocity[:,:,0], axis=1) / delS[:,1:,np.newaxis]
    dVy_ds = np.diff(Velocity[:,:,1], axis=1) / delS[:,1:,np.newaxis]
    
    # Pad to original shape
    dVx_ds = np.pad(dVx_ds, ((0,0), (1,0), (0,0)), mode='edge')
    dVy_ds = np.pad(dVy_ds, ((0,0), (1,0), (0,0)), mode='edge')
    
    mixed_partials = np.stack([dVx_ds, dVy_ds], axis=2)
    return dVx_ds, dVy_ds, mixed_partials



def compute_hessian(d2gamma_ds2, Acceleration, mixed_partials):
    """
    Compute Hessian tensor of the shoreline spacetime surface.
    
    The Hessian contains all second derivatives of position with respect to (s,t).
    
    Args:
        d2gamma_ds2: ∂²γ/∂s² (curvature vectors), shape (states, nodes, dims-1)
        Acceleration: ∂²γ/∂t², shape (states, nodes, dims-1) 
        mixed_partials: ∂²γ/∂s∂t, shape (states, nodes, dims-1)
    
    Returns:
        H: Hessian tensor, shape (states, nodes, dims-1, 2, 2)
            H[...,:,0,0] = ∂²γ/∂s²
            H[...,:,1,1] = ∂²γ/∂t²
            H[..., :,0,1] = H[..., :,1,0] = ∂²γ/∂s∂t
    """
    H = np.zeros((*d2gamma_ds2.shape[:2], d2gamma_ds2.shape[2], 2, 2))
    
    H[..., 0, 0] = d2gamma_ds2        # ∂²γ/∂s²
    H[..., 1, 1] = Acceleration       # ∂²γ/∂t²  
    H[..., 0, 1] = mixed_partials     # ∂²γ/∂s∂t
    H[..., 1, 0] = mixed_partials     # Symmetric
    
    return H


def compute_metrics(T, Velocity):
    """Compute metric tensor components from tangent vectors T and Velocity."""
    g_ss = np.sum(T * T, axis=2)  # spatial metric component
    g_tt = np.sum(Velocity * Velocity, axis=2)  # temporal metric component
    g_st = np.sum(T * Velocity, axis=2)  # mixed metric component: coupling T and Vel gives another component of the metric tensor, g_st = g_ts
    return g_ss, g_tt, g_st


def compute_metric_tensor(T, Velocity):
    """Build full metric tensor from tangent vectors T and Velocity. (First Fundamental Form)"""
    g = np.zeros((T.shape[0], T.shape[1], 2, 2))
    g[:,:,0,0] = np.sum(T * T, axis=2)  # g_ss
    g[:,:,1,1] = np.sum(Velocity * Velocity, axis=2)  # g_tt
    g[:,:,0,1] = np.sum(T * Velocity, axis=2)  # g_st
    g[:,:,1,0] = g[:,:,0,1]  # symmetric
    return g


def build_metric_tensor(J):
    # metric tensor
    g = np.einsum('...ji,...jk->...ik', J, J)
    return g


def compute_inverse_metric_tensor(g):
    """Compute inverse metric tensor from metric tensor g."""
    g_inv = np.zeros_like(g)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            det_g = np.linalg.det(g[i,j])
            valid_mask = np.abs(det_g) > 1e-6
            if valid_mask:
                g_inv[i,j] = np.linalg.inv(g[i,j])
    return g_inv


def compute_metric_eigvals(g):
    """Compute eigenvalues and eigenvectors of the metric tensor g."""
    eigvals = np.linalg.eigvals(g)
    eigvecs = np.linalg.eig(g)
    stretch_alongshoreline = np.sqrt(eigvals[:,:,0]) # principal stretch along shoreline (s-direction)
    stretch_along_worldline = np.sqrt(eigvals[:,:,1]) # principal stretch along worldline (t-direction)
    return eigvals, eigvecs


def compute_anistropy_ratio(g):
    """Compute anisotropy ratio from metric tensor g."""
    eigvals, _ = compute_metric_eigvals(g)
    ratio = np.sqrt(eigvals[:,:,0] / (eigvals[:,:,1] + 1e-8))  # avoid division by zero
    return ratio


def compute_metric_eigvals(g):
    """Compute eigenvalues and eigenvectors of the metric tensor g."""
    eigvals = np.zeros((g.shape[0], g.shape[1], 2))
    eigvecs = np.zeros((g.shape[0], g.shape[1], 2, 2))
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            vals, vecs = np.linalg.eig(g[i,j])
            eigvals[i,j] = vals
            eigvecs[i,j] = vecs
    return eigvals, eigvecs
