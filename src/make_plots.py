import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Union

from src.differential_geometry import *
from src.data_manager import *

# Reminder of Q structure for Shoreline data:
# Q: 3D array (time-state, nodes, spatial dimensions)
# X = Q[:,:,0]  # x-coordinates
# Y = Q[:,:,1]  # y-coordinates
# t = Q[:,:,2]  # parameterized time coordinate for a given record/particle
# stop calling it a transect

def plot_state_points(Q):
    """Plot shoreline data from Q array."""
    n_nodes = Q.shape[1]
    # each node gets a color based on its index
    colors = plt.cm.viridis(np.linspace(0, 1, n_nodes))
    plt.figure(figsize=(10, 6))
    for i in range(Q.shape[0]):
        # plt.plot(Q[i,:,0], Q[i,:,1], label=f'Time {i}')
        # plt.plot(Q[i,:,0], -1*Q[i,:,1], label=f'Time {i}')
        plt.scatter(Q[i,:,0], Q[i,:,1], s=10, color=colors)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    # invert y-axis if needed
    plt.gca().invert_yaxis()
    plt.title('Shoreline State Point Time Evolution')
    # plt.legend()
    plt.show()
    
    
def plot_state_curves(Q):
    """Plot shoreline curves from Q array."""
    m_nodes = Q.shape[0]
    colors = plt.cm.viridis(np.linspace(0, 1, m_nodes))
    plt.figure(figsize=(10, 6))
    for i in range(Q.shape[0]):
        plt.plot(Q[i,:,0], Q[i,:,1], color=colors[i], alpha=0.7)
        # plt.plot(Q[i,:,0], -1*Q[i,:,1], label=f'Time {i}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    # invert y-axis if needed
    plt.gca().invert_yaxis()
    plt.title('Shoreline State Curve Time Evolution')
    # plt.legend()
    plt.show()
    
    
# plot vector field of dQ and delQ on shoreline states
def plot_differentials(Q, dQ, delQ, step=5):
    """Plot vector field of dQ and delQ on shoreline states."""
    plt.figure(figsize=(10, 6))
    plt.plot(Q[:,:,0], Q[:,:,1], 'k-', alpha=0.5)  # Plot shoreline states
    for i in range(0, Q.shape[0], step):
            plt.quiver(Q[i,:,0], Q[i,:,1], dQ[i,:,0], dQ[i,:,1], color='r', scale=500, width=0.003)
            plt.quiver(Q[i,:,0], Q[i,:,1], delQ[i,:,0], delQ[i,:,1], color='b', scale=500, width=0.003)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.gca().invert_yaxis()
    plt.title('Vector Field of dQ (red) and delQ (blue)')
    plt.show()
    
    
def plot_arc_lengths(tau, s):
    """Plot arc lengths tau and s over time."""
    # two subplots
    plt.figure(figsize=(12, 6))
    # plot paths along columns because these are the worldlines for each particle across time
    plt.subplot(1, 2, 1)
    plt.plot(tau, 'r-', label='tau (Worldline Arc Length)', alpha=0.7)
    plt.xlabel('Time Index')
    plt.ylabel('Arc Length')
    plt.title('Temporal Arc Length over Time')
    # plt.legend()

    # plot paths along rows because each row is a particle over time and they are connected for a given record
    # plot s versus column index for each row
    plt.subplot(1, 2, 2)
    plt.plot(s.T, 'b-', alpha=0.7)
    plt.xlabel('Spatial Node Index')
    plt.ylabel('Arc Length Coordinate s')
    plt.title('Spatial Arc Length Coordinate s')
    plt.tight_layout()
    plt.show()
    
    
def plot_arc_length_tidal_phase_space(tidal_time, s):
    "plot periodic tidal phase space using arc lengths for nodes as their radial distance with time as the polar angle"
    plt.figure(figsize=(10, 6))
    # plot for each node its tidal time vs arc length s
    n_nodes = s.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_nodes))
    for i in range(n_nodes):
        plt.scatter(tidal_time, s[:,i], s=10, color=colors[i], alpha=0.7)
    plt.xlabel('Tidal Time (hours)')
    plt.ylabel('Spatial Arc Length Coordinate s')
    plt.title('Tidal Phase Space: s vs Tidal Time')
    plt.show()


# You can also keep the original polar plot function and add this new 3D version
def plot_arc_length_tidal_phase_space_polar(tidal_time, s, tidal_period=12.4):
    """
    Plot periodic tidal phase space in polar coordinates using arc lengths 
    for nodes as their radial distance with tidal time as the polar angle.
    
    Parameters:
    tidal_time: array of tidal times (hours)
    s: array of arc lengths for each node
    tidal_period: tidal period in hours (default 12.4 for M2 tide)
    """
    plt.figure(figsize=(10, 10))
    
    # Convert tidal time to radians (angle)
    theta = 2 * np.pi * (tidal_time % tidal_period) / tidal_period
    
    # Arc length s becomes the radial coordinate
    r = s
    
    n_nodes = s.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_nodes))
    
    # Create polar plot
    ax = plt.subplot(111, projection='polar')
    
    for i in range(n_nodes):
        ax.scatter(theta, r[:, i], s=10, color=colors[i], alpha=0.7, label=f'Node {i+1}')
    
    # Customize the polar plot
    ax.set_theta_zero_location('N')  # 0° at the top
    ax.set_theta_direction(-1)       # clockwise direction (typical for tidal phases)
    
    # Set radial labels to show arc length
    ax.set_ylabel('Arc Length s')
    
    # Customize angular ticks to show tidal hours
    tidal_ticks = np.linspace(0, 2*np.pi, 13)  # 12 ticks for hours
    tidal_labels = [f'{int(i*tidal_period/12)}h' for i in range(12)] + ['']
    ax.set_xticks(tidal_ticks)
    ax.set_xticklabels(tidal_labels)
    
    plt.title('Tidal Phase Space: Polar Plot (Angle = Tidal Time, Radius = Arc Length)')
    
    # Add legend if not too many nodes
    if n_nodes <= 10:
        plt.legend(bbox_to_anchor=(1.1, 1.0), loc='upper left')
    
    plt.tight_layout()
    plt.show()


def plot_arc_length_tidal_phase_space_helix(tidal_time, s, linear_time=None, tidal_period=12.4, world_lines=False):
    """
    Plot tidal phase space as a helix in 3D, where:
    - x, y: polar coordinates (tidal time converted to angle, arc length as radius)
    - z: linear time evolution
    
    Parameters:
    tidal_time: array of tidal times (hours)
    s: array of arc lengths for each node
    linear_time: array of linear times for the third dimension (if None, uses index)
    tidal_period: tidal period in hours (default 12.4 for M2 tide)
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert tidal time to radians (angle)
    theta = 2 * np.pi * (tidal_time % tidal_period) / tidal_period
    
    # Arc length s becomes the radial coordinate in x-y plane
    r = s
    
    # Use linear_time for z-axis if provided, otherwise use index
    if linear_time is None:
        z = np.arange(len(tidal_time))
    else:
        z = linear_time
    
    n_nodes = s.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_nodes))
    
    # Convert polar to Cartesian coordinates for 3D plotting
    x = r * np.cos(theta)[:, np.newaxis]
    y = r * np.sin(theta)[:, np.newaxis]
    
    for i in range(n_nodes):
        # Plot the helix lines
        if world_lines:
            ax.plot(x[:, i], y[:, i], z, color=colors[i], alpha=0.5, label=f'Node {i+1} Worldline')
        # Plot scatter points
        ax.scatter(x[:, i], y[:, i], z, s=10, color=colors[i], alpha=0.7, label=f'Node {i+1}')
    
    # Customize the plot
    ax.set_xlabel('X (cos component)')
    ax.set_ylabel('Y (sin component)')
    ax.set_zlabel('Linear Time')
    
    # Set equal aspect ratio for x and y axes to maintain circular shape
    # FIXED: Remove max() call since r.max() returns a single value
    max_range = r.max() if r.size > 0 else 1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    
    plt.title('Tidal Phase Space: Helix Plot\n(X,Y = Polar Coords, Z = Linear Time)')
    
    # Add legend if not too many nodes
    if n_nodes <= 10:
        ax.legend(bbox_to_anchor=(1.1, 1.0), loc='upper left')
    
    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    plt.show()


def plot_tangent_field(Q, T, step=5):
    """Plot tangent vector field T."""
    plt.figure(figsize=(10, 6))
    plt.plot(Q[:,:,0], Q[:,:,1], 'k-', alpha=0.5)  # Plot shoreline states
    for i in range(0, T.shape[0], step):
        plt.quiver(Q[i,:,0], Q[i,:,1], T[i,:,0], T[i,:,1], color='g', scale=50, width=0.003)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.gca().invert_yaxis()
    plt.title('Tangent Vector Field T')
    plt.show()
    

def plot_normal_field(Q, N, step=5):
    """Plot normal vector field N."""
    plt.figure(figsize=(10, 6))
    plt.plot(Q[:,:,0], Q[:,:,1], 'k-', alpha=0.5)  # Plot shoreline states
    for i in range(0, N.shape[0], step):
        plt.quiver(Q[i,:,0], Q[i,:,1], N[i,:,0], N[i,:,1], color='b', scale=50, width=0.003)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.gca().invert_yaxis()
    plt.title('Normal Vector Field N')
    plt.show()
    
    
def plot_velocity_field(Q, V, step=5):
    """Plot velocity vector field V."""
    plt.figure(figsize=(10, 6))
    plt.plot(Q[:,:,0], Q[:,:,1], 'k-', alpha=0.5)  # Plot shoreline states
    for i in range(0, V.shape[0], step):
        plt.quiver(Q[i,:,0], Q[i,:,1], V[i,:,0], V[i,:,1], color='m', scale=500, width=0.003)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.gca().invert_yaxis()
    plt.title('Velocity Vector Field V')
    plt.show()
    
    
def plot_speed_timeseries(t, speed, nodes=range(10, 20, 2)):
    """Plot speed time series for each particle."""
    print(f"Shape of t: {t.shape}, Shape of speed: {speed.shape}")
    print(f"Last few time values: {t[-5:,0]}")
    plt.figure(figsize=(10, 6))
    for i in nodes:
        plt.scatter(t[:,0], speed[:,i], s=10, alpha=0.7)
    plt.xlabel('Time Index')
    plt.ylabel('Speed')
    plt.title('Speed Time Series for Each Particle')
    plt.show()


def plot_statepoint_timeseries(t, Q, nodes=range(10, 20, 2)):
    """Plot state point time series for each particle."""
    plt.figure(figsize=(10, 6))
    for node in nodes:
        plt.scatter(t[:,node], Q[:,node,1], s=10, alpha=0.7)
    plt.xlabel('Time (hours)')
    plt.ylabel('X Coordinate')
    plt.gca().invert_yaxis()
    plt.title('State Point Time Series for Each Particle')
    plt.show()
    
    
def plot_worldline_tangents(Q, dQdS, step=5):
    """Plot worldline tangent vectors dQ/dS."""
    plt.figure(figsize=(10, 6))
    plt.plot(Q[:,:,0], Q[:,:,1], 'k-', alpha=0.5)  # Plot shoreline states
    for i in range(0, dQdS.shape[0], step):
        plt.quiver(Q[i,:,0], Q[i,:,1], dQdS[i,:,0], dQdS[i,:,1], color='c', scale=50, width=0.003)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.gca().invert_yaxis()
    plt.title('Worldline Tangent Vectors dQ/dS')
    plt.show()



def add_arc_lengths(fig, axes, tau, s, subplot_indices: Tuple[int, int]):
    """Add arc length plots to specified subplot position."""
    row, col = subplot_indices
    
    # Create subplot if axes is the figure object, or use existing axes grid
    if isinstance(axes, plt.Axes):
        ax = axes
    else:
        ax = axes[row, col]
    
    ax.plot(tau, 'r-', alpha=0.7)
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Arc Length')
    ax.set_title('Temporal Arc Length over Time')


def add_tangent_field(fig, axes, Q, T, subplot_indices: Tuple[int, int], step=5):
    """Add tangent field to specified subplot position."""
    row, col = subplot_indices
    
    if isinstance(axes, plt.Axes):
        ax = axes
    else:
        ax = axes[row, col]
    
    # ax.plot(Q[:, :, 0], Q[:, :, 1], 'k-', alpha=0.5)
    for i in range(0, T.shape[0], step):
        ax.quiver(Q[i, :, 0], Q[i, :, 1], T[i, :, 0], T[i, :, 1],
                color='g', scale=50, width=0.003)
    ax.invert_yaxis()
    ax.set_title('Tangent Vector Field')