import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

try:
    from data_manager import load_data, load_3d_arrays, build_slpt_tensor, build_astro_features, build_time_features
    from differential_geometry import *
except ImportError:
    from .data_manager import load_data
    from .differential_geometry import *
    
# Most data is stored as a configuration space of (records, nodes, space-time dimensions) (wide format)
# For logistic regression, we need to reshape the tensors into a feature matrix (samples, features) (long format)
# Then we can easily drop nan records on missing samples for individual nodes.
# Finally, we can one-hot encode the node indices for binomial or multinomial classification.
# the easiest way to do this is to turn each space-time layer into a dataframe, melt it, and concatenate the layers by node-timestamp
# if a dataset is already in hdf5 we can load the tensor directory and melt it.
# if the dataset has not been processed into hdf5 stored tensor features, then we can build the tensor feature from the raw csv data by calling build_slpt_tensor,
# then construct the desired tensor feature from the Q config space (examples are arc length, velocity, kinetic energy, etc.)

def build_feature_space(s_melt: pd.DataFrame, nodes: list = [], feature_cols: list = [], one_hot_encode: bool = False):
    # drop rows with NaN arc length
    S_melt = s_melt.dropna(subset=feature_cols)
    
    # if no nodes specified, just return all nodes as integer labels
    if len(nodes) == 0 and one_hot_encode is False:
        y = S_melt['Node'].values
        encoder = None
    # one hot encode for binomial targets
    elif len(nodes) == 1:
        print("For binomial regression, please provide two nodes to classify between.")
        return 
    elif len(nodes) == 2 and one_hot_encode:
        S_melt = S_melt[S_melt['Node'].isin(nodes)]    
        encoder = OneHotEncoder(sparse_output=False)
        nodes = S_melt[['Node']]
        encoded_nodes = encoder.fit_transform(nodes)
        y = encoded_nodes[:,0]
        print(f"Encoded nodes shape: {encoded_nodes.shape}")
        print(f"Encoded nodes sample:\n{encoded_nodes[:5,:]}")
    elif len(nodes) >= 2 and one_hot_encode is False:
        S_melt = S_melt[S_melt['Node'].isin(nodes)]
        y = S_melt['Node'].values
        encoder = None
    else:
        print("For multinomial regression, please provide more than two nodes or set one_hot_encode to False.")
        return 
            
    X = S_melt[feature_cols].values
    print(f"Feature X shape: {X.shape}")
    
    print(f"Target y shape: {y.shape}")
    return X, y, encoder

    
# this function expects the input data to be preprocessed into a feature matrix X and a single binomial target vector y
def nodal_logistic_reg(X: np.ndarray, y: np.ndarray):
    # encoder = OneHotEncoder(sparse_output=False) # encoding should be done in preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    # evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training set accuracy: {train_score}")
    print(f"Test set accuracy: {test_score}")
    return model, scaler



##################################################
# Make a simple class in make_plots for
# class LOGREGPLOTS: (or something)
##################################################
def plot_binom_reg(X: np.ndarray, model: LogisticRegression, scaler: StandardScaler,
                node_label: str = "Node", arc_length_label: str = "Arc Length"):
    import matplotlib.pyplot as plt
    # filter X to the columns in nodes list
        
    X_new = np.linspace(np.min(X), np.max(X), 1000).reshape(-1, 1)
    X_new_scaled = scaler.transform(X_new)
    y_proba = model.predict_proba(X_new_scaled)
    print(f"Predicted probabilities shape: {y_proba.shape}")
    decision_boundary_index = np.argmax(y_proba >= 0.5, axis=0)
    decision_boundary = X_new[decision_boundary_index, 0]
    print(f"Decision boundary for {node_label} at {arc_length_label}: {decision_boundary}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(X_new, y_proba[:,1], label=f'Probability of {node_label}')
    plt.axvline(x=decision_boundary[0], color='r', linestyle='--', label='Decision Boundary')
    plt.xlabel(arc_length_label)
    plt.ylabel('Probability')
    plt.title(f'Logistic Regression Probability for {node_label}')
    plt.legend()
    plt.show()
    
    
    
def plot_binom_reg_2d(X: np.ndarray, model: LogisticRegression, scaler: StandardScaler,
                    node_label: str = "Node"):
    # Create mesh grid for both features
    arc_length_range = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 1000)
    tidal_range = np.linspace(0, 12.4, 1000)  # Full tidal cycle
    
    # Create 2D grid
    AL_grid, tidal_grid = np.meshgrid(arc_length_range, tidal_range)
    
    # Flatten and scale for prediction
    X_grid = np.column_stack([AL_grid.ravel(), tidal_grid.ravel()])
    X_grid_scaled = scaler.transform(X_grid)
    
    # Predict probabilities
    y_proba = model.predict_proba(X_grid_scaled)
    
    # Reshape back to grid for plotting
    proba_grid = y_proba[:,1].reshape(AL_grid.shape)
    
    # Create polar/cyclic plot
    plt.figure(figsize=(12, 8))
    contour = plt.contourf(AL_grid, tidal_grid, proba_grid, levels=20, cmap='RdBu_r')
    plt.colorbar(contour, label=f'Probability of {node_label}')
    plt.xlabel('Arc Length')
    plt.ylabel('Tidal Phase (hours)')
    plt.title(f'Probability of {node_label} vs Arc Length and Tidal Phase')
    plt.show()


def plot_polar_probability(X: np.ndarray, model: LogisticRegression, scaler: StandardScaler,
                        node_label: str = "Node"):
    # Create polar grid
    arc_length_range = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 1000)
    print(f"Arc length range: {np.min(arc_length_range)} to {np.max(arc_length_range)}")
    theta_range = np.linspace(0, 2*np.pi, 1000)  # Convert hours to radians
    
    # Create polar mesh grid  
    R_grid, THETA_grid = np.meshgrid(arc_length_range, theta_range)
    
    # Convert back to hours for model input (0-12.4 scale)
    tidal_hours = THETA_grid * (12.4 / (2*np.pi))
    
    # Flatten and scale for prediction
    X_grid = np.column_stack([R_grid.ravel(), tidal_hours.ravel()])
    X_grid_scaled = scaler.transform(X_grid)
    
    # Predict probabilities
    y_proba = model.predict_proba(X_grid_scaled)
    proba_grid = y_proba[:,1].reshape(R_grid.shape)
    
    # Create polar plot
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection='polar')
    contour = ax.contourf(THETA_grid, R_grid, proba_grid, levels=20, cmap='RdBu_r')
    plt.colorbar(contour, label=f'Probability of {node_label}')
    ax.set_title(f'Probability of {node_label} (Radial: Arc Length, Angular: Tidal Phase)')
    
    # Set tidal hour labels instead of radians
    ax.set_xticks(np.linspace(0, 2*np.pi, 13))
    ax.set_xticklabels([f'{h:.1f}h' for h in np.linspace(0, 12.4, 13)])
    
    plt.show()
    
    
def plot_polar_nodal_regions(X: np.ndarray, model: LogisticRegression, scaler: StandardScaler):
    """Plot nodal classification regions in polar coordinates"""
    arc_length_range = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 1000)
    theta_range = np.linspace(0, 2*np.pi, 1000)
    R_grid, THETA_grid = np.meshgrid(arc_length_range, theta_range)
    tidal_hours = THETA_grid * (12.4 / (2*np.pi))
    
    X_grid = np.column_stack([R_grid.ravel(), tidal_hours.ravel()])
    X_grid_scaled = scaler.transform(X_grid)
    
    # Get predicted class for each point
    y_pred_grid = model.predict(X_grid_scaled)
    class_grid = y_pred_grid.reshape(R_grid.shape)
    
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection='polar')
    contour = ax.contourf(THETA_grid, R_grid, class_grid, cmap='tab10', alpha=0.7)
    ax.set_title('Nodal Classification Regions')
    ax.set_xticks(np.linspace(0, 2*np.pi, 13))
    ax.set_xticklabels([f'{h:.1f}h' for h in np.linspace(0, 12.4, 13)])
    plt.show()

def plot_polar_decision_boundaries(X: np.ndarray, model: LogisticRegression, scaler: StandardScaler):
    """Plot decision boundaries based on probability differences"""
    arc_length_range = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 1000)
    theta_range = np.linspace(0, 2*np.pi, 1000)
    R_grid, THETA_grid = np.meshgrid(arc_length_range, theta_range)
    tidal_hours = THETA_grid * (12.4 / (2*np.pi))
    
    X_grid = np.column_stack([R_grid.ravel(), tidal_hours.ravel()])
    X_grid_scaled = scaler.transform(X_grid)
    
    y_proba = model.predict_proba(X_grid_scaled)
    # Difference between highest and second highest probability
    prob_sorted = np.sort(y_proba, axis=1)
    prob_diff = prob_sorted[:,-1] - prob_sorted[:,-2]
    diff_grid = prob_diff.reshape(R_grid.shape)
    
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection='polar')
    ax.contourf(THETA_grid, R_grid, diff_grid, levels=20, cmap='viridis', alpha=0.7)
    # Decision boundaries where confidence is low
    boundary = ax.contour(THETA_grid, R_grid, diff_grid, levels=[0.1, 0.3], colors='red', linestyles='--')
    ax.set_title('Decision Boundaries (Red) and Classification Confidence')
    ax.set_xticks(np.linspace(0, 2*np.pi, 13))
    ax.set_xticklabels([f'{h:.1f}h' for h in np.linspace(0, 12.4, 13)])
    plt.colorbar(boundary, label='Probability Difference (Max - Second)')
    plt.show()

def plot_polar_node_probability_contours(X: np.ndarray, model: LogisticRegression, scaler: StandardScaler, target_node: int):
    """Plot probability contours for a specific node"""
    arc_length_range = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 1000)
    theta_range = np.linspace(0, 2*np.pi, 1000)
    R_grid, THETA_grid = np.meshgrid(arc_length_range, theta_range)
    tidal_hours = THETA_grid * (12.4 / (2*np.pi))
    
    X_grid = np.column_stack([R_grid.ravel(), tidal_hours.ravel()])
    X_grid_scaled = scaler.transform(X_grid)
    
    y_proba = model.predict_proba(X_grid_scaled)
    node_idx = list(model.classes_).index(target_node)
    proba_node = y_proba[:,node_idx].reshape(R_grid.shape)
    
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection='polar')
    contour = ax.contourf(THETA_grid, R_grid, proba_node, levels=20, cmap='RdBu_r', alpha=0.7)
    # High probability contours
    prob_contours = ax.contour(THETA_grid, R_grid, proba_node, levels=[0.5, 0.8, 0.95], 
                            colors='black', linestyles='-', linewidths=1.5)
    ax.clabel(prob_contours, inline=True, fontsize=10)
    ax.set_title(f'Probability Contours for Node {target_node}')
    ax.set_xticks(np.linspace(0, 2*np.pi, 13))
    ax.set_xticklabels([f'{h:.1f}h' for h in np.linspace(0, 12.4, 13)])
    plt.colorbar(contour, label=f'Probability of Node {target_node}')
    plt.show()
    