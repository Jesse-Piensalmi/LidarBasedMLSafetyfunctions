import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.nn.init as init

from collections import deque
from sklearn.kernel_approximation import RBFSampler

PYSIM = True
#PYSIM = False # for experiment or running via ROS

if PYSIM:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

warnings.filterwarnings('ignore')
'''________________________ color map ______________________________________'''
#red to green color map for safe and unsafe areas 
cdict = {'green':  ((0.0, 0.0, 0.0),   # no red at 0
                  (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.8, 0.8)),  # set to 0.8 so its not too bright at 1

        'red': ((0.0, 1, 1),   # set to 0.8 so its not too bright at 0
                  (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.0, 0.0)),  # no green at 1

        'blue':  ((0.0, 0.0, 0.0),   # no blue at 0
                  (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.0, 0.0))   # no blue at 1
       }
RdGn = colors.LinearSegmentedColormap('GnRd', cdict)
'''________________________ functions ______________________________________'''

def kernel(f):      
        return lambda a, b,c,d: np.array(
        [[np.float64(f(a[i], b[j],c[i]-d[j]))  for j in range(b.shape[0])]
        for i in range(a.shape[0])] )

def value_eye_fill(value,shape):
    result = np.zeros(shape)
    np.fill_diagonal(result, value)
    return result

# Calculate the relative error
def relative_error(A,x,B):
    # Check the accuracy of the solution
    # Calculate the residual (error)
    residual = np.dot(A, x) - B 
    re= np.linalg.norm(residual) / np.linalg.norm(B)
    
    if re < 1e-6:
        accurate=True
    else:
        accurate=False
    return accurate

class LearnableSmoothingLayer(nn.Module):
    def __init__(self, kernel_size=3):
        super(LearnableSmoothingLayer, self).__init__()
        self.smoothing_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        # Initialize kernel weights with uniform values for smoothing
        nn.init.constant_(self.smoothing_conv.weight, 1 / kernel_size)

    def forward(self, h_values):
        # Expect input shape to be (batch_size, 1, num_values) for 1D convolution
        if h_values.dim() == 2:
            h_values = h_values.unsqueeze(1)  # Add channel dimension
        smoothed_h = self.smoothing_conv(h_values)
        return smoothed_h.squeeze(1)  # Remove channel dimension after convolution

# Define the neural network
class SafetyNN(nn.Module):
    def __init__(self, input_dim=74, hidden_dim=64, output_dim=4, padding_value=1e6):
        super(SafetyNN, self).__init__()
        self.padding_value = padding_value
        # The input should now have 1 channel (feature map), and each point has 74 features
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)  # 1 channel, 74 length
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        # Replace NaN values with 0
        x = torch.nan_to_num(x, nan=0.0)

        # Create a mask for valid points
        mask = (x != self.padding_value).all(dim=1)  # Mask along the feature dimension
        #x[~mask.unsqueeze(-1).expand_as(x)] = 0.0   # Set invalid points to 0

        # Convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        # Global pooling
        x = self.global_pool(x)  # Shape: [batch_size, hidden_dim, 1]
        x = x.squeeze(-1)        # Remove the last dimension

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # Shape: [batch_size, output_dim]


class SafetyNet():
    def __init__(self, min_d_sample=0.1, iter_mem=50, grid_size_plot=0.1, dh_dt=0.01, smoothing_window = 5):
        self.reset_data()
        self.min_d_sample = min_d_sample
        self.mem_num = iter_mem
        self.grid_size = grid_size_plot
        self.dh_dt = dh_dt
        self.safe_offset = 0.5
        # Initialize the neural network model with appropriate parameters
        input_dim = 73  # Dimensionality of the state space
        hidden_dim = 64
        output_dim = 4
        self.lstm_hidden_dim = 128
        # Define the neural network
        self.model = SafetyNN(input_dim, hidden_dim, output_dim)
        # ADD YOUR OWN FULL PATH TO WEIGHTS in lidar_gp_CBF folder to load weights
        self.model.load_state_dict(torch.load('/home/jesse/LiDARBasedCBFRobo666/lidar_gp_cbf/safety_nn_model_simple.pth'))
        #self.model = torch.load('/home/jesse/LiDARBasedCBFRobo666/lidar_gp_cbf/safety_nn_model.pth')
        # Define the architecture of the neural network with LSTM
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5) 
        self.criterion = nn.MSELoss()
        self.initial_fit = False
        self.__prediction_plot = None
        self.init = False
        self.init_map = True
        self.set = False
        self.batch_size = 360
        self.smoothing_window = smoothing_window
        self.trained = False
        # Initialize deque to store recent safety function values for smoothing
        self.h_values = deque(maxlen=self.smoothing_window)
       
        # Initialize running stats
        self.mean_x, self.mean_y = 0.0, 0.0  # Initially set to some starting position
        self.std_x, self.std_y = 1.0, 1.0  # Some initial values (you can tune this)

        # Learning rate for online updates (tune this)
        self.alpha = 0.1     

        self.x_min = float('inf')
        self.x_max = float('-inf')
        self.y_min = float('inf')
        self.y_max = float('-inf')
       

    def reset_data(self):
        self.data_X = None
        self.data_Y = None
        self.N = 0
        self.k = 0
        self.iter = None
        self.iter_mem = None
        
    def new_iter(self):
        #update iteration number
        self.k+=1
        #go through recorded data and remove data that are older than iteration memory
        if (self.N!=0):
            forgetting_iters=list(range(self.k - self.mem_num, max(1, self.k - 2 * self.mem_num) - 1, -1))
            mask = ~np.isin(self.iter, forgetting_iters)
            # Ensure that the mask size matches with data_X and data_Y size
            if mask.shape[0] == self.data_X.shape[0]:
                # Apply the mask to retain only relevant data
                self.iter = self.iter[mask]
                self.data_X = self.data_X[mask]

            # If after filtering, data becomes empty, reset the dataset
            if len(self.data_X) == 0:
                self.data_X = None
                self.iter = None
                self.N = 0

    # Modify to construct Tensor for training dataset
    def set_new_data(self, new_X, new_Y=np.array([[-3.0]]), dist=np.inf,sense_dist=np.inf,rob_pos=np.array([0,0,0]), safe_offset=0.35):
        """
        Update the SVM model with new sensor data, ensuring a minimum sampling distance.
        
        Parameters:
        new_X : np.array
            The newly sensed position(s) (shape: 1x2 or more).
        new_Y : np.array
            The corresponding label(s), default is unsafe.
        sense_dist : float
            The sensing distance, used to determine if a point is 'inf' or not.
        safe_offset : float
            The offset to generate a safe point from the measured unsafe point.
        """
        
        if self.data_X is None:
            if dist >= sense_dist:
                # Direction towards the point from roboto
                direction_vec = rob_pos[:2] - new_X
                distance = np.linalg.norm(direction_vec)
                direction_vec = direction_vec / distance # Normalize the direction vector

                # No previous data, so initialize with the new data
                self.data_X = new_X
                self.iter = np.array([self.k])     # Initialize iteration tracking array
            else:
                # Direction towards the point from roboto
                direction_vec = rob_pos[:2] - new_X
                distance = np.linalg.norm(direction_vec)
                direction_vec = direction_vec / distance # Normalize the direction vector

                # Depending on label type distance will be calculated differently            
                #label = (distance*4)-1
                self.data_X = new_X
                self.iter = np.array([self.k])     # Initialize iteration tracking array
            self.N = len(self.data_X)        
        # Check the distance of the new data point
        else:
            dis_to_mem = np.linalg.norm(self.data_X[:, 0:2] - new_X[0:2], axis=1)
            #if min(dis_to_mem) > self.min_d_sample:
                # If the distance is larger than sensing distance, it is safe
            if dist >= sense_dist:
                        direction_vec = rob_pos[:2] - new_X
                        distance = np.linalg.norm(direction_vec)
                        direction_vec = direction_vec / distance # Normalize the direction vector
                        
                        self.data_X = np.append(self.data_X, new_X, axis=0)
                        self.iter = np.append(self.iter, self.k)
            else:
                        # Obstacle detected, two points are generated: one unsafe and one safe

                        # Direction towards the point from roboto
                        direction_vec = rob_pos[:2] - new_X
                        distance = np.linalg.norm(direction_vec)
                        direction_vec = direction_vec / distance
                            
                        # 1. Unsafe point (Obstacle detected)
                        self.data_X = np.append(self.data_X, new_X, axis=0)
                        self.iter = np.append(self.iter, self.k)

            self.N = len(self.data_X)
            #else:
                # Update label if a new data point was detected nearby
                #arg = np.argmin(dis_to_mem)
                #self.iter[arg] = self.k

    def get_cbf_safety_prediction(self, t):
        """
        Compute the safety prediction using the SVM, returning both the value and the gradient of h.
        
        Parameters:
        t : np.array
            The input state (position, etc.) where we want to compute the safety prediction.
        
        
        Returns:
        gp_G : np.array
            The gradient of the function h(x).
        gp_h : np.array
            The value of the function h(x) minus dh_dt.
        hgp_xq : np.array
            The value of the function h(x).
        """
        self.model.eval()
        n = t.shape[0]  # Number of input points (rows in t)  
        # Initialize arrays to store h values and gradients
        hsvm_xq = np.zeros((n, 1))  # Array for h values
        svm_h = np.zeros((n, 1))    # Array for h values minus dh_dt
        svm_G = np.zeros((n, 3))  # Array for gradients, with the same dimensionality as t
        
        # Loop over all input points in t
        for i in range(n):
            # Get the actual value of h for the current input t[i]
            probabilities = self.model.forward(t)
            #h_value = 2 * probabilities - 1
            h_value = probabilities[0,0]
            hsvm_xq[i, 0] = h_value.item()  # Store the h value
            
            # Compute the numerical gradient of h with respect to t[i]
            #gradient = self.compute_gradient(torch.FloatTensor(t[i].reshape(1, -1)))
            gradient= np.array([probabilities[0,1:].detach().numpy()]) 
            svm_G[i, :] = gradient # Store the gradient
            
            # Compute the adjusted h value for safety constraints
            svm_h[i, 0] = h_value -0.3
        
        return svm_G, svm_h, hsvm_xq
                

    """................ Mapping the safety prediction..................... """
    
    def __create_mesh_grid(self, field_x, field_y):
        aa=0
        m = int( (field_x[1]+aa - field_x[0]-aa) //self.grid_size ) 
        n = int( (field_y[1]+aa - field_y[0]-aa) //self.grid_size ) 
        gx, gy = np.meshgrid(np.linspace(field_x[0]-aa, field_x[1]+aa, m), np.linspace(field_y[0]-aa, field_y[1]+aa, n))
        return gx.flatten(), gy.flatten()

    def draw_gp_whole_map_prediction(self, ax, field_x, field_y, ic, robot_pos, sensing_rad, color='r'):
        if self.init_map:
            """ initializing the mapping """
            data_point_x, data_point_y = self.__create_mesh_grid(field_x, field_y)
            r_x=data_point_x.shape[0]
            r_y=data_point_y.shape[0]
            self.t_map=np.append(np.reshape(data_point_x,(1,r_x)).T,np.reshape(data_point_y,(1,r_y)).T, axis=1)
            self.init_map=False
            
            # Assign handler, data will be updated later
            self._pl_dataset, = ax.plot(robot_pos[0], robot_pos[1], '.', color=color)

            circle_linspace = np.linspace(0., 2*np.pi, num=360, endpoint=False)
            self.def_circle = np.transpose(np.array([np.cos(circle_linspace), np.sin(circle_linspace)]))
        
        self.h_val_toplot = np.ones(self.t_map.shape[0])
        # Grab the closest data
        is_computed = np.linalg.norm(robot_pos[:2] - self.t_map, axis=1) < sensing_rad*1.5
        map_to_plot = self.t_map[is_computed]
        loca_map_to_plot = map_to_plot-robot_pos[:2]
        """ updating the map """
        #print('iter',self.iter)
        #print('data',self.data_X)
        if self.trained and (self.data_X is not None): # Assign with data
            i = 0
            for datapoint in map_to_plot:
                localized_points = self.data_X-np.array([robot_pos[0],robot_pos[1]])
                padding = np.array([[1.0, 1.0]] * (36 - len(localized_points)))
                # Stack the original detections and the padding
                padded_edges = np.vstack([localized_points, padding])
                data_point = np.vstack([np.array([data_point[0]-robot_pos[0],data_point[1]-robot_pos[1]]),padded_edges]).reshape(-1)
                # Convert to tensor
                data_tensor = torch.tensor(data_point, dtype=torch.float32)

                # Add batch dimension
                data_tensor = data_tensor.unsqueeze(0)  # Shape: [1, 74]

                # Add sequence dimension
                data_tensor = data_tensor.unsqueeze(-1)  # Shape: [1, 74, 1]
                if i ==0:
                    _,_,self.hpg_map=self.get_cbf_safety_prediction(data_tensor)
                else:
                    _,_,cur_hgp=self.get_cbf_safety_prediction(data_tensor)
                    self.hpg_map = np.append(self.hpg_map,cur_hgp)
            self._pl_dataset.set_data(self.data_X[:,0], self.data_X[:,1])
        else: # Reset map + assing current position to plot
            self.hpg_map=np.ones(len(map_to_plot))
            self._pl_dataset.set_data([robot_pos[0]], [robot_pos[1]])    

        self.h_val_toplot[is_computed] = self.hpg_map.T[0]
        # self.hpg_map,_,_,_,_=self.update_gp_computation(self.t_map)
        

        circle_data = np.array([robot_pos[:2]]) + (self.def_circle*sensing_rad)

        if self.__prediction_plot is not None:
            self.__prediction_plot.set_array(self.h_val_toplot)
            # update circle
            self._pl_circle.set_data(circle_data[:,0], circle_data[:,1])
        else:
            self.__prediction_plot = ax.tripcolor(self.t_map[:,0],self.t_map[:,1], self.h_val_toplot,
                    vmin = -3, vmax = 3, shading='gouraud', cmap=RdGn)
            if PYSIM:
                axins1 = inset_axes(ax, width="25%", height="2%", loc='lower right')
                plt.colorbar(self.__prediction_plot, cax=axins1, orientation='horizontal', ticks=[-1,0,1])
                axins1.xaxis.set_ticks_position("top")
                # draw circle first time
                self._pl_circle, = ax.plot(circle_data[:,0], circle_data[:,1], '--', color='gray')
