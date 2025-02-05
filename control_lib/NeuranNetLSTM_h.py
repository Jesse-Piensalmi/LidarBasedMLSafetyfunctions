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

    

'''________________________ GP Reg _________________________________________'''

class SafetyNet():
    def __init__(self, min_d_sample=0.1, iter_mem=50, grid_size_plot=0.1, dh_dt=0.01, smoothing_window = 5):
        self.reset_data()
        self.min_d_sample = min_d_sample
        self.mem_num = iter_mem
        self.grid_size = grid_size_plot
        self.dh_dt = dh_dt
        self.safe_offset = 0.5
        # Initialize the neural network model with appropriate parameters
        input_dim = 2  # Dimensionality of the state space
        hidden_dim1 = 64
        hidden_dim2 = 32
        self.lstm_hidden_dim = 128
        super(SafetyNet, self).__init__()

        # Define the architecture of the neural network with LSTM
        self.model = self.construct_net(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2)
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

        


    def construct_net(self, input_dim, hidden_dim1, hidden_dim2, drop_prob = 0.2):
        """
        Function to construct the neural network architecture.
        Returns:
            model: A PyTorch model representing the network.
        """
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            #nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            # Latent layers to extract meaningful stuff from data
            nn.Linear(hidden_dim1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),

        )

        # Initialize the LSTM separately and the final output layer
        #self.lstm = nn.LSTM(input_size=self.lstm_hidden_dim, hidden_size=self.lstm_hidden_dim, batch_first=True)
        #self.output_layer = nn.Sequential(
        #   nn.Linear(self.lstm_hidden_dim, 1),  # Single output for safety value (h)
        #    nn.Tanh()
        #)

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                # Use Xavier initialization for weights
                init.xavier_uniform_(layer.weight)
                # Initialize biases to zero
                if layer.bias is not None:
                    init.zeros_(layer.bias)
        return model
    
    def forward(self, x):
        """
        Forward pass for the model.
        """
        x = self.model(x)
        return x
    
    def train_online(self, dataloader, epochs=10, max_grad_norm = 1, L = 0.01):
        """
        Perform online training using mini-batch SGD.
        Input: dataloader - DataLoader object containing training data
        """
        self.model.train()
        #self.lstm.train()
        #self.output_layer.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for states_batch, safety_values_batch in dataloader:
                self.optimizer.zero_grad()
                states_batch.requires_grad = True

                
                # Forward pass
                predictions = self.forward(states_batch)
                gradients = torch.autograd.grad(predictions.sum(), states_batch, create_graph=True, retain_graph= True)[0]
                smoothness_loss = smoothness_loss = torch.mean((torch.sum(gradients**2, dim=1))) / torch.mean(torch.abs(predictions))
                # Compute main loss
                # Compute class weights
                positive_weight = len(safety_values_batch) / (2 * torch.sum(safety_values_batch == 1))
                negative_weight = len(safety_values_batch) / (2 * torch.sum(safety_values_batch == -1))

                # Apply weights to the loss
                weights = torch.where(safety_values_batch > 0, positive_weight, negative_weight)
                weighted_loss = torch.mean(weights * (predictions - safety_values_batch) ** 2)
                #loss = self.criterion(predictions,safety_values_batch)
                total_loss = weighted_loss+smoothness_loss*L 

                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                nn_utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                # Optimization step
                self.optimizer.step()
                running_loss += total_loss.item()
            avg_loss = running_loss / len(dataloader)
            print("Avergae Loss: " + str(avg_loss))
        self.model.eval()
    
    def compute_gradient(self, state, aggregation_method = 'sum'):
        """
        Compute the gradient of the safety function w.r.t. the input state.
        Input: state - The state for which to compute the gradient
        Output: gradient - Gradient of the safety function (dx, dy, ...)
        """
        self.model.eval()  # Set model to evaluation mode
        #gradients = np.zeros((self.data_X.shape[0], self.data_X.shape[1])) 
        state_tensor = torch.FloatTensor(state)
        state_tensor.requires_grad = True
        # Forward pass
        predicted_safety_value = self.forward(state_tensor)
            
        gradient_vector = torch.ones_like(predicted_safety_value)
    
        # Backward pass to compute the gradient
        predicted_safety_value.backward(gradient=gradient_vector)
        
        # Extract and return the computed gradients
        gradients = state_tensor.grad.detach().numpy()
        # Normalize the aggregated gradient for consistent direction
        #aggregated_gradient /= (np.linalg.norm(aggregated_gradient) + 1e-8)
        
        return -gradients


    def data_and_train(self):
        #Normalize the data
        x_min = -0.5
        x_max = 0.5
        y_min = -0.5
        y_max = 0.5
        normalized_coordinates = self.data_X.copy() 
        normalized_coordinates[:, 0]  = (self.data_X[:, 0] - x_min) / (x_max-x_min)
        normalized_coordinates[:, 1]  = (self.data_X[:, 1] - y_min) / (y_max-y_min)
        reshaped_data = normalized_coordinates[:, :2]
        #dataloader = DataLoader(TensorDataset(torch.FloatTensor(normalized_data_array), torch.FloatTensor(self.data_Y)), batch_size=len(self.data_X), shuffle=True)
        dataloader = DataLoader(TensorDataset(torch.FloatTensor(self.data_X), torch.FloatTensor(self.data_Y)), batch_size=len(self.data_X), shuffle=True)
        # Simulate 10 epochs for each batch size
        loss = self.train_online(dataloader)
        #print(f"Batch size: {self.data_X}, Epoch: {epoch+1}, Loss: {loss}")
        self.trained = True
        if len(self.data_X)<=0:
            self.trained= False
    
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
            if mask.shape[0] == self.data_X.shape[0] and mask.shape[0] == self.data_Y.shape[0]:
                # Apply the mask to retain only relevant data
                self.iter = self.iter[mask]
                self.data_Y = self.data_Y[mask]
                self.data_X = self.data_X[mask]

            # If after filtering, data becomes empty, reset the dataset
            if len(self.data_X) == 0:
                self.data_X = None
                self.data_Y = None
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

                pos_X = new_X + direction_vec*0.1
                # No previous data, so initialize with the new data
                self.data_X = pos_X
                self.data_Y = [[1]]  # Initialize empty array for Y
                self.iter = np.array([self.k])     # Initialize iteration tracking array
            else:
                # Direction towards the point from roboto
                direction_vec = rob_pos[:2] - new_X
                distance = np.linalg.norm(direction_vec)
                direction_vec = direction_vec / distance # Normalize the direction vector

                # Depending on label type distance will be calculated differently            
                #label = (distance*4)-1
                self.data_X = new_X
                self.data_Y = [[-1]]   # Initialize empty array for Y
                self.iter = np.array([self.k])     # Initialize iteration tracking array

                # Positive point
                pos_X = new_X + direction_vec*0.1
                self.data_X = np.append(self.data_X, pos_X, axis=0)
                self.data_Y = np.append(self.data_Y, [[1]], axis=0)   # Initialize empty array for Y
                self.iter = np.append(self.iter, self.k)    # Initialize iteration tracking array
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
                        
                        pos_X = new_X + direction_vec*0.1
                        self.data_X = np.append(self.data_X, pos_X, axis=0)
                        self.data_Y = np.append(self.data_Y, [[+1.0]], axis=0)
                        self.iter = np.append(self.iter, self.k)
            else:
                        # Obstacle detected, two points are generated: one unsafe and one safe

                        # Direction towards the point from roboto
                        direction_vec = rob_pos[:2] - new_X
                        distance = np.linalg.norm(direction_vec)
                        direction_vec = direction_vec / distance
                            
                        # 1. Unsafe point (Obstacle detected)
                        self.data_X = np.append(self.data_X, new_X, axis=0)
                        self.data_Y = np.append(self.data_Y,[[-1.0]],axis=0)  # Label as unsafe
                        self.iter = np.append(self.iter, self.k)

                        # 2. Safe point (offset from obstacle)
                        safe_point = new_X+(direction_vec*safe_offset)
                        # Safe distance meaning the value from point to obstacle is +1
                        # Get range from +1 to -1 where +1 would be no obstacles 
                        dist_s = np.linalg.norm(safe_point-new_X)
                        self.data_X = np.append(self.data_X, safe_point, axis=0)
                        self.data_Y = np.append(self.data_Y, [[+1.0]], axis=0)  # Label as safe
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
        svm_G = np.zeros((n, t.shape[1]))  # Array for gradients, with the same dimensionality as t
        
        # Loop over all input points in t
        for i in range(n):
            # Get the actual value of h for the current input t[i]
            probabilities = self.forward(torch.FloatTensor(t[i].reshape(1, -1)))
            #h_value = 2 * probabilities - 1
            h_value = probabilities
            hsvm_xq[i, 0] = h_value  # Store the h value
            
            # Compute the numerical gradient of h with respect to t[i]
            gradient = self.compute_gradient(torch.FloatTensor(t[i].reshape(1, -1)))
            svm_G[i, :] = gradient  # Store the gradient
            
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
            _,_,self.hpg_map=self.get_cbf_safety_prediction(map_to_plot)
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
