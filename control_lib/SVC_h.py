import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
import warnings

from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors

import math
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


# Distance transformer
class DistanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, reference_point=np.array([0, 0])):  # Default reference point (origin)
        self.reference_point = reference_point

    def fit(self, X, y=None):
        return self  

    def transform(self, X):
        # Calculate the Euclidean distance to the reference point
        scaling_factor = 100
        distance = np.linalg.norm(X - self.reference_point, axis=1).reshape(-1, 1)
        return distance

'''________________________ GP Reg _________________________________________'''

class OnlineSVMModel():
    def __init__(self, kernel_type='rbf', C=1.0, epsilon=0.1, min_d_sample=0.1, iter_mem=50, grid_size_plot=0.1, dh_dt=0.1, pre_process=1, envelope=0):
        self.reset_data()
        self.min_d_sample = min_d_sample
        self.mem_num = iter_mem
        self.grid_size = grid_size_plot
        self.dh_dt = dh_dt
        self.safe_offset = 0.5
        # Initialize the SVR model with appropriate parameters
        if envelope:
            self.C = 100
            self.gamma = 100
        else:
            self.C = 1
            self.gamma = 1
        #self.svm_model = SVR(kernel=kernel_type, C=3, epsilon=0.01, gamma=self.gamma)
        # Add weights to classes to penalize mislabeling -1 more 
        self.svm_model = SVC(kernel=kernel_type, C=1, gamma=self.gamma, class_weight={-1:1.5, 1:1})
        self.scaler = StandardScaler()  # Add a scaler for feature scaling
        self.regr = make_pipeline(self.svm_model) # Add a pipeline for ease of use
        self.initial_fit = False
        self.__prediction_plot=None
        self.init = False
        self.init_map=True
        self.set = False
        self.trained = False
        self.pre_process = pre_process

    def reset_data(self):
        self.data_X = None
        self.data_Y = None
        self.data_X_art = None
        self.data_Y_art = None
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
                self.data_X_art = None
                self.data_Y_art = None
                self.iter = None
                self.N = 0

    def create_bubble(self, point, num_of_points, radius):
        """
        Create a 'bubble' of points around a world point to generate an unsafe perimiter around a point

        Parameters:
        point : np.array
            the world coordinate ie center of the bubble
        num_of_points : int
            the number of points around the center point
        radius : float
            the radius of the bubble
        """
        # Initialize array to store the points
        arc_points = []

        # Angle increment for each point
        angle_increment = 2 * np.pi / num_of_points

        # Generate the circle of points
        for i in range(num_of_points):
            angle = i * angle_increment
            # Calculate x and y coordinates
            x = point[0,0] + radius * np.cos(angle)
            y = point[0,1] + radius * np.sin(angle)
            arc_points.append([x, y])

        # Convert list to numpy array
        arc_points = np.array(arc_points)
        return arc_points

    def pre_process_data(self, min_dist = 0.2):
        """
        Clears any positive points that 'reflect' onto negative points to avoid positive classification incorrectly

        """

        # Let's check the distances between positive and negative points to see if some points are too close
        # Get nearest neighbour for each points
        positives = self.data_X[self.data_Y == 1]
        negatives = self.data_X[self.data_Y == -1]
        # Create a mask for deleting positive point
        mask = np.zeros_like(self.data_Y)
        for point in negatives:
            # find index where negative point is in closest to current positive point
            cur_dist = np.inf
            cur_index = None
            for i in range(0,len(positives)):
                dist = np.linalg.norm(point-positives[i])
                if dist<cur_dist:
                    cur_dist = dist
                    cur_index = i
            if cur_index is not None:
                # add to mask
                if dist<min_dist:
                    for i in range(0,len(self.data_X)):
                        data = self.data_X[i]
                        asd = np.all((data==positives[cur_index]))
                        if asd:
                            mask[i] = 1
        # Let's remove the points where the positive point is close enough to a negative one
        indices = np.where(mask == 0)[0]
        self.data_X = self.data_X[indices]
        self.data_Y = self.data_Y[indices]
        self.iter = self.iter[indices]
        self.N = len(self.data_X)


    def set_new_data(self, new_X, new_Y=np.array([[-1.0]]), dist=np.inf,sense_dist=np.inf,rob_pos=np.array([0,0,0]), safe_offset=0.35, bubbles = False):
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
        # Direction towards the point from roboto
        #direction_vec = rob_pos[:2] - new_X
        #distance = np.linalg.norm(direction_vec)
        #direction_vec = direction_vec / distance # Normalize the direction vector
        
        #distance_norm = distance/sense_dist
        # Depending on the label type calculate the label 
        #label = (distance_norm*2)-1
        label = -1
        if self.data_X is None:
            if dist >= sense_dist:
                # Direction towards the point from roboto
                direction_vec = rob_pos[:2] - new_X
                distance = np.linalg.norm(direction_vec)
                direction_vec = direction_vec / distance # Normalize the direction vector

                pos_X = new_X + direction_vec*0.2
                # No previous data, so initialize with the new data
                self.data_X = pos_X
                self.data_Y = 1  # Initialize empty array for Y
                self.iter = np.array([self.k])     # Initialize iteration tracking array
            else:
                # Direction towards the point from roboto
                direction_vec = rob_pos[:2] - new_X
                distance = np.linalg.norm(direction_vec)
                direction_vec = direction_vec / distance # Normalize the direction vector

                # Depending on label type distance will be calculated differently            
                #label = (distance*4)-1
                self.data_X = new_X
                self.data_Y = label   # Initialize empty array for Y
                self.iter = np.array([self.k])     # Initialize iteration tracking array

                # Positive point
                pos_X = new_X + direction_vec*safe_offset
                self.data_X = np.append(self.data_X, pos_X, axis=0)
                self.data_Y = np.append(self.data_Y, 1)   # Initialize empty array for Y
                self.iter = np.append(self.iter, self.k)    # Initialize iteration tracking array
                # Zero point
                #neut_X = new_X + direction_vec*(safe_offset/2)
                #self.data_X = np.append(self.data_X, neut_X, axis=0)
                #self.data_Y = np.append(self.data_Y, 0)   # Initialize empty array for Y
                #self.iter = np.append(self.iter, self.k)   # Initialize iteration tracking array
            self.N = len(self.data_X)
        # Check the distance of the new data point
        else:
            dis_to_mem = np.linalg.norm(self.data_X[:, 0:2] - new_X[0:2], axis=1)
            if min(dis_to_mem) > self.min_d_sample:
                # If the distance is larger than sensing distance, it is safe
                if dist >= sense_dist:
                            # Direction towards the point from roboto
                            direction_vec = rob_pos[:2] - new_X
                            distance = np.linalg.norm(direction_vec)
                            direction_vec = direction_vec / distance # Normalize the direction vector
                            pos_X = new_X+direction_vec*0.2
                            self.data_X = np.append(self.data_X, pos_X, axis=0)
                            self.data_Y = np.append(self.data_Y, 1)
                            self.iter = np.append(self.iter, self.k)
                else:
                            # Obstacle detected, two points are generated: one unsafe and one safe

                            # Direction towards the point from roboto
                            direction_vec = rob_pos[:2] - new_X
                            distance = np.linalg.norm(direction_vec)
                            direction_vec = direction_vec / distance # Normalize the direction vector
                            
                            #label = (distance*4)-1
                            # 1. Unsafe point (Obstacle detected)
                            self.data_X = np.append(self.data_X, new_X, axis=0)
                            self.data_Y = np.append(self.data_Y, label)  # Label as unsafe
                            self.iter = np.append(self.iter, self.k)

                            # Positive point
                            pos_X = new_X+direction_vec*safe_offset
                            self.data_X = np.append(self.data_X, pos_X, axis=0)
                            self.data_Y = np.append(self.data_Y, +1)   # Initialize empty array for Y
                            self.iter = np.append(self.iter, self.k)     # Initialize iteration tracking array
                            # Zero point
                            #neut_X = new_X+direction_vec*(safe_offset/2)
                            #self.data_X = np.append(self.data_X, neut_X, axis=0)
                            #self.data_Y = np.append(self.data_Y, 0)  # Initialize empty array for Y
                            #self.iter = np.append(self.iter, self.k)     # Initialize iteration tracking array

                            # 2. arc points to create an envelope of safety
                            if (bubbles):
                                arc_points = self.create_bubble(new_X,8,0.05)
                                for arc_point in arc_points:
                                    arc_label =-1
                                    self.data_X = np.append(self.data_X, np.reshape(arc_point, (1, 2)), axis=0)
                                    self.data_Y = np.append(self.data_Y, arc_label)  # Label as unsafe
                                    self.iter = np.append(self.iter, self.k)


                self.N = len(self.data_X)
            else:
                # Update label if a new data point was detected nearby
                arg = np.argmin(dis_to_mem)
                self.iter[arg] = self.k
# Fit the SVM model with the current data
            #print("Fitting the SVM model with the latest data...")
            # Combine the real and artificial points for fitting
            #combined_X = np.vstack((self.data_X, self.data_X_art))
            #combined_Y = np.concatenate((self.data_Y, self.data_Y_art))

    def generate_artificial_points(self, safe_offset = 0.2):
        dbscan = DBSCAN(eps = 1.0, min_samples = 5)
        clusters = dbscan.fit_predict(self.data_X)

        unique_clusters = set(clusters) - {-1}
        centroids = []
        for cluster_id in unique_clusters:
            cluster_points = self.data_X[clusters == cluster_id]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)

        for cluster_id in unique_clusters:
            cluster_points = self.data_X[clusters == cluster_id]
            centroid = centroids[cluster_id]
            tangent = self.calculate_tanget(cluster_points=cluster_points)

            for point in cluster_points:
                direction = point - centroid
                direction = direction/np.linalg.norm(direction)

                align_direction = direction + tangent
                align_direction = align_direction/np.linalg.norm(align_direction)

                safe_point = point + safe_offset*align_direction
                zero_point = point + (safe_offset/2)*align_direction
                self.data_X = np.append(self.data_X, np.reshape(safe_point, (1, 2)), axis=0)
                self.data_Y = np.append(self.data_Y, 1)  # Label as unsafe
                self.iter = np.append(self.iter, self.k)
                self.data_X = np.append(self.data_X, np.reshape(zero_point, (1, 2)), axis=0)
                self.data_Y = np.append(self.data_Y, 0)  # Label as unsafe
                self.iter = np.append(self.iter, self.k)

    def calculate_tanget(self, cluster_points):
        pca = PCA(n_components=1)
        pca.fit(cluster_points)
        tangent = pca.components_[0]
        return tangent

    def update_SVG(self):
        # Perform batch learning
        if len(self.data_X) > 5:  # Ensure there are enough points before training
            if self.pre_process:
                self.pre_process_data()
            negative_weight = len(self.data_X[self.data_Y==+1])/len(self.data_X[self.data_Y == -1])
            
            self.svm_model = SVC(kernel='rbf', C=self.C, gamma=self.gamma, class_weight={-1:negative_weight, 1:1})
            #self.svm_model.class_weight_={-1:negative_weight,+1:1}
            # Generate artificial points based on cluster
            #self.generate_artificial_points()
            
            # Fit your SVM model or any other classifier
            #self.regr.fit(combined_X, combined_Y)
            self.svm_model.fit(self.data_X, self.data_Y)

            # Fit and SVR model to this
            #decision_values = self.svm_model.decision_function(self.data_X)
            #self.svr.fit(self.data_X, decision_values)
            # Now call the plotting functions
            #print("Generating visualizations...")

            # 1. Plot 1D learned safety function
            #self.plot_learned_safety_function_1d()

            # 2. Plot 2D learned safety function
            #self.plot_learned_safety_function_2d()

            # 3. Plot the support vectors
            #self.plot_support_vectors()

            # 4. Plot the gradient field of the safety function
            #self.plot_gradient_field()

            #print("Visualizations complete.")
                

    def plot_learned_safety_function_1d(self):
        plt.figure()
        
        # Generate a range of distances
        distances = np.linspace(0, 5, 100).reshape(-1, 1)  # Adjust range as necessary
        # Predict safety values using the trained model
        h_values = self.regr.predict(distances)

        plt.plot(distances, h_values, label='Learned Safety Function h(x)')
        plt.axhline(y=0, color='r', linestyle='--', label='Decision Boundary (h(x)=0)')
        plt.xlabel("Distance")
        plt.ylabel("Safety Function h(x)")
        plt.title("Learned Safety Function vs Distance")
        plt.legend()
        plt.show()
    
    def plot_learned_safety_function_2d(self):
        plt.figure()

        # Generate a mesh grid over the 2D space
        x_min, x_max = self.data_X[:, 0].min() - 1, self.data_X[:, 0].max() + 1
        y_min, y_max = self.data_X[:, 1].min() - 1, self.data_X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

        grid_points = np.c_[xx.ravel(), yy.ravel()]
        # Predict the safety function values for each point on the grid
        h_values = self.regr.predict(grid_points).reshape(xx.shape)

        # Plot the safety function values
        plt.contourf(xx, yy, h_values, levels=100, cmap=RdGn, alpha=0.8)
        plt.colorbar(label='Safety Function h(x)')
        plt.scatter(self.data_X[:, 0], self.data_X[:, 1], c=self.data_Y, cmap=RdGn, edgecolors='k')
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("2D Learned Safety Function and Decision Boundary")
        plt.show()

    def plot_support_vectors(self):
        plt.figure()
        support_vectors = self.svm_model.support_vectors_
        plt.scatter(self.data_X[:, 0], self.data_X[:, 1], c=self.data_Y, cmap=RdGn, label="Training Data")
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors='none', edgecolors='k', 
                    linewidths=2, label="Support Vectors")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Support Vectors in Feature Space")
        plt.legend()
        plt.show()

    def plot_gradient_field(self):
        plt.figure()

        grid_points = self.data_X
        
        # Calculate gradients
        gradients = np.array([self.compute_gradient(pt.reshape(1, -1)) for pt in grid_points])
        grad_x = gradients[:,:, 0]
        grad_y = gradients[:,:, 1]

        # Plot the gradient field
        plt.quiver(self.data_X[:,0], self.data_X[:,1], grad_x, grad_y, color='blue', alpha=0.5)
        plt.scatter(self.data_X[:, 0], self.data_X[:, 1], c=self.data_Y, cmap=RdGn, edgecolors='k')
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Gradient Field of Safety Function")
        plt.show()

    def get_h_value(self, t):
        """
        Get the actual value of h(x) at the given point t using the SVM model.
        """
        #distances = self.distance_transformer.transform(t)
        
        return self.svm_model.predict(t)

    def compute_gradient(self, t, sigma=1.0):
        """
        Computes the gradient of the SVM decision function.
        
        Parameters:
        t : np.array
            Input point where the gradient is to be calculated.
        sigma : float
            Width parameter for the RBF kernel.
        
        Returns:
        grad : np.array
            Gradient of the decision function at point t.
        """
        # Get support vectors and dual coefficients
        support_vectors = self.svm_model.support_vectors_
        dual_coef = self.svm_model.dual_coef_[0]
        gamma = self.gamma
        # Initialize the gradient
        grad = np.zeros_like(t)
        
        # Iterate over all support vectors
        for i in range(len(support_vectors)):
            sv = support_vectors[i]
            alpha = dual_coef[i]
            
            # Compute RBF kernel value
            K_x_sv = np.exp(-gamma * np.linalg.norm(t - sv) ** 2)
            
            # Compute the gradient of the prediction
            grad += alpha * K_x_sv * (-2 * gamma * (t - sv))
        
        return -grad

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
        # self.update_SVG()

        # Initialize arrays for predictions
        n = t.shape[0]
        h_values = np.zeros((n, 1))
        safety_margin_values = np.zeros((n, 1))
        gradients = np.zeros((n, t.shape[1]))
        grad_scaler = 1
        # Iterate over all input points
        for i in range(n):
            h_value_pred = self.get_h_value(t[i].reshape(1, -1))
            decision_values = self.svm_model.decision_function(t[i].reshape(1, -1))
            #h_value = -1*normalized_values[0,0]+1*normalized_values[0,1]
            h_value = decision_values[0]
            gradient = self.compute_gradient(t[i].reshape(1, -1))*grad_scaler

            # Store the safety value and the gradient
            h_values[i, 0] = h_value
            gradients[i, :] = gradient
            # Apply a small safety margin offset
            safety_margin_values[i, 0] = h_value - self.dh_dt

        clipped_h = np.clip(h_values, -1, 1)
        return gradients, safety_margin_values, clipped_h

    def main_kernel(self,a, b,c,d, hypers):
        """
        kernel BETWEEN inputs a,b  
        kernel hyperparamters:       
        ell is the characteristic length scale
        sigma_f is the signal scale
        sigma_y is the observation noise
        hypers = np.array([l_gp, sigma_f, sigma_y])
        to ensure positivity, use full float range """
        """fixed hyper params"""
        [l_gp, sigma_f, sigma_y] = hypers
        # square exponential kernel
        SE = (lambda a, b,e: np.exp(-np.abs(e)*self.decay_rate)*np.exp( np.dot(a-b, a-b) / (-2.0*l_gp**2) ))
        #SE = (lambda a, b,e: np.exp( np.dot(a-b, a-b) / (-2.0*l_gp**2) ))
        kSE = kernel(SE)
        kSEab = kSE(a, b,c,d)
        
        # noise observation kernel
        kNOab= value_eye_fill(1,kSEab.shape)
        # main kernel is the sum of all kernels
        kab = ( sigma_f ** 2 * kSEab + sigma_y**2 * kNOab )
        return kab
                

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

        """ updating the map """
        #print('iter',self.iter)
        #print('data',self.data_X)
        if self.trained: # Assign with data
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