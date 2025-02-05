import numpy as np

'''_________________ Circulation Embedded Coefficients______________________'''

class CE():
    def __init__(self,h_c_0,a_c,k_abs_n,k_theta,h_dw):
        self.h_c_0=h_c_0
        self.a_c=a_c
        self.k_abs_n=k_abs_n
        self.hold_k_cir=False
        self.k_theta=k_theta
        # the value will be updated once hold_k_cir is True and then used
        self.d_gtg_0=0.0
        self.k_cir=np.array([[-1]])
        self.h_dw=h_dw
        
    def get_cbf_circulation_params(self, h, G, u_nom, d_gtg):
        
        # calculate circulation direction u_c=R(theta)grad_h/||h||
        h_sqz = np.squeeze(h)[()]
        theta_c=np.pi/2+self.k_theta*(h_sqz-self.h_dw)
        #print('corection',h,theta_c)
        rotation_matrix=np.array([ [np.cos(theta_c), -np.sin(theta_c), 0],[np.sin(theta_c), np.cos(theta_c), 0],[0, 0, 1]])
        grad_h_n=-G/np.linalg.norm(G)
        u_cir=np.dot(rotation_matrix,grad_h_n.T).T
        
        # calculate circulation input norm k_abs
        unn=np.linalg.norm(u_nom)
        k_abs=self.k_abs_n*unn
        
        # calculate circulation coefficient k_cir
        if self.hold_k_cir and self.d_gtg_0-0.1> d_gtg:
            print('End of circulation!!!!!!!!!!!!!!!!!!!!!')
            self.hold_k_cir=False
        if self.hold_k_cir:
            self.k_cir=np.array([[1]])
            #print('Continue circulating')
        else:
            u_nom_n=u_nom/unn
            cos_psi=np.dot(u_nom_n,-grad_h_n.T)
            self.k_cir=2/(1+np.exp(-self.a_c*((cos_psi/h)-(1/self.h_c_0))))-1
            if  self.k_cir>0.99:
                self.hold_k_cir=True
                self.d_gtg_0=d_gtg
                print('Start of circulation!!!!!!!!!!!!!!!!!!!!!!!')
                
            
        #return circualtion constraint oefficients
        return u_cir, self.k_cir, k_abs

 