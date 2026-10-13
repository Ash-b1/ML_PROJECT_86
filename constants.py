import numpy as np

SVR_EPSILON = .0625
N_SEGMENTS = 200
SQUARE_SIZE = 10
C = .125   

COMPACTNESS = 10
SLIC_SIGMA = 1

# For evaluating different values of C and epsilon (train.py --sweep)
C_LIST = [.015625, .03125, .0625, .125, .25, .5]
EPSILON_LIST = [.015625, .03125, .0625, .125]

# For tuning the ICM constants (test.py --tune-icm)
WEIGHT_DIFF_LIST = [1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
THRESHOLD_LIST = [0, 20, 25, 30, 35, 40]    # 0 = no smoothing at all

# Constants for running ICM on the MRF
ICM_ITERATIONS = 10
ITER_EPSILON = .01
COVAR = 0.25       
WEIGHT_DIFF = 2    
THRESHOLD = 25   
BOOST = 1

YUV_FROM_RGB = np.array([[0.299, 0.587, 0.114],
                         [-0.14713, -0.28886, 0.436],
                         [0.615, -0.51499, -0.10001]]).T

RGB_FROM_YUV = np.array([[1, 0, 1.13983],
                         [1, -0.39465, -0.58060],
                         [1, 2.03211, 0]]).T

U_MAX = 0.436
V_MAX = 0.615

MODEL = "svr"          
STANDARDIZE = True  
USE_POSITION = False   
USE_CONTEXT = False    
USE_STATS = True     
CONTEXT_SIZE = 32  
CONTEXT_DOWN = 8       

HGB_MAX_ITER = 300
HGB_LEARNING_RATE = 0.05