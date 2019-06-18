

DIM=226451;
NUM=226451;
QUERY=10;
FILE_NAME="CurlCurl_1";
k=100;
import numpy as np
from scipy.io import loadmat
x = loadmat(FILE_NAME+'.mat')

lon = x['Problem']['A']


from scipy.sparse import csr_matrix
B = lon[0][0][0:30000].todense() 
