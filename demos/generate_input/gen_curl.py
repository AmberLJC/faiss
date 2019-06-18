

# eculidean distance gt
# support faiss & FALCONN input

QUERY=1000;
ITER=40;
FILE_NAME="CurlCurl_1";
FILE_NAME_OUT="Curl_2w"
k=100;
MAX_MEM=1000;
NUM=MAX_MEM*ITER;
DIM=NUM;
import numpy as np
from scipy.io import loadmat
x = loadmat(FILE_NAME+'.mat')

lon = x['Problem']['A']
from scipy.sparse import csr_matrix

'''
B = lon[0][0][0:MAX_MEM].todense()


file = open(FILE_NAME+".txt", "w")
for i in range (0,ITER):
    np.savetxt(file, lon[0][0][i*MAX_MEM:i*MAX_MEM+MAX_MEM].todense(),fmt='%d')
    print("Read ",i*MAX_MEM, " to ",i*MAX_MEM+MAX_MEM)
file.close()
print("Regular data ready.")


a=np.arange(NUM)
a=a.reshape((NUM,1))
c = np.hstack((a,B))

np.savetxt(FILE_NAME+'_base.txt', c, fmt='%d')
print("Base data ready.")
learn=c[NUM-round(NUM/10):-1]
np.savetxt(FILE_NAME+'_learn.txt', learn, fmt='%d')
query=c[0:QUERY]
np.savetxt(FILE_NAME+'_query.txt', query, fmt='%d')

print("Generating data Done")
print("Starting finding the groud truth")


gt = np.zeros( (QUERY,k) ) # k=100

for i in range(QUERY):
    
  dis_list=np.zeros((1,NUM))

  for j in range(NUM):
      dis=0

      for d in range(DIM):

          dis+=dis+pow( query[i,d+1]- c[j,d+1],2)

      #print(dis)  
      dis_list[0,j]=dis
  rank_no=np.argsort(dis_list)
  gt[i]=rank_no[0][1:k+1]

np.savetxt(FILE_NAME+'_gt.txt', gt , fmt='%d')
'''

print("Generating "+ FILE_NAME_OUT+ " input for Falconn & faiss support Euclidean distance")

'''    
file = open(FILE_NAME_OUT+".txt", "w")
for i in range (0,ITER):
    np.savetxt(file, lon[0][0][i*MAX_MEM:i*MAX_MEM+MAX_MEM].todense(),fmt='%d')
    print("Read regular",i*MAX_MEM, " to ",i*MAX_MEM+MAX_MEM)
file.close()
'''

'''    
file = open(FILE_NAME_OUT+"_base.txt", "w")
a=np.arange(MAX_MEM)
a=a.reshape((MAX_MEM,1))
for i in range (0,ITER):
    
    c = np.hstack((a,lon[0][0][i*MAX_MEM:i*MAX_MEM+MAX_MEM].todense()))
    np.savetxt(file,c ,fmt='%d')
    print("Read base",i*MAX_MEM, " to ",i*MAX_MEM+MAX_MEM)
file.close()


print("Base data ready.")

'''

learn=lon[0][0][int(round(NUM*0.9)):NUM].todense()

a=np.arange(learn.shape[0])
a=a.reshape((learn.shape[0],1))
learn = np.hstack((a,learn))
np.savetxt(FILE_NAME_OUT+'_learn.txt', learn, fmt='%d')
query=lon[0][0][0:QUERY].todense()
a=np.arange(QUERY)
a=a.reshape((QUERY,1))
query = np.hstack((a,query))
np.savetxt(FILE_NAME_OUT+'_query.txt', query, fmt='%d')

print("Generating data Done")
print("Starting finding the groud truth")


gt = np.zeros( (QUERY,k) ) # k=100

for i in range(QUERY):
   
   
  dis_list=np.zeros((1,NUM))
  
  for j in range(NUM):
      dis=0
      
      for d in range(DIM):
          tmp=lon[0][0][j].todense()
          dis+=pow( query[i,d+1]- tmp[0,d],2)
          
      #print(dis)  
      dis_list[0,j]=dis
  rank_no=np.argsort(dis_list)
  gt[i]=rank_no[0][1:k+1]
  if i%10 ==0 :
      print("Find ",i,"th of gt")
  


np.savetxt(FILE_NAME_OUT+'_gt.txt', gt , fmt='%d')




