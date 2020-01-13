
echo "export CPLUS_INCLUDE_PATH=$CPLUS_INLCUDE_PATH:/home/amberljc/faiss
export LIBRARY_PATH=$LIBRARY_PATH:/home/amberljc/faiss
export LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/amberljc/faiss">>~/.bash_profile
source ~/.bash_profile
./configure --prefix=/home/amberljc/faiss --without-cuda 

make demos -j 30
make install

cd demos/
git clone https://github.com/luyi0619/datagen.git 

cd datagen/
./compile.sh
cd ..
mkdir pipeline/
mkdir res/


mv ../../naive_PQ_sparse/real_main.cpp datagen/
cd datagen/
mkdir res
g++ real_main.cpp -o fake_main  -fpermissive -fopenmp
