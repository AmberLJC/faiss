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



