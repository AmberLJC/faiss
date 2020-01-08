//
// Created by 刘嘉晨 on 2020-01-08.
//
#include "omp.h"
#include "assert.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include<iostream>
#include<fstream>
using namespace std;
#include <sys/time.h>
#include <getopt.h>
#include "../AutoTune.h"

#include <string.h>
#include <vector>
#include <utility>

using namespace std;
typedef unsigned char uchar;
const size_t MAX_DATA = 100000;
const int SEED = 1234;
class I_ItrReader{
public:
    virtual ~I_ItrReader() {}
    virtual bool IsEnd() = 0;
    virtual std::vector<float> Next() = 0;
};

// Iterative reader for fvec file
class FvecsItrReader : I_ItrReader{
public:
    FvecsItrReader(std::string filename);
    bool IsEnd();
    std::vector<float> Next();
private:
    FvecsItrReader(); // prohibit default construct
    std::ifstream ifs;
    std::vector<float> vec; // store the next vec
    bool eof_flag;
};

// Iterative reader for bvec file
class BvecsItrReader : I_ItrReader{
public:
    BvecsItrReader(std::string filename);
    bool IsEnd();
    std::vector<float> Next(); // Read bvec, but return vec<float>
private:
    BvecsItrReader(); // prohibit default construct
    std::ifstream ifs;
    std::vector<float> vec; // store the next vec
    bool eof_flag;
};

class ItrReader{
public:
    // ext must be "fvecs" or "bvecs"
    ItrReader(std::string filename, std::string ext);
    ~ItrReader();

    bool IsEnd();
    std::vector<float> Next();

private:
    ItrReader();
    I_ItrReader *m_reader;
};


FvecsItrReader::FvecsItrReader(std::string filename)
{
    ifs.open(filename, std::ios::binary);
    assert(ifs.is_open());
    Next();
}

bool FvecsItrReader::IsEnd() { return eof_flag; }

std::vector<float> FvecsItrReader::Next()
{
    std::vector<float> prev_vec = vec; // return the currently stored vec
    int D;
    if(ifs.read( (char *) &D, sizeof(int) )){ // read "D"
        // Then, read a D-dim vec
        vec.resize(D); // allocate D-dim
        assert(ifs.read( (char *) vec.data(), sizeof(float) * D)); // Read D * float.
        eof_flag = false;
    }else{
        vec.clear();
        eof_flag = true;
    }
    return prev_vec;
}

BvecsItrReader::BvecsItrReader(std::string filename)
{
    ifs.open(filename, std::ios::binary);
    assert(ifs.is_open());
    Next();
}

bool BvecsItrReader::IsEnd() { return eof_flag; }

std::vector<float> BvecsItrReader::Next()
{
    std::vector<float> prev_vec = vec; // return the currently stored vec
    int D;
    if(ifs.read( (char *) &D, sizeof(int) )){ // read "D"
        // Then, read a D-dim vec
        vec.resize(D); // allocate D-dim
        std::vector<uchar> buff(D);

        assert(ifs.read( (char *) buff.data(), sizeof(uchar) * D)); // Read D * uchar.

        // Convert uchar to float
        for(int d = 0; d < D; ++d){
            vec[d] = static_cast<float>(buff[d]);
        }

        eof_flag = false;
    }else{
        vec.clear();
        eof_flag = true;
    }
    return prev_vec;
}

ItrReader::ItrReader(std::string filename, std::string ext){
    if(ext == "fvecs"){
        m_reader = (I_ItrReader *) new FvecsItrReader(filename);
    }else if(ext == "bvecs"){
        m_reader = (I_ItrReader *) new BvecsItrReader(filename);
    }else{
        std::cerr << "Error: strange ext type: " << ext << "in ItrReader" << std::endl;
        exit(1);
    }
}

ItrReader::~ItrReader(){
    delete m_reader;
}

bool ItrReader::IsEnd(){
    return m_reader->IsEnd();
}

std::vector<float> ItrReader::Next(){
    return m_reader->Next();
}

void rand_perm(int *perm, size_t n) {
    for (size_t i = 0; i < n; i++) perm[i] = i;
    srand(SEED);
    for (size_t i = 0; i + 1 < n; i++) {
        int i2 = rand() % n;
        std::swap(perm[i], perm[i2]);
    }
}
float * fvecs_read( const char  *filename,  string ext, int d, int top_n){

    float * vecs = new float[top_n*d];
    //vecs.reserve(d * top_n);
    ItrReader reader(filename, ext);
    cout << "initial ItrReader" << endl;

    int cnt = 0;
    while(!reader.IsEnd()){
        if(top_n != -1 && top_n <= cnt ){
            cout << "Stop reading" << endl;
            break;
        }

        std::vector<float> tmp =reader.Next();
        //vecs.push_back(reader.Next());
        //std::copy(tmp.data(), tmp.data() + d,vecs.begin()+cnt*d);

        for(int i = 0; i < d; ++i){
            vecs[cnt*d+i]=tmp[i];
        }
        cnt++;
        if(cnt % 100000 ==0 ){
            cout<<"read "<<cnt<< " lines"<<endl;
        }
    }
    return vecs;
}



float * read_dataset(string file_name,   int d, int top_n ){
    float *dataset = new float [d*top_n];
    std::ifstream file(file_name.c_str());
    if (!file) {
        cout << "can't open the file with the dataset" << endl;
    }
    string s;
    size_t count=0;
    while(getline(file,s)){
        if(count >= top_n){
            break;
        }
        stringstream ss;
        ss<<s;
        count++;
        float tmp;
        while(ss>>tmp){
            dataset[count] = tmp;
        }
    }
    return dataset;
}




double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, nullptr);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}


int main(int argc, char *argv[]) {

    double t0 = elapsed();
    size_t dim =0;

    size_t query_num=0;
    size_t num = 0;
    char filename[50];
    char index_key[50];
    struct option longOpts[] = {
            {"file",  required_argument, 0, 'f'},
            {"num",   required_argument, 0, 'n'},
            {"dim",   required_argument, 0, 'd'},
            {"query", required_argument, 0, 'q'},
            {"index", required_argument, 0, 'i'}
            //{"prune_M", required_argument, 0, 'p'}
    };

    char opt = 0;
    int indext = 0;


    while ((opt = getopt_long(argc, argv, "f:n:d:q:i:", longOpts, &indext)) != -1) {
        switch (opt) {
            case 'f':
                strcpy(filename, optarg);
                //filename = *optarg;
                break;

            case 'i':
                strcpy(index_key, optarg);
                break;

            case 'q':
                query_num = atoi(optarg);
                break;
            case 'n':
                num = atoi(optarg);
                break;
            case 'd':
                dim = atoi(optarg);
                break;

            default:
                cout << "Flag not recognized!\n";
        }
    }


    // const char *index_key = "IVF4096,Flat";
    // const char *index_key = "LSH4096";
    // const char *index_key = "HNSW128_2x32";
    //    const char *index_key = "Flat";
//   const char *index_key = "HNSW128";
    // const char *index_key = "PCA80,Flat";
    // const char *index_key = "IVF4096,PQ8+16";
    //  const char *index_key = "IVF2048,PQ10";
    // const char *index_key = "IMI2x8,PQ32";
    // const char *index_key = "IMI2x8,PQ8+10";
    // const char *index_key = "OPQ16_64,IMI2x8,PQ8+16";

     double t1;
    double idx_cons;
    double trn;
    double src;

    faiss::Index *index;


    printf("[%.3f s] Loading train set\n", elapsed() - t0);

    size_t nt = int(num / 10);
    size_t nb = num-query_num;
    size_t nq = query_num;
    float *x = new float [dim * num];
    string a ="fvecs";
    if(strstr( filename,a.c_str() ) == NULL){
        cout << "read txt file\n";//输出结果。
        x = read_dataset(filename,dim,num);


    }
    else {
        cout << "read fvecs file\n"; //输出结果。
         x  = fvecs_read(filename, "fvecs", dim, num);

    }


    //   for(int i=0; i < 100 ; i=i+1){printf("%f , ",xt[i]);}
    printf("[%.3f s] Preparing index \"%s\" d=%ld\n",
           elapsed() - t0, index_key, dim);
    index = faiss::index_factory(dim, index_key);

    printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);
    t1 = elapsed() - t0;
    float *xt = new float [nt*dim];

    vector<int> perm(num);
    rand_perm(perm.data(), nt);
    for (int i = 0; i < nt; ++i) {
        copy(x + perm[i] * dim, x  + (1 + perm[i]) * dim, xt + dim * i);
    }

    index->train(nt, xt);
    trn = elapsed() - t1 - t0;
    printf("(****%.3f s****)  TRAINING TIME. \n", trn);
    delete[] xt;

    float *xb = new float [(num-query_num)*dim];

    copy(x +query_num*dim , x + num*dim , xb );


    t1 = elapsed() - t0;
    printf("[%.3f s] Indexing database, size %ld*%ld\n",
           t1, nb, dim);
    index->add(nb, xb);
    idx_cons = elapsed() - t1 - t0;
    printf("(****%.3f s****) INDEX CONSTRUCTION TIME. \n", elapsed() - t1 - t0);
    delete[] xb;


    printf("[%.3f s] Loading queries\n", elapsed() - t0);


    float *xq = new float [query_num*dim];
    copy(x , x + query_num*dim, xq  );

    size_t k = 1;
    faiss::Index::idx_t *gt;
    gt = new faiss::Index::idx_t[k * nq];


    printf("[%.3f s] Finding ground truth\n");

#pragma omp parallel for
    for (int i = 0; i < query_num; ++i) {
        float min_dis = 1e20;
        int min_idx = -1;
#pragma omp parallel for
        for (int j = 0; j < nb; j++) {
            int dis = 0;
//#pragma omp parallel for
            for (int d = 0; d < dim; ++d) {
                dis += (xb[j * dim + d] - xq[i * dim + d]) *
                       (xb[j * dim + d] - xq[i * dim + d]);
                if(dis > min_dis) break;
            }
            if (min_dis > dis) {
                min_dis = dis;
                min_idx = j;
            }
        }
        gt[i] = min_idx;

    }
    std::string selected_params;

    printf("[%.3f s] Preparing auto-tune criterion 1-recall at 1 "
           "criterion, with k=%ld nq=%ld\n", elapsed() - t0, k, nq);



    faiss::OneRecallAtRCriterion crit(nq, 1);
    crit.set_groundtruth(k, nullptr, gt);
    crit.nnn = k; // by default, the criterion will request only 1 NN

    printf("[%.3f s] Preparing auto-tune parameters\n", elapsed() - t0);

    faiss::ParameterSpace params;
    params.initialize(index);

    printf("[%.3f s] Auto-tuning over %ld parameters (%ld combinations)\n",
           elapsed() - t0, params.parameter_ranges.size(),
           params.n_combinations());

    faiss::OperatingPoints ops;
    params.explore(index, nq, xq, crit, &ops);

    printf("[%.3f s] Found the following operating points: \n",
           elapsed() - t0);

    ops.display();

    // keep the first parameter that obtains > 0.5 1-recall@1
    for (int i = 0; i < ops.optimal_pts.size(); i++) {
        if (ops.optimal_pts[i].perf > 0.5) {
            selected_params = ops.optimal_pts[i].key;
            break;
        }
    }


    printf("[%.3f s] Setting parameter configuration \"%s\" on index\n",
           elapsed() - t0, selected_params.c_str());

    params.set_index_parameters(index, selected_params.c_str());
    t1 = elapsed() - t0;
    printf("[%.3f s] Perform a search on %ld queries\n",
           t1, nq);
    // output buffers
    faiss::Index::idx_t *I = new faiss::Index::idx_t[nq * k];
    float *D = new float[nq * k];

    index->search(nq, xq, k, D, I);
    src = elapsed() - t1 - t0;

    printf("(**** %s *****) Index\n ", index_key);
    printf("(****%.3f s****)  TRAINING TIME. \n", trn);
    printf("(****%.3f s****) INDEX CONSTRUCTION TIME. \n", idx_cons);
    printf("(****%.3f s****) SEARCHING  TIME. \n", src);
    printf("[%.3f s] Compute recalls\n", elapsed() - t0);



    int n_1 = 0, n_10 = 0, n_100 = 0;
    for (int i = 0; i < nq; i++) {
        int gt_nn = gt[i * k];
        for (int j = 0; j < k; j++) {
            //              printf(" the %d th predicted 1-nn is pt %d\n",i,I[i*k+j]);
            if (I[i * k + j] == gt_nn) {
                if (j < 1) n_1++;
                if (j < 10) n_10++;
                if (j < 100) n_100++;
            }
        }
    }
    printf("R@1 = %.4f\n", n_1 / float(nq));
    printf("R@10 = %.4f\n", n_10 / float(nq));
    printf("R@100 = %.4f\n", n_100 / float(nq));
    ofstream write;

    delete[] xq;
    delete[] gt;
    delete index;
    return 0;




}


