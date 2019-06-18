/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */



#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <sys/time.h>

#include "../AutoTune.h"


/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
 *
 * and unzip it to the sudirectory sift1M.
 **/

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/


float * fvecs_read (const char *fname,size_t n,size_t d,
                    size_t *d_out, size_t *n_out)
{
    FILE *f = fopen(fname, "r");
    if(!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
   
    //fread(&d, 1, sizeof(int), f);
    //assert((d > 0 && d < 1000000) || !"unreasonable dimension");
  
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
 //   printf(" file size:  %d   ????\n",sz);
   // assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
 //   size_t n = sz / ((d + 1) * 4);
    
   // n=149608;
 
    *d_out = d; *n_out = n;
    float *x = new float[n * (d + 1)];
   int cnt =0;
   float in;
  while (cnt< n*(d+1)){ 
	  fscanf(f,"%f",&in);
x[cnt]=in;
//printf("%.1f , ",x[cnt]);
cnt++;
  }
  
  
  /*size_t nr = fread(x, sizeof(int), n * (d + 1), f);
   printf(" total read in nb: %d\n",nr);
   // assert(nr == n * (d + 1) || !"could not read whole file");
   for(int i=0; i<100 ; i++){printf("%d , ",x[i]);}
*
*/
 // for(int i=0; i<n*d+n ; i++){printf("%f , ",x[i]);}
  // shft array to remove row headers
    for(size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

	//for(int i=0; i<n*d+n ; i++){printf("%f , ",x[i]);}

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
/*
 int *ivecs_read(const char *fname, size_t n,size_t d, size_t *d_out, size_t *n_out)
{
    return (int*)fvecs_read(fname, n, d, d_out, n_out);
}
*/
double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, nullptr);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}



int main()
{double t1;
    double t0 = elapsed();
 //const char *index_key = "IMI2x8,PQ8+10";// this is typically the fastest one.
    // const char *index_key = "IVF4096,Flat";
    // const char *index_key = "LSH4096";
    // const char *index_key = "HNSW128_2x32";
    // const char *index_key = "Flat";
  //  const char *index_key = "HNSW16";
    // const char *index_key = "PCA80,Flat";
  //   const char *index_key = "IVF2048,PQ10";
    // const char *index_key = "IVF4096,PQ32";
    // const char *index_key = "IMI2x8,PQ32";
    // const char *index_key = "IMI2x8,PQ8+16";
   const char *index_key = "PQ40";
    faiss::Index * index;

    size_t dim=1000;
   // size_t num=149608;
   size_t num = 100000;
    size_t d;
size_t q=1000;
    {
        printf ("[%.3f s] Loading train set\n", elapsed() - t0);

        size_t nt;
        float *xt = fvecs_read("img_sift_learn_ds.txt",  10000 ,dim, &d, &nt);
    //   for(int i=0; i < 100 ; i=i+1){printf("%f , ",xt[i]);}
	printf ("[%.3f s] Preparing index \"%s\" d=%ld\n",
                elapsed() - t0, index_key, d);
        index = faiss::index_factory(d, index_key);

        printf ("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);

        index->train(nt, xt);
        delete [] xt;
    }


    {
        printf ("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb, d2;
        float *xb = fvecs_read("img_sift_base_ds.txt",num,dim, &d2, &nb);
       // for(int i=0; i < 100 ; i=i+1){printf("%f , ",xb[i]);}
	assert(d == d2 || !"dataset does not have same dimension as train set");
        t1 =   elapsed() - t0;
		printf ("[%.3f s] Indexing database, size %ld*%ld\n",
                t1 , nb, d);
        index->add(nb, xb);

 printf ("(****%.3f s****) INDEX CONSTRUCTION TIME. \n", elapsed()-t1 - t0);
 delete [] xb;
    }

    size_t nq;
	float *xq;

    {
        printf ("[%.3f s] Loading queries\n", elapsed() - t0);

        size_t d2;
        xq = fvecs_read("img_sift_query_ds.txt",q,dim, &d2, &nq);
//	for(int i=0;i < 3000; i=i+1000){printf("%f , ",xq[i]);}
	assert(d == d2 || !"query does not have same dimension as train set");

    }

    size_t k; // nb of results per query in the GT
    faiss::Index::idx_t *gt;  // nq * k matrix of ground-truth nearest-neighbors

    {
        printf ("[%.3f s] Loading ground truth for %ld queries\n",
                elapsed() - t0, nq);

        // load ground-truth and convert int to long
        size_t nq2;
float *gt_int = fvecs_read("img_sift_gt_ds.txt",q,100, &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");
//printf("read %d ground truth with dimension: %d \n ",nq2,k);
        gt = new faiss::Index::idx_t[k * nq];
        for(int i = 0; i < k * nq; i++) {
            gt[i] =( int )  gt_int[i];
        }
 //for(int i=0; i < 300 ; i=i+100){printf("%d , ",gt[i]);}
 	delete [] gt_int;
    }

    // Result of the auto-tuning
    std::string selected_params;

    { // run auto-tuning

        printf ("[%.3f s] Preparing auto-tune criterion 1-recall at 1 "
                "criterion, with k=%ld nq=%ld\n", elapsed() - t0, k, nq);

        faiss::OneRecallAtRCriterion crit(nq, 1);
        crit.set_groundtruth (k, nullptr, gt);
        crit.nnn = k; // by default, the criterion will request only 1 NN

        printf ("[%.3f s] Preparing auto-tune parameters\n", elapsed() - t0);

        faiss::ParameterSpace params;
        params.initialize(index);

        printf ("[%.3f s] Auto-tuning over %ld parameters (%ld combinations)\n",
                elapsed() - t0, params.parameter_ranges.size(),
                params.n_combinations());

        faiss::OperatingPoints ops;
        params.explore (index, nq, xq, crit, &ops);

        printf ("[%.3f s] Found the following operating points: \n",
                elapsed() - t0);

        ops.display ();

        // keep the first parameter that obtains > 0.5 1-recall@1
        for (int i = 0; i < ops.optimal_pts.size(); i++) {
            if (ops.optimal_pts[i].perf > 0.5) {
                selected_params = ops.optimal_pts[i].key;
                break;
            }
        }
        assert (selected_params.size() >= 0 ||
                !"could not find good enough op point");
    }


    { // Use the found configuration to perform a search

        faiss::ParameterSpace params;

        printf ("[%.3f s] Setting parameter configuration \"%s\" on index\n",
                elapsed() - t0, selected_params.c_str());

        params.set_index_parameters (index, selected_params.c_str());
t1= elapsed() - t0;
        printf ("[%.3f s] Perform a search on %ld queries\n",
                 t1, nq);
 printf ("(****%.3f s****) SEARCHING  TIME. \n", elapsed()-t1 - t0);
        // output buffers
        faiss::Index::idx_t *I = new  faiss::Index::idx_t[nq * k];
        float *D = new float[nq * k];

        index->search(nq, xq, k, D, I);

        printf ("[%.3f s] Compute recalls\n", elapsed() - t0);

        // evaluate result by hand.
        int n_1 = 0, n_10 = 0, n_100 = 0;
        for(int i = 0; i < nq; i++) {
            int gt_nn = gt[i * k];
	    for(int j = 0; j < k; j++) {
  //              printf(" the %d th predicted 1-nn is pt %d\n",i,I[i*k+j]);
		if (I[i * k + j] == gt_nn ){ 
                    if(j < 1) n_1++;
                    if(j < 10) n_10++;
                    if(j < 100) n_100++;
                }
            }
        }
        printf("R@1 = %.4f\n", n_1 / float(nq));
        printf("R@10 = %.4f\n", n_10 / float(nq));
        printf("R@100 = %.4f\n", n_100 / float(nq));

    }

    delete [] xq;
    delete [] gt;
    delete index;
    return 0;
}
