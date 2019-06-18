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
    /*
    int d;
    fread(&d, 1, sizeof(int), f);
    printf(" d =====>  %d\n",d);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);
    *d_out = d; *n_out = n;
    float *x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");
//printf(" %.3f,  %.3f, %.3f.(+1)  %.3f,  %.3f, %.3f \n",x[0],x[128],x[256],x[1],x[129],x[257]);
*/
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
// shft array to remove row headers
    for(size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

        //for(int i=0; i<n*d+n ; i++){printf("%f , ",x[i]);}

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int *ivecs_read(const char *fname,size_t n,size_t d, size_t *d_out, size_t *n_out)
{
    return (int*)fvecs_read(fname, d_out, n_out);
}
int main()
{
 const char *index_key = "PQ16";
    faiss::Index * index;

    size_t d;

    {

        size_t nt;
        float *xt = fvecs_read("img_sift_gt100_inv.txt",1000,100, &d, &nt);
for(int i=0; i < 100 ; i++){printf("%f , ",xt[i]);}
//        printf ("[%.3f s] Preparing index \"%s\" d=%ld\n",
  //              elapsed() - t0, index_key, d);
        index = faiss::index_factory(d, index_key);

   //     printf ("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);

    //    index->train(nt, xt);
        delete [] xt;
    }






}
