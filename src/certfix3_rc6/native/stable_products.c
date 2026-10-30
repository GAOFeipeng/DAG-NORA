/* RC6 update-form products. Long double, no reassociation/FMA, no new precision.
 * Exact saved binary64 weights are promoted before differencing. Zero changes
 * contribute zero, so computing their matrix-vector rows is unnecessary.
 * These functions generate terms only. Existing decision radii and bounded
 * LD/binary128 final comparison are unchanged.
 */
#include <stddef.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
int rc6_linear_terms(int n,int m,const long double*g,const double*p,const double*q,const int64_t*cols,long double*out,double deadline) {
 for(int z=0;z<m;z++) {
  if(deadline>0 && now_s()>=deadline) return -20;
  int j=(int)cols[z];
  for(int i=0;i<n;i++) {
   long double v=(long double)q[(size_t)i*n+j]-(long double)p[(size_t)i*n+j];
   if(v==0) {out[(size_t)i*m+z]=0;continue;}
   long double gr=0,gv=0;
   for(int k=0;k<n;k++) {
    long double a=g[(size_t)i*n+k];
    gr+=a*((long double)p[(size_t)k*n+j]-(k==j));
    gv+=a*((long double)q[(size_t)k*n+j]-(long double)p[(size_t)k*n+j]);
   }
   out[(size_t)i*m+z]=v*gr+v*gv/2;
  }
 }
 return 0;
}
int rc6_ld_matmul_t(int n,int m,int k,const long double*a,const long double*bt,long double*out,double deadline) {
 for(int i=0;i<n;i++) {
  if(deadline>0 && now_s()>=deadline) return -20;
  for(int j=0;j<m;j++) {
   long double s=0;
   for(int l=0;l<k;l++) s+=a[(size_t)i*k+l]*bt[(size_t)j*k+l];
   out[(size_t)i*m+j]=s;
  }
 }
 return 0;
}
