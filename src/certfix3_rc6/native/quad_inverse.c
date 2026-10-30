/* Numerical fallback, not an optimizer. Binary64 W are squared in binary128.
 * Build with -O2 -ffp-contract=off, without fast-math, and link libquadmath.
 * Nonpivoted elimination exploits M-matrix signs. A separate residual enclosure
 * validates the computed inverse; a failed pivot is fail-closed, not proof that
 * the exact matrix is illegal. */
#include <quadmath.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <stddef.h>
static double up(__float128 a) { return nextafter((double)a, INFINITY); }
int quad_inverse(int n,const double*w,double s,double*out,double*eres,double*cast,double*ld,double*ldrad){
 size_t nn=(size_t)n*n;
 __float128 *m=calloc(nn,sizeof(__float128)),*a=calloc(nn,sizeof(__float128)),*b=calloc(nn,sizeof(__float128)),*k=calloc(nn,sizeof(__float128));
 if(!m||!a||!b||!k){free(m);free(a);free(b);free(k);return -2;}
 __float128 logs=0,eps=FLT128_EPSILON/2,gm=(n+4)*eps/(1-(n+4)*eps);
 for(int i=0;i<n;i++)for(int j=0;j<n;j++){
  size_t ij=(size_t)i*n+j;__float128 v=w[ij];m[ij]=a[ij]=(i==j?(__float128)s:0)-v*v;b[ij]=(i==j);
 }
 for(int z=0;z<n;z++){
  __float128 pivot=a[(size_t)z*n+z];
  if(!(pivot>0)||!finiteq(pivot)){free(m);free(a);free(b);free(k);return z+1;}
  logs+=logq(pivot);
  for(int i=z+1;i<n;i++){
   __float128 f=a[(size_t)i*n+z]/pivot;
   if(f==0)continue;
   for(int j=z+1;j<n;j++)a[(size_t)i*n+j]-=f*a[(size_t)z*n+j];
   for(int j=0;j<n;j++)b[(size_t)i*n+j]-=f*b[(size_t)z*n+j];
   a[(size_t)i*n+z]=0;
  }
 }
 for(int i=n-1;i>=0;i--)for(int j=0;j<n;j++){
  __float128 v=b[(size_t)i*n+j];
  for(int z=i+1;z<n;z++)v-=a[(size_t)i*n+z]*k[(size_t)z*n+j];
  k[(size_t)i*n+j]=v/a[(size_t)i*n+i];out[(size_t)i*n+j]=(double)k[(size_t)i*n+j];
  cast[(size_t)i*n+j]=up(fabsq(k[(size_t)i*n+j]-(__float128)out[(size_t)i*n+j]));
 }
 for(int i=0;i<n;i++)for(int j=0;j<n;j++){
  __float128 dot=0,adot=0;
  for(int z=0;z<n;z++){__float128 v=m[(size_t)i*n+z]*k[(size_t)z*n+j];dot+=v;adot+=fabsq(v);}
  __float128 ident=(i==j);
  eres[(size_t)i*n+j]=up(fabsq(ident-dot)+gm*(ident+adot));
 }
 *ld=(double)logs;*ldrad=up(fabsq(logs-(__float128)*ld)+gm*(1+fabsq(logs)));
 free(m);free(a);free(b);free(k);return 0;
}
int quad_witness(int n,const double*w,double s,const double*x,double*lo,double*hi){
 __float128 eps=FLT128_EPSILON/2,gm=(n+4)*eps/(1-(n+4)*eps);
 for(int i=0;i<n;i++){
  __float128 dot=(__float128)s*x[i],adot=fabsq(dot);
  for(int j=0;j<n;j++){__float128 v=(__float128)w[(size_t)i*n+j];v=v*v*(__float128)x[j];dot-=v;adot+=fabsq(v);}
  __float128 rad=gm*adot;
  lo[i]=nextafter((double)(dot-rad),-INFINITY);hi[i]=nextafter((double)(dot+rad),INFINITY);
 }
 return 0;
}
