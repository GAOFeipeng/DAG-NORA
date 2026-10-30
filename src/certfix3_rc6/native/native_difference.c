/* Decision-oriented objective-difference backend (research branch).
 * Exact stored binary64 inputs are promoted before squaring/differencing.
 * Compute in long double AND IEEE binary128, with no fast-math/FMA contraction.
 * Sign-preserving nonpivoted elimination is used only after the original
 * M-matrix/inverse guards pass. A nonpositive pivot is unresolved, never accepted.
 * Cross-precision agreement is numerical evidence, NOT an interval proof.
 */
#include <quadmath.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <time.h>

static _Thread_local double pc_deadline=0.;
static double pc_budget_seconds=2.;
static double pc_now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static int pc_expired(void){return pc_deadline>0. && pc_now()>=pc_deadline;}
void set_precision_budget_seconds(double v){pc_budget_seconds=(isfinite(v)&&v>0)?v:2.;}
double precision_budget_seconds(void){return pc_budget_seconds;}

#define MAKE_KERNEL(SFX,T,ABS,LOG,FINITE) \
static int logdet_##SFX(int n,const double *w,double s,T *ld,T *scale,T *minpivot){ \
 size_t nn=(size_t)n*n; T *a=malloc(nn*sizeof(T)); if(!a)return -1; \
 for(int i=0;i<n;i++){if((i%8)==0&&pc_expired()){free(a);return -20;}for(int j=0;j<n;j++){T z=(T)w[(size_t)i*n+j];a[(size_t)i*n+j]=(i==j?(T)s:0)-z*z;}} \
 *ld=0;*scale=0;*minpivot=(T)s; \
 for(int k=0;k<n;k++){T p=a[(size_t)k*n+k];if(!(p>0)||!FINITE(p)){free(a);return k+1;} \
  if(p<*minpivot)*minpivot=p;T l=LOG(p);*ld+=l;*scale+=ABS(l); \
  for(int i=k+1;i<n;i++){if((i%8)==0&&pc_expired()){free(a);return -20;}T f=a[(size_t)i*n+k]/p;if(f==0)continue; \
   for(int j=k+1;j<n;j++)a[(size_t)i*n+j]-=f*a[(size_t)k*n+j];a[(size_t)i*n+k]=0;} \
 }free(a);return 0;} \
static int full_##SFX(int n,const double*g,const double*p,const double*q,double mu,double s,double lam,int sym,T*out){ \
 T dl=0,d1=0,at=0; \
 for(int j=0;j<n;j++){if(pc_expired())return -20;int changed=0;for(int i=0;i<n;i++)if(p[(size_t)i*n+j]!=q[(size_t)i*n+j]){changed=1;break;}if(!changed)continue; \
  for(int i=0;i<n;i++){if((i%8)==0&&pc_expired())return -20;size_t ij=(size_t)i*n+j;T v=(T)q[ij]-(T)p[ij];if(v==0)continue;T gr=0,gv=0; \
   for(int k=0;k<n;k++){size_t kj=(size_t)k*n+j;T gik=(T)g[(size_t)i*n+k];T gs=sym?gik:(gik+(T)g[(size_t)k*n+i])/2; \
    gr+=gs*((T)p[kj]-(k==j));gv+=gik*((T)q[kj]-(T)p[kj]);} \
   T term=v*gr+v*gv/2;dl+=term;at+=ABS(term);d1+=ABS((T)q[ij])-ABS((T)p[ij]);}} \
 T l0,l1,a0,a1,m0,m1;int e=logdet_##SFX(n,p,s,&l0,&a0,&m0);if(e)return e<0?e:1000+e; \
 e=logdet_##SFX(n,q,s,&l1,&a1,&m1);if(e)return e<0?e:2000+e;T dh=l0-l1; \
 out[0]=dl;out[1]=d1;out[2]=(T)mu*dl;out[3]=(T)mu*(T)lam*d1;out[4]=dh;out[5]=(T)mu*(dl+(T)lam*d1)+dh; \
 out[6]=1+ABS((T)mu)*(at+ABS((T)lam*d1))+a0+a1;out[7]=m0<m1?m0:m1;return FINITE(out[5])?0:-3;} \
static int block_##SFX(int n,const double*g,const double*b,const double*c,const double*p,const double*q,double mu,double lam,T*out){ \
 T dl=0,d1=0,at=0,a=1,bb=1; \
 for(int i=0;i<n;i++){if((i%8)==0&&pc_expired())return -20;T v=(T)q[i]-(T)p[i],gr=-(T)b[i],gv=0; \
  for(int k=0;k<n;k++){gr+=(T)g[(size_t)i*n+k]*(T)p[k];gv+=(T)g[(size_t)i*n+k]*((T)q[k]-(T)p[k]);} \
  T term=v*gr+v*gv/2;dl+=term;at+=ABS(term);if(p[i]!=q[i])d1+=ABS((T)q[i])-ABS((T)p[i]);a-=(T)c[i]*(T)p[i]*(T)p[i];bb-=(T)c[i]*(T)q[i]*(T)q[i];} \
 if(!(a>0)||!(bb>0))return -4;T dh=LOG(a)-LOG(bb); \
 out[0]=dl;out[1]=d1;out[2]=(T)mu*dl;out[3]=(T)mu*(T)lam*d1;out[4]=dh;out[5]=(T)mu*(dl+(T)lam*d1)+dh; \
 out[6]=1+ABS((T)mu)*(at+ABS((T)lam*d1))+ABS(LOG(a))+ABS(LOG(bb));out[7]=a<bb?a:bb;return FINITE(out[5])?0:-3;}

MAKE_KERNEL(ld,long double,fabsl,logl,isfinite)
MAKE_KERNEL(q,__float128,fabsq,logq,finiteq)

int full_difference(int n,const double*g,const double*p,const double*q,double mu,double s,double lam,int sym,double*out,char*text,size_t cap){
 long double lo[8]={0};__float128 hi[8]={0};int a=full_ld(n,g,p,q,mu,s,lam,sym,lo);int b=full_q(n,g,p,q,mu,s,lam,sym,hi);
 if(a||b)return a?a:b;size_t used=0;
 for(int i=0;i<8;i++){out[i]=(double)lo[i];out[8+i]=(double)hi[i];char h[128];quadmath_snprintf(h,sizeof h,"%.36Qe",hi[i]);int k=snprintf(text+used,cap-used,"%.25Le|%s\n",lo[i],h);if(k<0||(size_t)k>=cap-used)return -5;used+=(size_t)k;}return 0;
}
int block_difference(int n,const double*g,const double*b,const double*c,const double*p,const double*q,double mu,double lam,double*out,char*text,size_t cap){
 long double lo[8]={0};__float128 hi[8]={0};int a=block_ld(n,g,b,c,p,q,mu,lam,lo);int z=block_q(n,g,b,c,p,q,mu,lam,hi);
 if(a||z)return a?a:z;size_t used=0;
 for(int i=0;i<8;i++){out[i]=(double)lo[i];out[8+i]=(double)hi[i];char h[128];quadmath_snprintf(h,sizeof h,"%.36Qe",hi[i]);int k=snprintf(text+used,cap-used,"%.25Le|%s\n",lo[i],h);if(k<0||(size_t)k>=cap-used)return -5;used+=(size_t)k;}return 0;
}
int long_double_mantissa_bits(void){return LDBL_MANT_DIG;}

/* Production interface: finite LD+binary128 route; no decimal text or mpmath.
 * This keeps the R1 separation idea, deriving the resolution floor from the
 * actual long-double machine epsilon instead of a decimal consensus constant.
 * Returns numerical evidence, not a rigorous interval certificate.
 */
static int predq(__float128 v,__float128 t,int kind){
 if(kind==0)return v<=t;if(kind==1)return v<t;if(kind==2)return fabsq(v)<=t;return 0;
}
static int decide_records(int n,const long double*lo,const __float128*hi,double bound,int kind,double*out){
 if(kind<0||kind>2||!isfinite(bound))return -7;
 __float128 t=(__float128)bound,a=(__float128)lo[5],b=hi[5];
 __float128 scale=fmaxq(1,fmaxq((__float128)lo[6],hi[6]));
 __float128 discrepancy=fabsq(a-b);
 __float128 rounding=256*FLT128_EPSILON*(__float128)n*n*n*scale;
 __float128 cast=8*DBL_EPSILON*(fabsq(b)+fabsq(t)+DBL_MIN);
 __float128 err=fmaxq(32*discrepancy,fmaxq(rounding,cast));
 __float128 resolution=fmaxq(8*LDBL_EPSILON*fmaxq(1,fabsq(b)),fabsq(t)/100);
 __float128 gap=fabsq((kind==2?fabsq(b):b)-t);
 int ok=finiteq(a)&&finiteq(b)&&finiteq(err)&&err<=resolution&&gap>4*err&&predq(a,t,kind)==predq(b,t,kind);
 out[0]=ok;out[1]=ok?predq(b,t,kind):-1;out[2]=(double)b;
 out[3]=(double)err;out[4]=(double)gap;out[5]=(double)discrepancy;
 out[6]=(double)scale;out[7]=(double)resolution;
 return 0;
}
static int full_decide_impl(int n,const double*g,const double*p,const double*q,double mu,double s,double lam,int sym,double bound,int kind,double*out){
 if(n<=0||!(s>0)||!(mu>0)||!isfinite(lam))return -6;
 long double lo[8]={0};__float128 hi[8]={0};
 int a=full_ld(n,g,p,q,mu,s,lam,sym,lo);if(a)return a;
 int b=full_q(n,g,p,q,mu,s,lam,sym,hi);if(b)return b;
 return decide_records(n,lo,hi,bound,kind,out);
}
static int block_decide_impl(int n,const double*g,const double*b,const double*c,const double*p,const double*q,double mu,double lam,double bound,int kind,double*out){
 if(n<=0||!(mu>0)||!isfinite(lam))return -6;
 long double lo[8]={0};__float128 hi[8]={0};
 int a=block_ld(n,g,b,c,p,q,mu,lam,lo);if(a)return a;
 int z=block_q(n,g,b,c,p,q,mu,lam,hi);if(z)return z;
 return decide_records(n,lo,hi,bound,kind,out);
}

/* One absolute CLOCK_MONOTONIC deadline shared by both precision levels.
 * Cooperative checks permit one check granule of scheduling overshoot. */
int full_decide_until(int n,const double*g,const double*p,const double*q,double mu,double s,double lam,int sym,double bound,int kind,double*out,double deadline){
 if(!isfinite(deadline)||deadline<=pc_now())return -20;
 pc_deadline=deadline;
 int code=full_decide_impl(n,g,p,q,mu,s,lam,sym,bound,kind,out);
 if(pc_expired())code=-20;
 pc_deadline=0.;return code;
}
int block_decide_until(int n,const double*g,const double*b,const double*c,const double*p,const double*q,double mu,double lam,double bound,int kind,double*out,double deadline){
 if(!isfinite(deadline)||deadline<=pc_now())return -20;
 pc_deadline=deadline;
 int code=block_decide_impl(n,g,b,c,p,q,mu,lam,bound,kind,out);
 if(pc_expired())code=-20;
 pc_deadline=0.;return code;
}

/* Retain the legacy ABI for archived diagnostic callers only. */
int full_decide(int n,const double*g,const double*p,const double*q,double mu,double s,double lam,int sym,double bound,int kind,double*out){
 pc_deadline=pc_now()+pc_budget_seconds;
 int code=full_decide_impl(n,g,p,q,mu,s,lam,sym,bound,kind,out);
 pc_deadline=0.;return code;
}
int block_decide(int n,const double*g,const double*b,const double*c,const double*p,const double*q,double mu,double lam,double bound,int kind,double*out){
 pc_deadline=pc_now()+pc_budget_seconds;
 int code=block_decide_impl(n,g,b,c,p,q,mu,lam,bound,kind,out);
 pc_deadline=0.;return code;
}
