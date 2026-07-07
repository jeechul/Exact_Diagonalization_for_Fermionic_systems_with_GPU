#ifndef __MAGMA_LAPACK__
#define __MAGMA_LAPACK__

#include <iostream>
#include <magma_v2.h>
#include <magma_lapack.h>

namespace LAPACK{
	void dsted(int n_, double *d, double *e, int il_, int iu_, double *m, double *z);
	void ssted(int n_, float *d, float *e, int il_, int iu_, float *m, float *z);
	void dsyevd(const int m, double * d_A, double * d_W);
	void zgesdd(int m, int n, magmaDoubleComplex * d_A, double * S);
}

namespace LAPACKf77{
	void dsyevd(const int m, double * A, double * W);
	void zheevd(const int m, magmaDoubleComplex * A, double * W);
    void dgeev(const int m, double * A, double * Wr, double * Wi);
	void zgesdd(int m, int n, magmaDoubleComplex * A, double * S);
	void dgesdd(int m, int n, double * A, double * S);
	void dsted(int n_, double *d, double *e, double *m, double *z);
}

void LAPACK::dsted(int n_, double *d, double *e, int il_, int iu_, double *m, double *z)
{
	magma_init();

	magma_int_t info, lwork=-1, liwork=-1;
	magma_range_t range = MagmaRangeI;
	magma_int_t n = n_;
	double vl = 0., vu = 1.;
	magma_int_t il = il_;
	magma_int_t iu = iu_;
	double tmp1[1];
	magma_int_t tmp2[1];

	double *d_h, *e_h;
	int dwork_size = 3*n*n/2+3*n; 
	magmaDouble_ptr dwork;
	magma_dmalloc_cpu(&d_h,n);  
	magma_dmalloc_cpu(&e_h,n-1);  
	magma_dmalloc_cpu(&dwork,dwork_size);  

	for(int i=0;i<n_;i++) d_h[i] = d[i];
	for(int i=0;i<n_-1;i++) e_h[i] = e[i];

	magma_dstedx(range,n,vl,vu,il,iu,d_h,e_h,z,n,tmp1,lwork,tmp2,liwork,dwork,&info);
	lwork = (magma_int_t)tmp1[0]; 
	liwork = tmp2[0]; 

	double *work; magma_int_t *iwork;
	magma_dmalloc_cpu(&work,lwork);
	iwork = (magma_int_t*)malloc(liwork*sizeof(magma_int_t));
	magma_dstedx(range,n,vl,vu,il,iu,d_h,e_h,z,n,work,lwork,iwork,liwork,dwork,&info);
	
	for(int i=0;i<n_;i++) m[i] = d_h[i];

	free(d_h);
	free(e_h);
	free(work);
	free(dwork);
	free(iwork);
	assert(info==0);

	magma_finalize();  
}

void LAPACK::ssted(int n_, float *d, float *e, int il_, int iu_, float *m, float *z)
{
	magma_init();

	magma_int_t info, lwork=-1, liwork=-1;
	magma_range_t range = MagmaRangeI;
	magma_int_t n = n_;
	float vl = 0., vu = 1.;
	magma_int_t il = il_;
	magma_int_t iu = iu_;
	float tmp1[1];
	magma_int_t tmp2[1];

	float *d_h, *e_h;
	int dwork_size = 3*n*n/2+3*n;
	magmaFloat_ptr dwork;
	magma_smalloc_cpu(&d_h,n);  
	magma_smalloc_cpu(&e_h,n-1);  
	magma_smalloc_cpu(&dwork,dwork_size);  

	for(int i=0;i<n_;i++) d_h[i] = d[i];
	for(int i=0;i<n_-1;i++) e_h[i] = e[i];

	magma_sstedx(range,n,vl,vu,il,iu,d_h,e_h,z,n,tmp1,lwork,tmp2,liwork,dwork,&info);
	lwork = (magma_int_t)tmp1[0]; 
	liwork = tmp2[0]; 

	float *work; magma_int_t *iwork;
	magma_smalloc_cpu(&work,lwork);
	iwork = (magma_int_t*)malloc(liwork*sizeof(magma_int_t));
	magma_sstedx(range,n,vl,vu,il,iu,d_h,e_h,z,n,work,lwork,iwork,liwork,dwork,&info);
	
	for(int i=0;i<n_;i++) m[i] = d_h[i];

	free(d_h);
	free(e_h);
	free(work);
	free(iwork);
	assert(info==0);

	//magma_finalize();  
}

void LAPACK::dsyevd(const int m, double *d_A, double *d_W)
{
	magma_init();
	
	double * h_work;
	magma_int_t lwork;
	magma_int_t *iwork;
	magma_int_t liwork;
	magma_int_t info;

	double aux_work[1];
	magma_int_t aux_iwork[1];
	magma_dsyevd(MagmaVec,MagmaLower,m,d_A,m,d_W,aux_work,-1,aux_iwork,-1,&info);	
	lwork = (magma_int_t)aux_work[0];
	liwork = aux_iwork[0];
	magma_imalloc_cpu(&iwork,liwork);
	magma_dmalloc_cpu(&h_work,lwork);

	magma_dsyevd(MagmaVec,MagmaLower,m,d_A,m,d_W,h_work,lwork,iwork,liwork,&info);

	free(h_work);
	free(iwork);
	assert(info==0);
	
	//magma_finalize();
}

void LAPACK::zgesdd(int m, int n, magmaDoubleComplex *d_A, double *S)
{
	magma_init();
	
	magmaDoubleComplex *U, *VT;
	magma_int_t info;
	magmaDoubleComplex *h_work;
	magma_int_t lwork;
	double * rwork;
	magma_int_t *iwork;	

	magma_zmalloc_cpu(&U,m*m);
	magma_zmalloc_cpu(&VT,n*n);
	magma_int_t nb = magma_get_zgesvd_nb(m,n);
	int min_mn = std::min(m,n);
	lwork = 2*min_mn*(1+nb);
	magma_zmalloc_pinned(&h_work,lwork);
	magma_dmalloc_pinned(&rwork,5*min_mn);
	magma_imalloc_pinned(&iwork,8*min_mn);

	magma_zgesdd(MagmaNoVec,m,n,d_A,m,S,U,m,VT,n,h_work,lwork,rwork,iwork,&info);

	free(U);
	free(VT);
	magma_free_pinned(h_work);
	magma_free_pinned(rwork);
	magma_free_pinned(iwork);	
	assert(info==0);

	//magma_finalize();
}

void LAPACKf77::dsyevd(const int m, double *A, double *W)
{
	double * h_work;
	magma_int_t lwork;
	magma_int_t *iwork;
	magma_int_t liwork;
	magma_int_t info, minus_one = -1;

	double aux_work[1];
	magma_int_t aux_iwork[1];
	lapackf77_dsyevd("V","L",&m,A,&m,W,aux_work,&minus_one,aux_iwork,&minus_one,&info);	
	lwork = (magma_int_t)aux_work[0];
	liwork = aux_iwork[0];
	iwork = (magma_int_t*)malloc(liwork*sizeof(magma_int_t));
	magma_dmalloc_cpu(&h_work,lwork);

	lapackf77_dsyevd("V","L",&m,A,&m,W,h_work,&lwork,iwork,&liwork,&info);

	free(h_work);
	free(iwork);
	assert(info==0);
}

void LAPACKf77::zheevd(const int m, magmaDoubleComplex *A, double *W)
{
	magmaDoubleComplex * h_work;
	magma_int_t lwork;
	magma_int_t *iwork;
	magma_int_t liwork;
    double * rwork;
    magma_int_t lrwork; 
	magma_int_t info, minus_one = -1;

	magmaDoubleComplex aux_work[1];
    double aux_rwork[1];
	magma_int_t aux_iwork[1];
	lapackf77_zheevd("V","L",&m,A,&m,W,aux_work,&minus_one,aux_rwork,&minus_one,aux_iwork,&minus_one,&info);	
	lwork = (magma_int_t)MAGMA_Z_REAL(aux_work[0]);
	liwork = aux_iwork[0];
	iwork = (magma_int_t*)malloc(liwork*sizeof(magma_int_t));
	magma_zmalloc_cpu(&h_work,lwork);
    lrwork = (magma_int_t)aux_rwork[0];
    magma_dmalloc_cpu(&rwork,lrwork);

	lapackf77_zheevd("V","L",&m,A,&m,W,h_work,&lwork,rwork,&lrwork,iwork,&liwork,&info);

	free(h_work);
	free(iwork);
    free(rwork);
	assert(info==0);
}

void LAPACKf77::dgeev(const int m, double *A, double *Wr, double *Wi)
{
    double *h_work, *VR, *VL;
    magma_int_t nb, info, lwork, m2=m*m;

    nb = magma_get_dgehrd_nb(m);
    lwork = m*(2+nb);
    lwork = std::max(lwork,m*(5+2*m));

    magma_dmalloc_cpu(&VL,m2);
    magma_dmalloc_cpu(&VR,m2);
    magma_dmalloc_cpu(&h_work,lwork);

    lapackf77_dgeev("N","V",&m,A,&m,Wr,Wi,VL,&m,VR,&m,h_work,&lwork,&info);

    free(h_work);
    free(VR);
    free(VL);
	assert(info==0);
}

void LAPACKf77::zgesdd(int m, int n, magmaDoubleComplex *A, double *S)
{
	magmaDoubleComplex *U, *VT;
	magma_int_t info;
	magmaDoubleComplex *h_work;
	magma_int_t lwork;
	double * rwork;
	magma_int_t *iwork;	

	magma_zmalloc_cpu(&U,m*m);
	magma_zmalloc_cpu(&VT,n*n);
	magma_int_t nb = magma_get_zgesvd_nb(m,n);
	int min_mn = std::min(m,n);
	lwork = 2*min_mn*(1+nb);
	magma_zmalloc_pinned(&h_work,lwork);
	magma_dmalloc_pinned(&rwork,5*min_mn);
	magma_imalloc_pinned(&iwork,8*min_mn);

	lapackf77_zgesdd("N",&m,&n,A,&m,S,U,&m,VT,&n,h_work,&lwork,rwork,iwork,&info);

	free(U);
	free(VT);
	magma_free_pinned(h_work);
	magma_free_pinned(rwork);
	magma_free_pinned(iwork);	
	assert(info==0);
}

void LAPACKf77::dgesdd(int m, int n, double *A, double *S)
{
	double *U, *VT;
	magma_int_t info;
	double *h_work;
	magma_int_t lwork;
	magma_int_t *iwork;	

	magma_dmalloc_cpu(&U,m*m);
	magma_dmalloc_cpu(&VT,n*n);
	magma_int_t nb = magma_get_dgesvd_nb(m,n);
	int min_mn = std::min(m,n);
	lwork = 2*min_mn*(1+nb);
	magma_dmalloc_pinned(&h_work,lwork);
	magma_imalloc_pinned(&iwork,8*min_mn);

	lapackf77_dgesdd("N",&m,&n,A,&m,S,U,&m,VT,&n,h_work,&lwork,iwork,&info);

	free(U);
	free(VT);
	magma_free_pinned(h_work);
	magma_free_pinned(iwork);	
	assert(info==0);
}

void LAPACKf77::dsted(int n_, double *d, double *e, double *m, double *z)
{
	magma_int_t info, lwork=-1, liwork=-1;
	magma_int_t n = n_;
	double tmp1[1];
	magma_int_t tmp2[1];

	double *d_h, *e_h;
	magma_dmalloc_cpu(&d_h,n);  
	magma_dmalloc_cpu(&e_h,n-1);  

	for(int i=0;i<n_;i++) d_h[i] = d[i];
	for(int i=0;i<n_-1;i++) e_h[i] = e[i];

	lapackf77_dstedc("I",&n,d_h,e_h,z,&n,tmp1,&lwork,tmp2,&liwork,&info);
	lwork = (magma_int_t)tmp1[0]; 
	liwork = tmp2[0]; 

	double *work; magma_int_t *iwork;
	magma_dmalloc_cpu(&work,lwork);
	iwork = (magma_int_t*)malloc(liwork*sizeof(magma_int_t));
	lapackf77_dstedc("I",&n,d_h,e_h,z,&n,work,&lwork,iwork,&liwork,&info);
	
	for(int i=0;i<n_;i++) m[i] = d_h[i];

	free(d_h);
	free(e_h);
	free(work);
	free(iwork);
	assert(info==0);
}

#endif








