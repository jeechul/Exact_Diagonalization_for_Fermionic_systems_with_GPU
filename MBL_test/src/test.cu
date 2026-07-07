#include <iostream>
#include <cmath>
#include <thrust/complex.h>
#include "../include/fermi_operator.h"

#include "../lieb_test/operator_algebra.h"
#include "../lieb_test/fermionic_operator.h"
#include "../lieb_test/basisLin.h"

typedef LinearOperator<Fop,double> LOP;

void add_hop(LOP& H,
        const double& hij, const int& i, const int& j)
{
    H.insert(hij,Fop('+',i),Fop('-',j));
    H.insert(hij,Fop('+',j),Fop('-',i));
}

void add_current(LOP& H,
        const double& t, const int& i, const int& j)
{
    H.insert(t,Fop('+',i),Fop('-',j));
    H.insert(-t,Fop('+',j),Fop('-',i));
}

void Periodic_Squared_3by3_ref(LOP& H)
{
	double t = -1.0;

	add_hop(H,t,0,1);
	add_hop(H,t,1,2);
	add_hop(H,t,2,0);
	add_hop(H,t,3,4);
	add_hop(H,t,4,5);
	add_hop(H,t,5,3);
	add_hop(H,t,6,7);
	add_hop(H,t,7,8);
	add_hop(H,t,8,6);

	add_hop(H,t,0,3);
	add_hop(H,t,3,6);
	add_hop(H,t,6,0);
	add_hop(H,t,1,4);
	add_hop(H,t,4,7);
	add_hop(H,t,7,1);
	add_hop(H,t,2,5);
	add_hop(H,t,5,8);
	add_hop(H,t,8,2);
}

void Periodic_Squared_3by3_J_ref(LOP& H)
{
	double t = 1.0;

	add_current(H,t,0,1);
	add_current(H,t,1,2);
	add_current(H,t,2,0);
	add_current(H,t,3,4);
	add_current(H,t,4,5);
	add_current(H,t,5,3);
	add_current(H,t,6,7);
	add_current(H,t,7,8);
	add_current(H,t,8,6);
}

void Periodic_Squared_3by3(H_TBq &Hq)
{
	int t = -1.0;

	Hq.Make_H(0,1,t);
	Hq.Make_H(1,2,t);
	Hq.Make_H(2,0,t);
	Hq.Make_H(3,4,t);
	Hq.Make_H(4,5,t);
	Hq.Make_H(5,3,t);
	Hq.Make_H(6,7,t);
	Hq.Make_H(7,8,t);
	Hq.Make_H(8,6,t);

	Hq.Make_H(0,3,t);
	Hq.Make_H(3,6,t);
	Hq.Make_H(6,0,t);
	Hq.Make_H(1,4,t);
	Hq.Make_H(4,7,t);
	Hq.Make_H(7,1,t);
	Hq.Make_H(2,5,t);
	Hq.Make_H(5,8,t);
	Hq.Make_H(8,2,t);
}

void Periodic_Squared_3by3_J(H_TBq &Hq)
{
	int t = 1.0;

	Hq.Make_J(0,1,t);
	Hq.Make_J(1,2,t);
	Hq.Make_J(2,0,t);
	Hq.Make_J(3,4,t);
	Hq.Make_J(4,5,t);
	Hq.Make_J(5,3,t);
	Hq.Make_J(6,7,t);
	Hq.Make_J(7,8,t);
	Hq.Make_J(8,6,t);
}

int main(int argc, char * argv[])
{
	const int N = atoi(argv[1]);
	const int Q = atoi(argv[2]);

	//-------- My operator -----------//

	H_TBq Hq(N,Q);

	Periodic_Squared_3by3_J(Hq);
	
	std::cout<<"My Operator : "<<std::endl;

	view_mat(&(Hq.Ax)[0],Hq.count,z*Q+1); std::cout<<std::endl;
	view_mat(&(Hq.Aj)[0],Hq.count,z*Q+1);

	//--------- Reference ------------//

	BasisLin basis(N,Q,Q);	
	
	LOP H;
	Periodic_Squared_3by3_J_ref(H);	

	thrust::host_vector<double> Ax_ref(Hq.count*(z*Q+1),0);
	thrust::host_vector<int> Aj_ref(Hq.count*(z*Q+1),0);

	MatrixOperator<LOP,BasisBit> M(H,basis.up_basis());
	M.SPwrite(&Ax_ref[0],&Aj_ref[0],z);	

	std::cout<<"Reference : "<<std::endl;

	view_mat(&Ax_ref[0],Hq.count,z*Q+1); std::cout<<std::endl;
	view_mat(&Aj_ref[0],Hq.count,z*Q+1);

	return 0;
}




