#include<iostream>
#include<fstream>
#include<complex>
#include<cmath>
#include<string>
#include<numeric>
#include<omp.h>
#include<stdlib.h>
#include<fftw3.h>

using namespace std;
// Golobal constants
const double PI=3.141592653589793;
//const double cminv2Thz=0.06424063631; //Ar LJ frequency. with epsilon=1.67e-21J, sigma=3.4e-10m in Lammps
const double cminv2Thz=0.0299793; // for converting to THz;
const int DIM=3;
const complex<double> im(0,1);
const int V_heat=9; //number of lines in the Rvel.dat dumpped from lammps.
// Functions
complex<double> **** read_eig(char*filename,int &Nk,int &Nbrch, int &Nb, double ** &K,double **&Freq);
int ** read_info(char *filename,double *&mb,int &Nb,int &Natyp,int *&Natom_atyp,int &NSegs,double &dt_dump,int &Ndump,int &Nkcalc);
double **read_r0(char *pos_file,int &Natom);


void  project_Nmodes_kinetic(char *Rvel_file,complex<double> ****Eigs,double **K,\
                     int **kscalc_index,int Nkscalc,int Ndump,double *mb,int Nb,int Natom,\
                     double **R0,complex<double> **&vqks);
double **calc_SED_NMA(complex<double> **vqks,double dt_dump, int Nkscalc,int Ndump, int NSegs, double &Ws, int &M);
complex<double> * FourierTrans(double Ts, int N, complex<double> *ft, double &Ws, int  &M);



void write_vec(string vec_file, double *Vec, int vec_len, double dump_intv);
void write_vec(string vec_file, complex<double> *Vec, int vec_len, double dump_intv);
//All allocates here already initialized the values to be zero.

template <typename T> string num2str ( T Number );
template<typename T> void allocate(T **&data, int m, int n);
template<typename T> void allocate(T ***&data, int m, int n,int k);
template<typename T> void allocate(T ****&data, int m, int n,int o,int p);
template<typename T> void deallocate(T **&data, int m, int n);
template<typename T> void deallocate(T ***&data, int m, int n,int k);
template<typename T> void deallocate(T ****&data, int m, int n,int o,int p);
template<typename T1, typename T2> double dot_product(T1 *u, T2 *v, int n);
template<typename T1, typename T2> complex<double> complx_dot_product(T1 *u, T2 *v, int n);


//char *info_file="info_job1.dat";
const int V_head=9; //number of rows for velocity files as head
const int maxlen_filename=500;

char pos0_file[maxlen_filename];
char R_vel_file[maxlen_filename];
char tau_file[maxlen_filename];
char eig_file[maxlen_filename];
double *mb=NULL;


// ------------------------ main program start -------------------------------------------------------------------------------------//

int main(int argc, char *argv[]){
    int Nk,Nkscalc,Nbrch,Nb,Nb2,Natom,Natyp,Ndump,NSegs;
    double dt_dump;
    double Ws;
    int M;
    double **K_full=NULL;
    int ** kscalc_index=NULL;
    double **Freqcminv=NULL;
    double **R0=NULL;


    int *Natom_atyp=NULL;

    complex<double> ****Eigs=NULL;
    complex<double> **vqks=NULL;
    double **SED = NULL;

    kscalc_index = read_info("CONTROL_job1.in",mb,Nb,Natyp,Natom_atyp,NSegs,dt_dump,Ndump,Nkscalc);
    //kscalc_index=read_info(argv[1],mb,Nb,Natyp,Natom_atyp,NSegs,dt_dump,Ndump,Nkscalc); // number of k points calculated. argv[1]
    //char *ef="Ar_conv.eig";

    Eigs=read_eig(eig_file,Nk,Nbrch,Nb2,K_full,Freqcminv);

    if (Nb!=Nb2){
        cout<<"info file does not match with Eigenvector file"<<endl;
        exit(10);
    }

    R0=read_r0(pos0_file,Natom);
    cout<<"Velocity file has total "<<Ndump<<" steps."<<endl;
    cout<<"Cutting to "<<NSegs<<" segments for SED calculation."<<endl;
    project_Nmodes_kinetic(R_vel_file,Eigs,K_full,kscalc_index,Nkscalc,Ndump,mb,Nb,Natom,R0,vqks);

    cout<<"Computing SEDs for all normal modes..."<<endl;
    SED = calc_SED_NMA(vqks,dt_dump,Nkscalc,Ndump,NSegs,Ws,M);


    string SED_file[Nkscalc];
    string Prefix("SED_");
    string Suffix(".dat");
    string Mode("Kpoint");
    string Brch("_Brch");
    int ik,iv;

    for (int ikscalc=0;ikscalc<Nkscalc;ikscalc++)
    {
        ik=kscalc_index[ikscalc][0];iv=kscalc_index[ikscalc][1];
        SED_file[ikscalc] = Prefix+Mode+to_string(ik)+Brch+to_string(iv)+Suffix;
        write_vec(SED_file[ikscalc],SED[ikscalc],M,Ws);
    }


    cout<< "All job done !"<<endl;
    return 0;
}

// --------------------------------- main program end -------------------------------------------------------------------------------------//

int ** read_info(char *filename,double *&mb,int &Nb,int &Natyp,int *&Natom_atyp,int &Nsegs,double &dt_dump,int &Ndump,int &Nkscalc)
{
	int **ks_index=NULL;

	cout<<"reading information of MD run: "<<filename<<endl;
	ifstream fid(filename);
	if(!fid){
		cout<<"infomation file "<<filename<<" doesn't exit"<<endl;
		exit(1);
	}
    string buff;
    //fid>>buff;
    //fid>>tau_file;
    fid>>buff;
    fid>>pos0_file;
    fid>>buff;
    fid>>R_vel_file;
    fid>>buff;
    fid>>eig_file;
    fid>>buff>>Nb;
    fid>>buff>>Natyp;
    /*Natom_atyp= new int [Natyp];
    fid>>buff;
    for(int iatyp=0;iatyp<Natyp;iatyp++)
    {
    	fid>>Natom_atyp[iatyp];
    }*/

    mb=new double[Natyp];
    //double *tempmb=new double[Natyp];

    fid>>buff; //fscanf(fid,"%s",buff);
    for(int iat=0; iat<Natyp; iat++)
    {
        fid>>mb[iat];//fscanf(fid,"%lf",&tempmb[ie]);
    }
    int iatyplo=0;
    int iatyphi=0;

    fid>>buff>>Nsegs;//	fscanf(fid,"%s %d",buff,&dump_len);
    fid>>buff>>dt_dump;//fscanf(fid,"%s %lf",buff,&dump_intv);//dump interval, in picoseconds
    fid>>buff>>Ndump;//fscanf(fid,"%s %lf",buff,&max_freq);
    fid>>buff>>Nkscalc;//fscanf(fid,"%s %d",buff,&Nk);

    allocate(ks_index,Nkscalc,2);// k index and branch index s.
    for(int ikscalc=0; ikscalc<Nkscalc; ikscalc++) fid>>ks_index[ikscalc][0]>>ks_index[ikscalc][1];

    fid.close();
    //delete[] tempmb;
    return ks_index;
}



complex<double> **** read_eig(char*filename,int &Nk,int &Nbrch, int &Nb, double ** &K,double **&Freqcminv){
// This function is written to read the eigenvector file from gulp.
    string buff;
    int i,ik,imode,ib,ibrch;
    cout<<"reading eigenvectors with specified k points "<<filename<<endl;
    ifstream fid(filename);
    if (!fid){
        cout<<"eigenvector file"<<filename<<" doesn't exist"<<endl;
        exit(1);
    }
    fid>>Nb;
    //cout<<Nb<<endl;
    for(i=0;i<Nb;i++){
        fid>>buff>>buff>>buff>>buff; // the first several lines are coordinates of the unitcell.
    }
    fid>>Nk; //Number k points calcualted
    fid>>Nbrch; // number of branches at each point.
    //cout<<Nk<<" "<<Nbrch<<endl;
    double ** Reig=NULL;
    double ** Ieig=NULL;
    complex<double> **** Eigs=NULL;


    // Allocate the eigen vector list.
    allocate(Eigs,Nk,Nbrch,Nb,DIM);

    allocate(K,Nk,DIM);
    allocate(Freqcminv,Nk,Nbrch);
    allocate(Reig,Nb,DIM);
    allocate(Ieig,Nb,DIM);

    for(ik=0;ik<Nk;ik++){
        fid>>buff>>buff>>buff>>K[ik][0]>>K[ik][1]>>K[ik][2]>>buff>>buff; // ik  is the index of k points specified in gulp.
        for (ibrch=0;ibrch<Nb*3;ibrch++){

            fid>>buff>>imode;
            if(imode-1 != ibrch){
                cout<<"reading format might be wrong"<<endl;
                exit(2);
            }
            fid>>Freqcminv[ik][ibrch];
            for(ib=0;ib<Nb;ib++){
                if (K[ik][0]==0.0 && K[ik][1]==0.0 && K[ik][2]==0.0){
                    fid>>Reig[ib][0]>>Reig[ib][1]>>Reig[ib][2];
                    }
                else{
                    fid>>Reig[ib][0]>>Reig[ib][1]>>Reig[ib][2]>>Ieig[ib][0]>>Ieig[ib][1]>>Ieig[ib][2];
                }
                for(int idim=0;idim<DIM;idim++){
                    Eigs[ik][ibrch][ib][idim]= Reig[ib][idim]+im*Ieig[ib][idim];
                }
            }
        }
    }
    fid.close();
    return Eigs;

}






double ** read_r0(char *pos_file,int &Natom){
	cout<<"reading position coordinates: "<<pos_file<<endl;
	double **R0=NULL;
	ifstream fid(pos_file);
	if(!fid){
		cout<<"position_file: "<<pos_file<<" doesn't exit"<<endl;
		exit(1);
	}
	int i=0;
	fid>>Natom;
	R0=new double *[Natom];
	while(fid.eof()!=1){
		R0[i]=new double[2*DIM+2];
		for(int j=0; j<2*DIM+2;j++)
		{
		fid>>R0[i][j];
		}
		i=i+1;
	}
	fid.close();

	if((i-1)!=Natom){
		cout<<"Number of atoms are incorrect"<<endl;

	    exit(1);
	}
    return R0;
}


void  project_Nmodes_kinetic(char *Rvel_file,complex<double> ****Eigs,double **K,\
                     int **kscalc_index,int Nkscalc,int Ndump,double *mb,int Nb,int Natom,\
                     double **R0,complex<double> **&vqks)
{
    cout<<"Reading pos and vel trajectories and Projecting onto normal modes ..."<<endl;
    string buff;

    double x,y,z;

    double **vel;

    double omega,dotKR0=0;


    allocate(vel,Natom,DIM);

    allocate(vqks,Nkscalc,Ndump);

    int it,ikscalc,ib,ik,iv,iatyp;
    complex<double> pos_proj_NM(0,0),vel_proj_NM(0,0);
    complex <double> pix2(2*PI,0),tempT(0,0),tempU(0,0);
    ifstream fid(Rvel_file);
    if(!fid)
    {
        cout<<"Rvel_file: "<<Rvel_file<<" doesn't exit"<<endl;
        exit(1);
    }



    for ( it=0;it<Ndump;it++){
            for (int jhead=0; jhead<V_head; jhead++)
            {
                getline(fid,buff);


            } //read the heat at each dump step.



            for (int iat=0; iat<Natom; iat++) // Read the atomic postion and velocities at each time step.
            {
                fid>>x>>y>>z>>vel[iat][0]>>vel[iat][1]>>vel[iat][2]; //read trajectory.
                fid.ignore(200, '\n');
                ib=(int) (R0[iat][2*DIM]+0.5)-1;// basis index.
                iatyp = (int) (R0[iat][2*DIM+1])-1;// basis index.


                for (ikscalc=0; ikscalc<Nkscalc; ikscalc++) //projected to the normal modes.
                {

                    ik=kscalc_index[ikscalc][0];
                    iv=kscalc_index[ikscalc][1];

                    //pos_proj_NM=complx_dot_product(Eigs[ik][iv][ib],pos_traj[iat],DIM);
                    vel_proj_NM=complx_dot_product(Eigs[ik][iv][ib],vel[iat],DIM);

                    dotKR0=K[ik][0]*R0[iat][3]+K[ik][1]*R0[iat][4]+K[ik][2]*R0[iat][5];

                    vqks[ikscalc][it]=vqks[ikscalc][it]+sqrt(mb[iatyp]/Natom)*exp(im*pix2*dotKR0)*vel_proj_NM;


                }// loop ikscalc

            }//loop iat



}


    deallocate(vel,Natom,DIM);


}

double **calc_SED_NMA(complex<double> **vqks,double dt_dump, int Nkscalc,int Ndump, int NSegs, double &Ws, int &M)
{

    double ** SED;
    int N_perseg = Ndump/NSegs;
    int M0 = (N_perseg+1)/2;


    complex<double> * vks_seg;
    complex<double> **Fvqks_seg;
    Fvqks_seg = new complex<double> *[NSegs];

    vks_seg = new  complex<double> [N_perseg];


    for (int ikscalc=0; ikscalc<Nkscalc; ikscalc++)
    {
    	for (int iseg=0;iseg<NSegs;iseg++)
    	{
            for (int j=0; j<N_perseg; j++) vks_seg[j]=0.0;


    	    for (int j = 0; j<N_perseg; j++)
    	    {
    	        vks_seg[j] = vqks[ikscalc][iseg*N_perseg+j]; // iseg = Nsegs, (Nsegs-1)*N_perseg +N_perseg -1 = Nsegs*N_perseg -1 = Ndump-1
    	    }



    	    Fvqks_seg[iseg] = FourierTrans(dt_dump,N_perseg,vks_seg,Ws,M);

    	    if (ikscalc==0 && iseg==0) allocate(SED,Nkscalc,M);

    	    if (M != M0)
    	    {
    	    	cout<<"Inconsistent length of Fourier transform"<<endl;
    	    	exit(1);
    	    }



    	    for (int m=0;m<M; m++)
    	    {
    	        SED[ikscalc][m] += real(conj(Fvqks_seg[iseg][m])*Fvqks_seg[iseg][m])/NSegs;
    	    }





    	}

    }

    return SED;

}

complex<double> * FourierTrans(double Ts, int N, complex<double> *ft, double &Ws, int  &M)
{

    Ws = 2.0*PI/N/Ts;
    M = (N+1)/2;

    complex<double> *Fw;
    Fw = new complex<double> [M];
    fftw_complex *in, *out;
    fftw_plan p;

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    in=reinterpret_cast<fftw_complex*>(ft);
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    for (int n=0; n<M; n++)
    {

        Fw[n]=(out[n][0]+im*out[n][1])*Ts*exp(im*(n*Ws)*((N-1)*Ts));
    }

    //fftw_destroy_plan(p);
    //fftw_free(in);
    //fftw_free(out);

    return Fw;

}


void write_vec(string vec_file, double *Vec, int vec_len, double dt_dump)
{
ofstream fid(vec_file.c_str());
double t;

for (int it=0;it<vec_len;it++)
{
    t=it*dt_dump;
    fid<<t<<"\t"<<Vec[it]<<endl;

}

fid.close();

} // This function is written to ouput the Eks_acf decay as time of a specific mode, and only for debug


void write_vec(string vec_file, complex<double> *Vec, int vec_len, double dump_intv)
{
ofstream fid(vec_file.c_str());
double t;

for (int it=0;it<vec_len;it++)
{
    t=it*dump_intv;
    fid<<t<<"\t"<<real(Vec[it])<<"\t"<<imag(Vec[it])<<endl;

}

fid.close();

}

// --------------------------------------------------------  Calculation associated codes end here -------------------------------------------------------------------------//
template <typename T> string num2str ( T Number )
{
	stringstream ss;
	ss << Number;
	return ss.str();
}



// Allocation/deallocation of multidimensional arrays.
template<typename T>void allocate(T **&data, int m,int n)
{
	data=new T*[m];
	for (int i=0;i<m;i++) {
		data[i]=new T[n];
	     for (int j=0;j<n;j++)  data[i][j]=0;
	}
}

template<typename T>void allocate(T***&data, int m,int n,int k)
{
	data=new T**[m];
	for (int i=0;i<m;i++){
		data[i]=new T *[n];
		for(int j=0;j<n;j++){
			data[i][j]=new T[k];
			for (int kk=0;kk<k;kk++) data[i][j][kk]=0;
		}
	}
}

template<typename T>void allocate(T****&data, int m,int n,int o,int p)
{
    data = new T *** [m];
    for (int i=0; i<m; i++){
        data[i] = new T **[n];
        for (int j=0; j<n; j++){
            data[i][j] = new T *[o];
            for (int k=0; k<o; k++){
                data[i][j][k]=new T [p];
                for (int l=0;l<p;l++) data[i][j][k][l]=0;
            }

        }
    }

}


template<typename T>void deallocate(T **&data, int m,int n)
{
	for(int i=0;i<m;i++){
		delete[] data[i];
	}
	delete[] data;
	data=NULL;
}

template<typename T>void deallocate(T ***&data, int m,int n,int k)
{
	for(int i=0;i<m;i++){
		for (int j =0;j<n;j++) delete[] data[i][j];
		delete[] data[i];
	}
	delete[] data;
	data=NULL;
}

template<typename T>void deallocate(T ****&data, int m, int n, int o,int p)
{
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            for (int k=0;k<o;k++){
                delete[] data[i][j][k];
            }
            delete[] data[i][j];
        }
        delete[] data[i];
    }
    delete[] data;
    data=NULL;
}

template<typename T1, typename T2> double dot_product(T1 *u, T2 *v, int n)
{
	double y=0;
	for(int i=0;i<n;i++){
        y=y+u[i]*v[i];
	}
	return y;
}

template<typename T1, typename T2> complex<double> complx_dot_product(T1 *u, T2 *v, int n)
{
    complex<double> y(0,0);
    for (int i=0; i<n;i++){
        y=y+conj(u[i])*v[i];

    }
    return y;
}

