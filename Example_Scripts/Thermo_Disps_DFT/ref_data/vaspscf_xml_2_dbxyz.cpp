#include<iostream>
#include<fstream>
#include<string>
#include<stdio.h>
#include<cstring>
#include<cmath>
#include <sstream>
#include <vector>

using namespace std;
const int N_max=400;
const int DIM=3;
const double eV2kbar=1602.1766208; // 1eV/A3=1602.1766208 kbar.
const double eV2Ry=0.073498810939358;
const double A2Bohr=1.889725989;


void read_CONTROL(string control_file,string &xml_list_file,string &config_type,string &config_name_prefix,string &xyz_file,string &config_name,\
                  string &vaspruns_dir, int &dump_intv, int &if_stress_flag, int &if_DFSET, string &sposcar);
double ** read_POSCAR(string filename,int &Natms,double **&latt_vec);
int check_vasprun_xml(string filename);
void vasprun_xml_to_xyz(string filename,string xyz_file,string config_name,string config_type,int dump_intv,int if_stress_flag,int if_DFSET,double **pos0);
void write_xyz(string xyz_file,string config_name,string config_type,int Natcell,string *elems,int *Z_elems,double ecutwfc,\
               double degauss,double **latt_vecs,int *kmesh,double **pos,double **forc_atms,int read_stress_begin,double **virial,\
               double energy,int dump_every,int if_stress_flag);
void write_elements_data();
void write_DFSET(string DFSET_file,int Natcell,double **latt_vecs,double **pos0,double **pos,double **forc_atms,int dump_every);
int ElemSym2Z(string ElemSymbol);
string mass2ElemSym(double mass);


template<typename T>void allocate(T **&data, int m,int n)
{
	data=new T*[m];
	for (int i=0;i<m;i++) {
		data[i]=new T[n];
	     for (int j=0;j<n;j++)  data[i][j]=0;
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



int main()
{
    int dump_intv,if_stress_flag,if_DFSET;
    int Natms;
    double **posr0=NULL;
    double **latt_vec0=NULL;

    string xml_list_file,config_type,config_name_prefix,xyz_file,config_name,vaspruns_dir,xml_i,dir_xml_i,w_mode,sposcar;
    read_CONTROL("CONTROL",xml_list_file,config_type,config_name_prefix,xyz_file,config_name,vaspruns_dir,dump_intv,if_stress_flag,if_DFSET,sposcar);
    if (if_DFSET) posr0=read_POSCAR(sposcar,Natms,latt_vec0);


    // loop over the list of vaspruns.

    ifstream list_fid(xml_list_file.c_str());
    int ivasprun=0,ivasprun_written=0;
    int xml_correct;

    while(getline(list_fid,xml_i))
    {
        dir_xml_i=vaspruns_dir+xml_i;
        xml_correct=check_vasprun_xml(dir_xml_i);



        if (!xml_correct)
        {
            cout<<xml_i<<" contains errors!"<<endl;
        }
        else
        {

            config_name=config_name_prefix+to_string(ivasprun+1);
            vasprun_xml_to_xyz(dir_xml_i,xyz_file,config_name,config_type,dump_intv,if_stress_flag,if_DFSET,posr0);
            ivasprun_written++;

        }
        ivasprun++;


    }
    cout<<"Program normal exit"<<endl;
    return 0;
}


void read_CONTROL(string control_file,string &xml_list_file,string &config_type,string &config_name_prefix,string &xyz_file,string &config_name,\
                  string &vaspruns_dir, int &dump_intv, int &if_stress_flag, int &if_DFSET, string &sposcar)
{
    string textLine;
    ifstream fid(control_file.c_str());
    char temp_str[N_max];
    if (!fid)
    {
        cout<<"information file "<<control_file<<" doesn't exit"<<endl;
        exit(1);
    }

    size_t if_xml_list = string::npos;
    size_t if_vaspruns_dir = string::npos;
    size_t if_xyz_file = string::npos;
    size_t if_config_name = string::npos;
    size_t if_config_type = string::npos;
    size_t if_dump_intv = string::npos;
    size_t if_stress_ind = string::npos;
    size_t if_DFSET_ind = string::npos;
    size_t if_sposcar_ind = string::npos;
    char last_dir_char;
    dump_intv=1;
    if_stress_flag=1;
    if_DFSET=0; // Whether output DFSET by alamode.
    sposcar="";

    while (getline(fid,textLine))
    {
        if_xml_list=textLine.find("xml_list_file=");
        if (if_xml_list!=string::npos)
        {
            sscanf(textLine.c_str(),"xml_list_file= %s",temp_str);
            xml_list_file=string(temp_str);
        }

        if_vaspruns_dir=textLine.find("vaspruns_dir=");
        if (if_vaspruns_dir!=string::npos)
        {
            sscanf(textLine.c_str(),"vaspruns_dir= %s",temp_str);
            vaspruns_dir=string(temp_str);

            last_dir_char=vaspruns_dir.back();
            if (last_dir_char!='/') vaspruns_dir=vaspruns_dir+"/"; //add  a slash if there is no slash at the end


        }

        if_xyz_file=textLine.find("xyz_file=");
        if (if_xyz_file!=string::npos)
        {
            sscanf(textLine.c_str(),"xyz_file= %s",temp_str);
            xyz_file=string(temp_str);
        }

        if_config_name=textLine.find("config_name_prefix=");
        if (if_config_name!=string::npos)
        {
            sscanf(textLine.c_str(),"config_name_prefix= %s",temp_str);
            config_name_prefix=string(temp_str);
        }

        if_config_type=textLine.find("config_type=");
        if (if_config_type!=string::npos)
        {
            sscanf(textLine.c_str(),"config_type= %s",temp_str);
            config_type=string(temp_str);
        }

        if_dump_intv = textLine.find("dump_intv=");
        if (if_dump_intv!=string::npos)
        {
            sscanf(textLine.c_str(),"dump_intv= %i",&dump_intv);
        }

        if_stress_ind = textLine.find("if_stress=");
        if(if_stress_ind!=string::npos)
        {
            sscanf(textLine.c_str(),"if_stress= %i",&if_stress_flag);
        }

        if_DFSET_ind = textLine.find("if_DFSET=");
        if(if_DFSET_ind!=string::npos)
        {
            sscanf(textLine.c_str(),"if_DFSET= %i",&if_DFSET);
        }

        if(if_DFSET)
        {
            if_sposcar_ind = textLine.find("SPOSCAR=");
            if(if_sposcar_ind!=string::npos)
            {
                sscanf(textLine.c_str(),"SPOSCAR= %s",temp_str);
                sposcar=string(temp_str); //sposcar is the poscar for equilibrium file.
            }


        }

    }

}

double ** read_POSCAR(string filename,int &Natms,double **&latt_vec)
{
    string textLine;
    ifstream fid(filename.c_str());
    double **posr0=NULL; //scaled
    //double **Latt0_vec=NULL;
    double aix,aiy,aiz;
    double scale; // overall scaling lattice constant.
    double xr0,yr0,zr0; //relative coordinates

    allocate(latt_vec,DIM,DIM);

    if(!fid)
    {
        cout<<"POSCAR file "<<filename<<" not found"<<endl;
        exit(1);
    }

    getline(fid,textLine); //read the first comment line.
    fid>>scale;
    fid.ignore(N_max, '\n');

    for(int i=0;i<DIM;i++)
    {
        fid>>aix>>aiy>>aiz;
        fid.ignore(N_max, '\n');
        latt_vec[i][0]=aix*scale;
        latt_vec[i][1]=aiy*scale;
        latt_vec[i][2]=aiz*scale;
    }

    getline(fid,textLine);

    // trim string white spaces
    size_t last = textLine.find_last_not_of(' ');
    size_t first = textLine.find_first_not_of(' ');
    textLine = textLine.substr(first, (last-first+1));

    // Find number of elements (atomic types by splitting string)
    string subs;
    istringstream iss(textLine);
    vector<string> result;
    for(string s;iss>>s;)
    {
        result.push_back(s);
    }
    int Neles=result.size(); //Number of elements

    int *Nats_ele = new int[Neles];
    for (int iele=0;iele<Neles;iele++) Nats_ele[iele]=0;

    // read number of atoms per element
    getline(fid,textLine);

    // trim string white spaces
    last = textLine.find_last_not_of(' ');
    first = textLine.find_first_not_of(' ');
    textLine = textLine.substr(first, (last-first+1));


    istringstream iss2(textLine);
    vector<string> result2;
    for(string s;iss2>>s;)
    {
        result2.push_back(s);
    }
    int Neles2=result2.size(); //Number of elements

    if (Neles!=Neles2)
    {
        cout<<"Number of element symbols not equal to the size of # of atoms each type"<<endl;
        exit(100);
    }

    istringstream iss3(textLine);

    Natms =0;
    for (int iele=0;iele<Neles;iele++)
    {
        iss3>>Nats_ele[iele];
        Natms+=Nats_ele[iele];
    }


    allocate(posr0,Natms,DIM);

    getline(fid,textLine); // Always write sposcar in fractional coordinate

    if(textLine.find("Direct")==string::npos)
    {
        cout<<"Looking for indicator \"Direct\", Please use fractional coordinates!"<<endl;
        exit(200);
    }

    for (int iat=0;iat<Natms;iat++)
    {
        fid>>xr0>>yr0>>zr0;
        fid.ignore(N_max, '\n');
        posr0[iat][0]=xr0;
        posr0[iat][1]=yr0;
        posr0[iat][2]=zr0;
    }


    fid.close();


    return posr0;


}


int check_vasprun_xml(string filename)
{
    string textLine;
    ifstream fid(filename.c_str());
    if (!fid)
    {
        cout<<"information file "<<filename<<" doesn't exit"<<endl;
        exit(1);
    }

    size_t if_model_end=string::npos;
    //size_t if_scf_end=string::npos;



    while (getline(fid,textLine))
    {
        if_model_end=textLine.find("</modeling>");
        if (if_model_end!=string::npos) return 1;

    }

    return 0;

}

void vasprun_xml_to_xyz(string filename,string xyz_file,string config_name_prefix,string config_type,int dump_intv,int if_stress_flag,int if_DFSET,double **posr0)
{
    double ecutwfc, degauss, energy, vol, **latt_vec, *mass_typ;
    int Natoms,Ntyps, *kmesh, *atyp_list, *Z_elems;
    int istep=0;

    double **pos,**forc_atms,**virial;
    double **posr; //reduced.

    string textLine,*elems,config_name;
    int lineCounter=0;

    ifstream fid(filename.c_str());
    if (!fid)
    {
        cout<<"information file "<<filename<<" doesn't exit"<<endl;
        exit(1);
    }
    size_t ifecutwfc=string::npos;
    size_t ifdegauss=string::npos;
    size_t ifatominfo=string::npos;
    size_t ifatominfo_end=string::npos;
    size_t ifNtyps=string::npos;
    size_t ifNatoms=string::npos;
    size_t ifatomtyps=string::npos;
    size_t ifelems_mass=string::npos;
    size_t ifbasis=string::npos;
    size_t ifkmesh=string::npos;
    size_t if_initpos=string::npos;
    size_t if_struct_end=string::npos;
    size_t ifpos=string::npos;
    size_t if_scf_calc=string::npos;
    size_t if_scf_calc_end=string::npos;
    size_t if_scstep_begin=string::npos;
    size_t if_scstep_end=string::npos;
    size_t if_vol=string::npos;
    size_t if_forc_atm=string::npos;
    size_t if_stress=string::npos;
    size_t if_e0=string::npos;

    int line_atominfo_begin=-1;
    int line_atominfo_end=-1;
    int line_atyplist=-1;
    int line_elems_mass=-1;
    int line_basis=-1;
    int line_kmesh=-1;
    int line_structure=-1;
    int line_structure_end=-1;
    int line_pos=-1;
    int line_scf_begin=-1;
    int line_scf_end=-1;
    int line_forc=-1;
    int line_stress=-1;
    int line_scstep_begin=-1;
    int line_scstep_end=-1;

    int iat,iatyp=0,idim,Nkx,Nky,Nkz;

    double mass_temp,ax,ay,az,x0,x1,x2,fx,fy,fz,sx,sy,sz;
    char elem_cstr[N_max];

    kmesh =new int[DIM];
    allocate(latt_vec,DIM,DIM);
    allocate(virial,DIM,DIM);

    while (getline(fid,textLine))
    {

        ifecutwfc=textLine.find("<i name=\"ENCUT\">");
        if (ifecutwfc!=string::npos)
        {
            sscanf(textLine.c_str(),"  <i name=\"ENCUT\">    %lg</i>",&ecutwfc);
        }

        ifdegauss=textLine.find("<i name=\"SIGMA\">");
        if (ifdegauss!=string::npos)
        {
            sscanf(textLine.c_str(),"  <i name=\"SIGMA\">      %lg</i>",&degauss);
        }

        ifkmesh=textLine.find("<kpoints>");
        if (ifkmesh!=string::npos)
        {
            line_kmesh=lineCounter+2;
        }

        if (lineCounter==line_kmesh)
        {
            sscanf(textLine.c_str(),"   <v type=\"int\" name=\"divisions\">       %i        %i        %i </v>",&Nkx,&Nky,&Nkz);
            kmesh[0]=Nkx; kmesh[1]=Nky; kmesh[2]=Nkz;
        }

        // block <atominfo>
        ifatominfo=textLine.find("<atominfo>");
        if (ifatominfo!=string::npos)
        {
            line_atominfo_begin=lineCounter;
            line_atominfo_end=-1;
        }
        if (line_atominfo_begin>0 && line_atominfo_end<0)
        {
            ifNatoms=textLine.find("<atoms>");
            if (ifNatoms!=string::npos)
            {
                sscanf(textLine.c_str(),"  <atoms>      %i </atoms>",&Natoms);
                atyp_list=new int [Natoms];
                elems = new string [Natoms];
                Z_elems= new int [Natoms];
                allocate(pos,Natoms,DIM);
                allocate(forc_atms,Natoms,DIM);
                allocate(posr,Natoms,DIM);

            }

            ifNtyps=textLine.find("<types>");
            if (ifNtyps!=string::npos)
            {
                sscanf(textLine.c_str(),"  <types>       %i </types>",&Ntyps);
                mass_typ=new double [Ntyps];
            }

            ifatomtyps=textLine.find("<field type=\"int\">atomtype</field>");
            if (ifatomtyps!=string::npos)
            {
                line_atyplist=lineCounter+2;
            }
            if (line_atyplist>0)
            {
                iat = lineCounter-line_atyplist;
                if (iat>=0 && iat<Natoms)
                {
                    sscanf(textLine.c_str(),"    <rc><c> %[^</c><c>]%*s %i</c></rc>",elem_cstr,&iatyp); // a field %*s is used to receive the delimiter.
                    atyp_list[iat]=iatyp-1; //convert to c++ indices which starts from 0.
                    elems[iat]=string(elem_cstr);
                }

            }


            ifelems_mass=textLine.find("<field type=\"string\">pseudopotential</field>");
            if (ifelems_mass!=string::npos) line_elems_mass=lineCounter+2;

            if (line_elems_mass>0)
            {
                iatyp=lineCounter-line_elems_mass;
                if (iatyp>=0 && iatyp<Ntyps)
                {
                    sscanf(textLine.c_str(),"    <rc><c>  %*i</c><c>%[^</c><c>]%*s    %lg</c><c>      %*i</c><c>  %*s %*s %*s                    </c></rc>",elem_cstr,&mass_temp);

                    mass_typ[iatyp]=mass_temp;
                }

            }

            ifatominfo_end=textLine.find("</atominfo>");
            if (ifatominfo_end!=string::npos)
            {
                line_atominfo_end=lineCounter;
                line_atominfo_begin=-1;
                for (iat=0; iat<Natoms; iat++)
                {
                    iatyp=atyp_list[iat];
                    elems[iat]=mass2ElemSym(mass_typ[iatyp]);
                    Z_elems[iat]=ElemSym2Z(elems[iat]);
                }
            }
        }
        // block <atominfo> end


        // read lattice vectors and positions. for scf runs, initial and final are the same.0.0007051



        if_scf_calc=textLine.find("<calculation>");

        if(if_scf_calc!=string::npos)
        {
            line_scf_begin=lineCounter;
            line_scf_end=-1;
        }

        if (line_scf_begin>0 && line_scf_end<0)
        {
            if_initpos=textLine.find("<structure>");
            if (if_initpos!=string::npos)
            {
                line_structure=lineCounter;
                line_structure_end=-1;
            }
            if (line_structure>0 && line_structure_end<0)
            {
                ifbasis=textLine.find("<varray name=\"basis\" >");
                if (ifbasis!=string::npos) line_basis=lineCounter+1;
                if (line_basis>0)
                {
                    idim=lineCounter-line_basis;
                    if (idim>=0 && idim<DIM)
                    {
                        sscanf(textLine.c_str(),"    <v>      %lg      %lg      %lg </v>",&ax,&ay,&az);
                        latt_vec[idim][0]=ax;
                        latt_vec[idim][1]=ay;
                        latt_vec[idim][2]=az;
                    }
                }

                if_vol=textLine.find("<i name=\"volume\">");
                if (if_vol!=string::npos)
                {
                    sscanf(textLine.c_str(),"    <i name=\"volume\">   %lg </i>",&vol);
                }


                ifpos=textLine.find("<varray name=\"positions\" >");
                if (ifpos!=string::npos)
                {
                    line_pos=lineCounter+1;
                }
                if (line_pos>0)
                {
                    iat=lineCounter-line_pos;
                    if (iat>=0 && iat<Natoms)
                    {
                        sscanf(textLine.c_str(),"   <v>      %lg      %lg      %lg </v>",&x0,&x1,&x2);
                        pos[iat][0]=x0*latt_vec[0][0]+x1*latt_vec[1][0]+x2*latt_vec[2][0];
                        pos[iat][1]=x0*latt_vec[0][1]+x1*latt_vec[1][1]+x2*latt_vec[2][1];
                        pos[iat][2]=x0*latt_vec[0][2]+x1*latt_vec[1][2]+x2*latt_vec[2][2];

                        if (if_DFSET)
                        {
                            posr[iat][0]=x0;
                            posr[iat][1]=x1;
                            posr[iat][2]=x2;
                        }

                    }
                }


                if_struct_end=textLine.find("</structure>");
                if (if_struct_end!=string::npos)
                {
                    line_structure_end=lineCounter;
                    line_structure=-1;
                }

            }

            if_scstep_begin=textLine.find("<scstep>");
            if (if_scstep_begin!=string::npos)
            {
                line_scstep_begin=lineCounter;
                line_scstep_end=-1;
            }

            if (line_scstep_begin>0 && line_scstep_end<0)
            {

                if_e0=textLine.find("<i name=\"e_0_energy\">");
                if (if_e0!=string::npos)
                {
                    sscanf(textLine.c_str(),"    <i name=\"e_0_energy\">   %lg </i>",&energy);
                }



                if_scstep_end=textLine.find("</scstep>");
                if (if_scstep_end!=string::npos)
                {
                    line_scstep_end=lineCounter;
                    line_scstep_begin=-1;
                }
            }





            if_forc_atm=textLine.find("<varray name=\"forces\" >");
            if (if_forc_atm!=string::npos)
            {
                line_forc=lineCounter+1;
            }

            if (line_forc>0)
            {
                iat=lineCounter-line_forc;
                if (iat>=0 && iat<Natoms)
                {
                    sscanf(textLine.c_str(),"   <v>      %lg      %lg      %lg </v>",&fx,&fy,&fz);
                    forc_atms[iat][0]=fx; forc_atms[iat][1]=fy; forc_atms[iat][2]=fz;
                }
            }


                if_stress=textLine.find("<varray name=\"stress\" >");
                if (if_stress!=string::npos)
                {
                    line_stress=lineCounter+1;
                }
                if (line_stress>0)
                {
                    idim=lineCounter-line_stress;

                    if (idim>=0 && idim<DIM)
                    {
                        sscanf(textLine.c_str(),"   <v>      %lg      %lg      %lg </v>",&sx,&sy,&sz); //in kbar
                        virial[idim][0]=sx/eV2kbar*vol;
                        virial[idim][1]=sy/eV2kbar*vol;
                        virial[idim][2]=sz/eV2kbar*vol;

                    }
                }






            if_scf_calc_end=textLine.find("</calculation>");

            if (if_scf_calc_end!=string::npos)
            {
                line_scf_end=lineCounter;
                line_scf_begin=-1;

                config_name=config_name_prefix+"_istep"+to_string(istep);

                write_xyz(xyz_file,config_name,config_type,Natoms,elems,Z_elems,ecutwfc,degauss,latt_vec,kmesh,pos,forc_atms,(line_stress>0),virial,energy,dump_intv,if_stress_flag);

                if (if_DFSET)
                {
                    write_DFSET("DFSET",Natoms,latt_vec,posr0,posr,forc_atms,dump_intv);
                }




                istep++;
            }






        } // <calculation> end.



        lineCounter+=1;

    }

    // after everything is read do a correction of element symbol.


    fid.close();

    deallocate(latt_vec,DIM,DIM); latt_vec=NULL;
    deallocate(virial,DIM,DIM); virial=NULL;
    deallocate(pos,Natoms,DIM); pos=NULL;
    deallocate(forc_atms,Natoms,DIM); forc_atms=NULL;
    delete[] mass_typ; mass_typ=NULL;
    delete[] kmesh; kmesh=NULL;
    delete[] atyp_list; atyp_list=NULL;
    delete[] Z_elems; Z_elems=NULL;
    delete[] elems; elems=NULL;

}


void write_DFSET(string DFSET_file,int Natcell,double **latt_vecs,double **posr0,double **posr,double **forc_atms,int dump_every)
{
    static int iconfig=0;
    string w_mode;
    double ux,uy,uz,fx,fy,fz;
    double urx,ury,urz,Urx,Ury,Urz;


    if (iconfig==0) w_mode="w";
    if (iconfig>0) w_mode="a+";

    if (iconfig%dump_every == 0)
    {
        FILE *fid;
        fid=fopen(DFSET_file.c_str(),w_mode.c_str());
        //fprintf(fid,"# CONFIG %i\n",iconfig);




        for (int i=0; i<Natcell; i++)
        {
            Urx = posr[i][0]-posr0[i][0];
            urx = Urx;
            if (abs(Urx+1.)<abs(Urx)) //pbc, wrap.
            {
                urx = Urx+1.;
            }
            if (abs(Urx-1.)<abs(Urx))
            {
                urx = Urx-1.;
            }

            Ury = posr[i][1]-posr0[i][1];
            ury = Ury;
            if (abs(Ury+1.)<abs(Ury))
            {
                ury = Ury+1.;
            }
            if (abs(Ury-1.)<abs(Ury))
            {
                ury = Ury-1.;
            }

            Urz = posr[i][2]-posr0[i][2];
            urz = Urz;
            if (abs(Urz+1.)<abs(Urz))
            {
                urz = Urz+1.;
            }
            if (abs(Urz-1.)<abs(Urz))
            {
                urz = Urz-1.;
            }

            ux = urx*latt_vecs[0][0]+ury*latt_vecs[1][0]+urz*latt_vecs[2][0];
            uy = urx*latt_vecs[0][1]+ury*latt_vecs[1][1]+urz*latt_vecs[2][1];
            uz = urx*latt_vecs[0][2]+ury*latt_vecs[1][2]+urz*latt_vecs[2][2];

            ux = ux;//*A2Bohr; No need to, phonopy corrected the units.
            uy = uy;//*A2Bohr;
            uz = uz;//*A2Bohr;

            fx = forc_atms[i][0];//*eV2Ry/A2Bohr;
            fy = forc_atms[i][1];//*eV2Ry/A2Bohr;
            fz = forc_atms[i][2];//*eV2Ry/A2Bohr;
            fprintf(fid,"% .9f % .9f % .9f % .9f % .9f % .9f\n",ux,uy,uz,fx,fy,fz);

        }

        fclose(fid);

    }


    iconfig++;

}


void write_xyz(string xyz_file,string config_name,string config_type,int Natcell,string *elems,int *Z_elems,double ecutwfc,\
               double degauss,double **latt_vecs,int *kmesh,double **pos,double **forc_atms,int read_stress_begin,double **virial,double energy,
               int dump_every,int if_stress_flag)
{
    static int iconfig=0;
    string w_mode;

    if (iconfig==0) w_mode="w";
    if (iconfig>0) w_mode="a+";

    if (iconfig%dump_every == 0)
    {
        FILE *fid;
        fid=fopen(xyz_file.c_str(),w_mode.c_str());
        fprintf(fid,"%i\n",Natcell);
        fprintf(fid,"Lattice=\"%.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\" ",\
        latt_vecs[0][0],latt_vecs[0][1],latt_vecs[0][2],\
        latt_vecs[1][0],latt_vecs[1][1],latt_vecs[1][2],\
        latt_vecs[2][0],latt_vecs[2][1],latt_vecs[2][2]);
        fprintf(fid,"Properties=species:S:1:pos:R:3:force:R:3:Z:I:1 ");
        fprintf(fid,"config_type=%s ",config_type.c_str());
        if (degauss>0) fprintf(fid,"degauss=%.12f ",degauss);
        fprintf(fid,"ecutwfc=%.8f ",ecutwfc);
        fprintf(fid,"pbc=\"T T T\" ");
        fprintf(fid,"kpoints=\"%i %i %i\" ",kmesh[0],kmesh[1],kmesh[2]);
        fprintf(fid,"energy=%.8f ",energy);
        fprintf(fid,"config_name=%s ",config_name.c_str());
        if (read_stress_begin>0 && if_stress_flag!=0)
        {
            fprintf(fid,"virial=\"%lg %lg %lg %lg %lg %lg %lg %lg %lg\"",\
            virial[0][0],virial[0][1],virial[0][2],\
            virial[1][0],virial[1][1],virial[1][2],\
            virial[2][0],virial[2][1],virial[2][2]);
        }

    fprintf(fid,"\n");



    for (int i=0; i<Natcell; i++)
    {
        fprintf(fid,"%2s      % .8f      % .8f      % .8f      % .8f      % .8f      % .8f      %i\n",elems[i].c_str(),pos[i][0],pos[i][1],pos[i][2],\
                forc_atms[i][0],forc_atms[i][1],forc_atms[i][2],Z_elems[i]);
    }

    fclose(fid);

    }


    iconfig++;

}

void write_elements_data()
{
    FILE *fid;
    fid=fopen("Elements.data","w");
    fprintf(fid,"%i\t%s\t%lg\n",1,"H",1.0079);
    fprintf(fid,"%i\t%s\t%lg\n",2,"He",4.0026);
    fprintf(fid,"%i\t%s\t%lg\n",3,"Li",6.941);
    fprintf(fid,"%i\t%s\t%lg\n",4,"Be",9.0122);
    fprintf(fid,"%i\t%s\t%lg\n",5,"B",10.811);
    fprintf(fid,"%i\t%s\t%lg\n",6,"C",12.0107);
    fprintf(fid,"%i\t%s\t%lg\n",7,"N",14.0067);
    fprintf(fid,"%i\t%s\t%lg\n",8,"O",15.9994);
    fprintf(fid,"%i\t%s\t%lg\n",9,"F",18.9984);
    fprintf(fid,"%i\t%s\t%lg\n",10,"Ne",20.1797);
    fprintf(fid,"%i\t%s\t%lg\n",11,"Na",22.9897);
    fprintf(fid,"%i\t%s\t%lg\n",12,"Mg",24.305);
    fprintf(fid,"%i\t%s\t%lg\n",13,"Al",26.9815);
    fprintf(fid,"%i\t%s\t%lg\n",14,"Si",28.0855);
    fprintf(fid,"%i\t%s\t%lg\n",15,"P",30.9738);
    fprintf(fid,"%i\t%s\t%lg\n",16,"S",32.065);
    fprintf(fid,"%i\t%s\t%lg\n",17,"Cl",35.453);
    fprintf(fid,"%i\t%s\t%lg\n",18,"Ar",39.948);
    fprintf(fid,"%i\t%s\t%lg\n",19,"K",39.0983);
    fprintf(fid,"%i\t%s\t%lg\n",20,"Ca",40.078);
    fprintf(fid,"%i\t%s\t%lg\n",21,"Sc",44.9559);
    fprintf(fid,"%i\t%s\t%lg\n",22,"Ti",47.867);
    fprintf(fid,"%i\t%s\t%lg\n",23,"V",50.9415);
    fprintf(fid,"%i\t%s\t%lg\n",24,"Cr",51.9961);
    fprintf(fid,"%i\t%s\t%lg\n",25,"Mn",54.938);
    fprintf(fid,"%i\t%s\t%lg\n",26,"Fe",55.845);
    fprintf(fid,"%i\t%s\t%lg\n",27,"Co",58.9332);
    fprintf(fid,"%i\t%s\t%lg\n",28,"Ni",58.6934);
    fprintf(fid,"%i\t%s\t%lg\n",29,"Cu",63.546);
    fprintf(fid,"%i\t%s\t%lg\n",30,"Zn",65.39);
    fprintf(fid,"%i\t%s\t%lg\n",31,"Ga",69.723);
    fprintf(fid,"%i\t%s\t%lg\n",32,"Ge",72.64);
    fprintf(fid,"%i\t%s\t%lg\n",33,"As",74.9216);
    fprintf(fid,"%i\t%s\t%lg\n",34,"Se",78.96);
    fprintf(fid,"%i\t%s\t%lg\n",35,"Br",79.904);
    fprintf(fid,"%i\t%s\t%lg\n",36,"Kr",83.8);
    fprintf(fid,"%i\t%s\t%lg\n",37,"Rb",85.4678);
    fprintf(fid,"%i\t%s\t%lg\n",38,"Sr",87.62);
    fprintf(fid,"%i\t%s\t%lg\n",39,"Y",88.9059);
    fprintf(fid,"%i\t%s\t%lg\n",40,"Zr",91.224);
    fprintf(fid,"%i\t%s\t%lg\n",41,"Nb",92.9064);
    fprintf(fid,"%i\t%s\t%lg\n",42,"Mo",95.94);
    fprintf(fid,"%i\t%s\t%lg\n",43,"Tc",98.);
    fprintf(fid,"%i\t%s\t%lg\n",44,"Ru",101.07);
    fprintf(fid,"%i\t%s\t%lg\n",45,"Rh",102.9055);
    fprintf(fid,"%i\t%s\t%lg\n",46,"Pd",106.42);
    fprintf(fid,"%i\t%s\t%lg\n",47,"Ag",107.8682);
    fprintf(fid,"%i\t%s\t%lg\n",48,"Cd",112.411);
    fprintf(fid,"%i\t%s\t%lg\n",49,"In",114.818);
    fprintf(fid,"%i\t%s\t%lg\n",50,"Sn",118.71);
    fprintf(fid,"%i\t%s\t%lg\n",51,"Sb",121.76);
    fprintf(fid,"%i\t%s\t%lg\n",52,"Te",127.6);
    fprintf(fid,"%i\t%s\t%lg\n",53,"I",126.9045);
    fprintf(fid,"%i\t%s\t%lg\n",54,"Xe",131.293);
    fprintf(fid,"%i\t%s\t%lg\n",55,"Cs",132.9055);
    fprintf(fid,"%i\t%s\t%lg\n",56,"Ba",137.327);
    fprintf(fid,"%i\t%s\t%lg\n",57,"La",138.9055);
    fprintf(fid,"%i\t%s\t%lg\n",58,"Ce",140.116);
    fprintf(fid,"%i\t%s\t%lg\n",59,"Pr",140.9077);
    fprintf(fid,"%i\t%s\t%lg\n",60,"Nd",144.24);
    fprintf(fid,"%i\t%s\t%lg\n",61,"Pm",145.);
    fprintf(fid,"%i\t%s\t%lg\n",62,"Sm",150.36);
    fprintf(fid,"%i\t%s\t%lg\n",63,"Eu",151.964);
    fprintf(fid,"%i\t%s\t%lg\n",64,"Gd",157.25);
    fprintf(fid,"%i\t%s\t%lg\n",65,"Tb",158.9253);
    fprintf(fid,"%i\t%s\t%lg\n",66,"Dy",162.5);
    fprintf(fid,"%i\t%s\t%lg\n",67,"Ho",164.9303);
    fprintf(fid,"%i\t%s\t%lg\n",68,"Er",167.259);
    fprintf(fid,"%i\t%s\t%lg\n",69,"Tm",168.9342);
    fprintf(fid,"%i\t%s\t%lg\n",70,"Yb",173.04);
    fprintf(fid,"%i\t%s\t%lg\n",71,"Lu",174.967);
    fprintf(fid,"%i\t%s\t%lg\n",72,"Hf",178.49);
    fprintf(fid,"%i\t%s\t%lg\n",73,"Ta",180.9479);
    fprintf(fid,"%i\t%s\t%lg\n",74,"W",183.84);
    fprintf(fid,"%i\t%s\t%lg\n",75,"Re",186.207);
    fprintf(fid,"%i\t%s\t%lg\n",76,"Os",190.23);
    fprintf(fid,"%i\t%s\t%lg\n",77,"Ir",192.217);
    fprintf(fid,"%i\t%s\t%lg\n",78,"Pt",195.078);
    fprintf(fid,"%i\t%s\t%lg\n",79,"Au",196.9665);
    fprintf(fid,"%i\t%s\t%lg\n",80,"Hg",200.59);
    fprintf(fid,"%i\t%s\t%lg\n",81,"Tl",204.3833);
    fprintf(fid,"%i\t%s\t%lg\n",82,"Pb",207.2);
    fprintf(fid,"%i\t%s\t%lg\n",83,"Bi",208.9804);
    fprintf(fid,"%i\t%s\t%lg\n",84,"Po",209.);
    fprintf(fid,"%i\t%s\t%lg\n",85,"At",210.);
    fprintf(fid,"%i\t%s\t%lg\n",86,"Rn",222.);
    fprintf(fid,"%i\t%s\t%lg\n",87,"Fr",223.);
    fprintf(fid,"%i\t%s\t%lg\n",88,"Ra",226.);
    fprintf(fid,"%i\t%s\t%lg\n",89,"Ac",227.);
    fprintf(fid,"%i\t%s\t%lg\n",90,"Th",232.0381);
    fprintf(fid,"%i\t%s\t%lg\n",91,"Pa",231.0359);
    fprintf(fid,"%i\t%s\t%lg\n",92,"U",238.0289);
    fprintf(fid,"%i\t%s\t%lg\n",93,"Np",237.);
    fprintf(fid,"%i\t%s\t%lg\n",94,"Pu",244.);
    fprintf(fid,"%i\t%s\t%lg\n",95,"Am",243.);
    fprintf(fid,"%i\t%s\t%lg\n",96,"Cm",247.);
    fprintf(fid,"%i\t%s\t%lg\n",97,"Bk",247.);
    fprintf(fid,"%i\t%s\t%lg\n",98,"Cf",251.);
    fprintf(fid,"%i\t%s\t%lg\n",99,"Es",252.);
    fprintf(fid,"%i\t%s\t%lg\n",100,"Fm",257.);
    fprintf(fid,"%i\t%s\t%lg\n",101,"Md",258.);
    fprintf(fid,"%i\t%s\t%lg\n",102,"No",259.);
    fprintf(fid,"%i\t%s\t%lg\n",103,"Lr",262.);
    fprintf(fid,"%i\t%s\t%lg\n",104,"Rf",261.);
    fprintf(fid,"%i\t%s\t%lg\n",105,"Db",262.);
    fprintf(fid,"%i\t%s\t%lg\n",106,"Sg",263.);
    fprintf(fid,"%i\t%s\t%lg\n",107,"Bh",262.);
    fprintf(fid,"%i\t%s\t%lg\n",108,"Hs",265.);
    fprintf(fid,"%i\t%s\t%lg\n",109,"Mt",266.);
    fprintf(fid,"%i\t%s\t%lg\n",110,"Ds",281.);
    fprintf(fid,"%i\t%s\t%lg\n",111,"Rg",272.);
    fprintf(fid,"%i\t%s\t%lg\n",112,"Cn",285.);
    fprintf(fid,"%i\t%s\t%lg\n",113,"Nh",284.);
    fprintf(fid,"%i\t%s\t%lg\n",114,"Fl",289.);
    fprintf(fid,"%i\t%s\t%lg\n",115,"Mc",288.);
    fprintf(fid,"%i\t%s\t%lg\n",116,"Lv",292.);
    fprintf(fid,"%i\t%s\t%lg\n",117,"Ts",293.);
    fprintf(fid,"%i\t%s\t%lg\n",118,"Og",294.);
    fclose(fid);


}


int ElemSym2Z(string ElemSymbol)
{
    write_elements_data();
    ifstream fid("Elements.data");
    if(!fid)
    {
        cout<<"Elements.data doesn't exit"<<endl;
        exit(1);
    }

    string textLine;
    string symbol;
    int Z=0;
    int Zelem=0;
    char Esym_cstr[N_max];

    while (getline(fid,textLine))
    {
        sscanf(textLine.c_str(),"%i\t%s\t%*g",&Z,Esym_cstr);
        if (!strcmp(ElemSymbol.c_str(),Esym_cstr))
        {
            Zelem=Z;
            remove("Elements.data");
            return Zelem;
        }
    }


    remove("Elements.data");
    return Zelem;

}

string mass2ElemSym(double mass)
{
    string textLine;
    double mass_elm,delem;
    write_elements_data();
    ifstream fid("Elements.data");
    if(!fid)
    {
        cout<<"Elements.data doesn't exit"<<endl;
        exit(1);
    }


    char Esym_cstr[N_max];

    while (getline(fid,textLine))
    {
        sscanf(textLine.c_str(),"%*i\t%s\t%lg",Esym_cstr,&mass_elm);
        delem=abs(mass-mass_elm);
        if (delem<0.5)
        {
            remove("Elements.data");
            return string(Esym_cstr);

        }
    }
    remove("Elements.data");
    return string(" ");

}

