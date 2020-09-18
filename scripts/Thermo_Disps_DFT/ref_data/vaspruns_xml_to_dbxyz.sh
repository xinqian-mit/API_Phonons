#!/bin/bash
vaspruns_dir="./vaspruns"
xml_list_file="vaspruns_list.txt"
config_type="NaCl_snaps300K"
config_name_prefix="NaCl_snaps300K_"
xyz_file="$config_type.xyz"
vasp_scf_xmls_to_dbxyz="/home/proteus/Softwares/vaspxml_2_dbxyz/bin/Release/vaspxml_2_dbxyz"

current_dir=$PWD

cd $vaspruns_dir

for i in *.xml
do
   echo $i 
done > $xml_list_file  # build the list of vaspruns

mv $xml_list_file $current_dir
cd $current_dir


cat> CONTROL <<EOF
xml_list_file= $xml_list_file
vaspruns_dir= $vaspruns_dir
xyz_file= $xyz_file
config_type= $config_type
config_name_prefix= $config_name_prefix
dump_intv= 1
if_stress= 1
if_DFSET= 1
SPOSCAR= ./SPOSCAR
EOF

$vasp_scf_xmls_to_dbxyz
