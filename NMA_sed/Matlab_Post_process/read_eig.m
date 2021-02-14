function [ Freq_ks ] = read_eig( eigs_file,cminv2units )
fid = fopen(eigs_file);

textLine = fgets(fid); % The first line is number of basis in unit cell.
nbasis=sscanf(textLine, '%g ');


for iline=1:nbasis;
    textLine = fgets(fid);
end

textLine= fgets(fid);
Nk=sscanf(textLine,'%g');

textLine=fgets(fid);
Nbrch=sscanf(textLine,'%g');

Freq_ks=zeros(Nk,Nbrch);
for ik= 1:Nk
    textLine=fgets(fid); % Kpoint indication
    for ibrch =1 : Nbrch
        textLine=fgets(fid); % Mode indication
        textLine=fgets(fid);
        Freq_cminv=sscanf(textLine,'%g');
        Freq_ks(ik,ibrch)=cminv2units*Freq_cminv;
        for ib=1:nbasis
           textLine=fgets(fid); 
        end
    end
end
fclose(fid);


end

