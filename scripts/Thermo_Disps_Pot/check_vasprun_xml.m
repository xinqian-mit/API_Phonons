function [ xml_error_free ] = check_vasprun_xml( filename )

fid=fopen(filename,'r');
textstr = fileread(filename);
xml_error_free=0; % contains error

ifmodel_end=strfind(textstr,'</modeling>');
if ifmodel_end
    xml_error_free=1;
end

end

