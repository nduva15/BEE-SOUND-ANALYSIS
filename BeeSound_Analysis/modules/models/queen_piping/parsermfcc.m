function [descriptorMatrix_m, timeSegment_sec_m, name_c]=FfeatureModelLoader(fileName)
%
%
% function [descriptorMatrix_m,timeSegment_sec_m]=FfeatureModelLoader(fileName)
%
% DESCRIPTION:
% ============
% read a feature model file
%
%
% INPUTS:
% =======
% - fileName                    : name of the file to read
%
%
% OUTPUTS:
% ========
% - descriptorMatrix_m (nbFrame , vectorSize)    :  descriptor matrix
%
% - timeSegment_sec_m(nbFrame , 3)        : segemtation matrix
% [time deb , time middle, time end ] in seconds
%
% - name_c {1,vectorSize}                 : column description
% christophe.charbuillet@ircam.fr 20/04/2011





% ------------------------------------------------------------------------
% Open

 descriptorMatrix_m = [];
 timeSegment_sec_m    = [];

fid=fopen(fileName,'rb','native','UTF-8');
if (fid == -1)
    error(['can not open the file : ' fileName]);
    return
end

% ------------------------------------------------------------------------
% Header read



[multiheader, count]=fread(fid, 1, 'uint32');
if(count~=1),
    fclose(fid);
    error(['fread error on ' fileName ' : number of readen data = ' num2str(count) ' , ' num2str(1) ' expected' ]);
    return
end

[headerSize, count]=fread(fid, 1, 'uint32');
if(count~=1),
    fclose(fid);
    error(['fread error on ' fileName ' : number of readen data = ' num2str(count) ' , ' num2str(1) ' expected' ]);
    return
end


[vectorNumber, count]=fread(fid, 1, 'uint32');
if(count~=1),
    fclose(fid);
    error(['fread error on ' fileName ' : number of readen data = ' num2str(count) ' , ' num2str(1) ' expected' ]);
    return
end


[vectSize, count]=fread(fid, 1, 'uint32');
if(count~=1),
    fclose(fid);
    warning(['fread error on ' fileName ' : number of readen data = ' num2str(count) ' , ' num2str(1) ' expected' ]);
    return
end

% ------------------------------------------------------------------------
% name_c read

[header_str, count]=fread(fid, headerSize, 'char');
if (count ~= headerSize),
    fclose(fid);
    warning(['fread error on ' fileName ' : number of readen data = ' num2str(count) ' , ' num2str(headerSize) ' expected' ]);
    return
end

if (nargout == 3)
    if (multiheader == 0)
        name_c = num2cell(repmat(char(header_str'), vectSize-3,1), 2)';
    else
        name_c = strread(char(header_str), '%s', 'delimiter', ',')';
    end
end

% ------------------------------------------------------------------------
% Data read

[Data_v, count] = fread(fid, (vectSize)*vectorNumber, 'float32=>float32');

if (count~=(vectSize)*vectorNumber)
    fclose(fid);
    warning(['fread error on ' fileName ' : number of readen data = ' num2str(count) ' , ' num2str((vectSize+1)*vectorNumber) ' expected' ]);
    return
end

% ------------------------------------------------------------------------
% Data formating

Data_v                = reshape(Data_v, [vectorNumber, vectSize]);
descriptorMatrix_m    = Data_v(:, 4:end);
timeSegment_sec_m    = Data_v(:, 1:3);



% ------------------------------------------------------------------------
% Close

status = fclose(fid);

if (status == -1)
    descriptorMatrix_m    = [];
    timeSegment_sec_m    = [];
    warning(['fclose error on ' fileName ' : can not close the fid ']);
    return
end 
