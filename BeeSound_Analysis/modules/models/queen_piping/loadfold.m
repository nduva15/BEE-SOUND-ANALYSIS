function [ fold ] = loadfold( fname )
% [ fold ] = loadfold( fname )
% read the content of a fold file 
%

fold = {};
fp = fopen(fname, 'r+t');

index = 1;
while(~feof(fp))
  fold{index} = fgetl(fp);
  index = index + 1;    
end
fclose(fp);

end

