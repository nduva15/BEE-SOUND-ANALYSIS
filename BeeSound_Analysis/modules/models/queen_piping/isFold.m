function [ val ] = isFold( ftest, fold )
% [ val ] = isFold( ftest, fold )
%
% return true if ftest in in fold list
%

val = false;

i = 1;
N = length(fold);
while (i <= N && val == false)
% val = ~(isempty(regexpi(fold{i}, ftest, 'match')));
 val = strcmpi(ftest, fold{i});
 i = i+1;   
end

end

