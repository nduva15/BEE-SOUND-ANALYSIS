function [x,fs,nbits] = wavread(filename, param)

nbits = 16;
%warning('Deprecated: using my compatibility wavread function')
if ~exist('param', 'var')
  [x,fs] = audioread(filename);
elseif param == 'size'
 [x,fs] = audioread(filename);
 x = size(x);
end