function [v] = rms(s)
%%  [v,t] = rms(s, N, rec, Fs)
% compute linear Root Mean Square

 v = sqrt(mean(s.^2));
