function [ Sw, mod_tfr, mod_tfr_ref] = my_stft(s, w, rec)
% [Sw] = my_stft(s,w,rec)
%
% 
% INPUT:
% s: input signal
% w: analysis window
% rec: overlap between adjacent frames (ratio is 1/rec)
% 
%
% OUTPUT:
% Sw: STFT, spectrogram given by |Sw|^2
% Author : D. Fourer (dominique@fourer.fr)
% Date : 01-oct-2021

 %% default window
 if ~exist('w', 'var')
   w = hann(2048, 'periodic').';
   %w = kbdwin(2048, 4).';
 end

 if ~exist('rec', 'var')
   rec = 2;
 end
 
 N = length(w);
 len = length(s);

 step = round(N/rec);
 nb_trame = floor(len/step);

 Sw       = zeros(N, nb_trame);

 mod_tfr      = zeros(N, nb_trame);
 mod_tfr_ref  = zeros(N, nb_trame);
 wref = w;
 for i_t = 1:nb_trame
   i0 = (i_t-1)*step+1;
   i1 = min([len i0+N-1]);
 
   trame = s(i0:i1);
 
   N_tmp = length(i0:i1);

   if N_tmp < N
     w = resample(wref, N_tmp, N);
   end
   
   Sw(1:N_tmp, i_t) = fft( trame .* w );
 
 end 
end
