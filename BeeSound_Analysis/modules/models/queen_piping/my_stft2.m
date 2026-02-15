function [ Sw, mod_tfr, mod_tfr_ref] = my_stft(s, w, rec, q_method, a_method, T, p_method)
% [Sw] = my_stft(s,w,rec)
%
% 
% INPUT:
% s: input signal
% w: analysis window
% rec: overlap between adjacent frames
% 
%
% OUTPUT:
% Sw: STFT, spectrogram given by |Sw|^2
% Author : D. Fourer (dominique@fourer.fr)
% Date : 01-oct-2021

k        = 2;
if ~exist('q_method', 'var')
 q_method = 0;
end

if ~exist('a_method', 'var')
 a_method = 1;
end

if ~exist('T', 'var')
 T = 1e-5;  %1e-5
end

if ~exist('p_method', 'var')
 p_method = 1;
end

 %% default window
 if ~exist('w', 'var')
   w = hann(2048, 'periodic').';
   %w = kbdwin(2048, 4).';
 end

 if ~exist('rec', 'var')
   rec = 2;
 end

 if ~exist('Fs', 'var')
   Fs = 22100;
 end
 
 
 N = length(w);
 len = length(s);

 step = round(N/rec);
 nb_trame = floor(len/step);

 Sw       = zeros(N, nb_trame);

 mod_tfr      = zeros(N, nb_trame);
 mod_tfr_ref  = zeros(N, nb_trame);
 wref = w;
 sum_w = sum(w);
 for i_t = 1:nb_trame
   i0 = (i_t-1)*step+1;
   i1 = min([len i0+N-1]);
 
   trame = s(i0:i1);
 
   N_tmp = length(i0:i1);

   if N_tmp < N
     w = resample(wref, N_tmp, N);
     sum_w = sum(w);
   end
   
   Sw(1:N_tmp, i_t) = fft( trame .* w );
   
   if q_method > 0
     if p_method == 1  
      m = peak_detect( abs(Sw(1:N_tmp, i_t)), T );
     else
      m = find( (abs(Sw(1:N_tmp, i_t))/sum_w) > T);   
      %[ m ] = 1:N_tmp; %peak_detect( abs(Sw(1:N_tmp, i_t)), T );
     end
     %m = find( (abs(Sw(1:N_tmp, i_t))/sum_w) > T);
     if length(m) > 1
      if q_method ~= 5
       [a_hat, mu_hat, phi_hat, omega_hat, psi_hat, delta_t_hat] = my_reassignment(trame, Fs, k, q_method, a_method, 1:N_tmp);
      else  %% phase vocoder
       [a_hat, mu_hat, phi_hat, omega_hat, psi_hat, delta_t_hat] = my_phasevocoder(trame, Fs, 1:N_tmp);  
      end
      
      mod_tfr_ref(m, i_t) =  (mu_hat(m) + 1j * psi_hat(m)).';   %% store as complex number
      mod_tfr(m, i_t) = ( real(mu_hat(m)./log(eps+2*a_hat(m))) + 1j * psi_hat(m)./(eps+omega_hat(m))).';   %% store as complex number
     end
   end
 end
 end
