function s_hat = my_inv_stft(Sw, w, rec, len)
% [Sw] = inv_spectrogram(Sw, w, rec)
%
% 
% INPUT:
% s: input signal
% w: analysis window
% rec: overlap between adjacent frames
%
%
% OUTPUT:
% s_hat: reconstructed signal
% Author : D. Fourer (dominique@fourer.fr)
% Date : 01-oct-2021


%% default window
if ~exist('w', 'var')
 %w = hann(2048).';
 w = kbdwin(2048, 4).';
end

if ~exist('rec', 'var')
 rec = 2;
end

w = w(:).';

N = length(w);
step = round(N/rec);
sz = size(Sw);
nb_trame = sz(2);

if ~exist('len', 'var')
 len = (nb_trame-1) * step + nnz(abs(Sw(:,end)));
end

wref = w;
s_hat = zeros(1, len);
for i_t = 1:nb_trame
 i0 = (i_t-1)*step+1;
 i1 = min([len i0+N-1]);
 N_tmp = length(i0:i1);
 if N_tmp < N
   if N_tmp < 1
    break;
   end
   w = resample(wref, N_tmp, N);
 end
 s_hat(i0:i1) = s_hat(i0:i1) + real(ifft( Sw(1:N_tmp,i_t)) )' .* w * 2/rec; 
end

%% eof