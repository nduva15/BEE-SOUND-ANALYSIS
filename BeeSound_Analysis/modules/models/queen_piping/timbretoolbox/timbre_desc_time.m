function [t, desc_struc, fname] =  timbre_desc_time(s, Fs, N, rec, do_s, config_s)
%  function [t, desc_struc, fname] =  timbre_desc_time(s, Fs, N, rec, do_s, config_s)
%
% Compute instantaneous timbre toolbox values, return 0 when rms is not significant
%
% INPUT:
% s:  input signal
% Fs: sampling frequency in Hz
% N: Frame length (number of sample)
% rec: overval ratio, the hop size is round(N/rec)
% do_s, config_s: timbre toolbox parameters/options
%
% OUTPUT:
% t: time index of each row 
% desc_struc: T x D matrix of computed descriptors, T is the number of time indices, D the number of descriptors
% fname: name of descriptor

 nb_desc = 64;
 
 step = round(N/rec);
 nb_trame = ceil( length(s) / step );
 desc_struc = [];
 T_NOISE = -inf;
 
 
 t = zeros(1, nb_trame);
 for k = 1:nb_trame
  i0 = (k-1)*step +1;
  i1 = min(length(s), i0+N-1);
  t(k) = (i0 + i1)/2 * Fs;

  trame_s = s(i0:i1);
  
  if 10 * log10(rms(trame_s)) > T_NOISE
   try
    desc_tmp0     = Gget_desc_onefile2(trame_s, Fs, do_s, config_s);
    modeling_tmp0 = Gget_temporalmodeling_onefile(desc_tmp0);

    if ~exist('fname', 'var')
     fname = fieldnames(modeling_tmp0); 
    end
    
    for fid = 1:length(fname)
     desc_struc(k, fid) = getfield(modeling_tmp0, fname{fid});
    end
   catch
     warning('failed to extract param, replace by 00')
     desc_struc(k, :) = zeros(1, max(nb_desc, length(fname)));
     if ~exist('fname', 'var')
      fname = {}; 
     end
   end
  else  %% skip silences 
   desc_struc(k, :) = zeros(1, max(nb_desc, length(fname)));
  end
 end
 
