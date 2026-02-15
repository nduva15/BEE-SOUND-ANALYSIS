clear
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              Save descriptors and STFT of the splitted piping dataset
%
% Author : D. Fourer (dominique@fourer.fr)
% Date : 01-oct-2021


%%%%%%%%%%%%%%%%%%%%%% timbre toolbox settings
if ~exist('Gget_desc_onefile')
  addpath('./timbretoolbox')
  addpath('./timbretoolbox/_tools')
  addpath('./timbretoolbox/_tools_sf')
  addpath('./timbretoolbox/classes')
end
fs_target = 44100;
% === PARAMETERS
do_s.b_TEE				= 1;    % descriptors from the Temporal Energy Envelope
do_s.b_STFTmag			= 1;    % descriptors from the STFT magnitude
do_s.b_STFTpow			= 1;    % descriptors from the STFT power
do_s.b_Harmonic			= 1;	% descriptors from Harmonic Sinusoidal Modeling representation
do_s.b_ERBfft			= 1;    % descriptors from ERB representation (ERB being computed using FFT)
do_s.b_ERBgam			= 1;    % descriptors from ERB representation (ERB being computed using Gamma Tone Filter)
config_s.xcorr_nb_coeff = 12;	% === defines the number of auto-correlation coefficients that will be sued
config_s.threshold_harmo= 0.3;	% === defines the threshold [0,1] below which harmonic-features are not computed
config_s.nb_harmo		= 20;	% === defines the number of harmonics that will be extracted
% Specify respective analysis methods for ERB.
%config_s.ERBfft.w_Method	= 'fft';
%config_s.ERBgam.w_Method	= 'gammatone';
% ====================


%% input: contains the wav files
dataset_dir = './dataset/piping_database';

%% output: contains the STFT and the TTB descriptors
dataset_outdir = './dataset/piping_database/output';

output_dir = sprintf('%s/wav', dataset_outdir);

mkdir(dataset_outdir);
mkdir(output_dir);

quack.dur = 0;
quack.nb  = 0;

toot.dur = 0;
toot.nb  = 0;

qc = {};
tt = {};

chk_dur = 1;  %% chunk duration expressed inseconds


%% STFT parameters
N = 1024;
rec = 2;
w = hann(N, 'periodic');

d = dir(dataset_dir);

for i = 1:length(d)
   if d(i).name(1) == '.' || length(d(i).name) < 5
      continue;  
   end
   
   if ~strcmpi(d(i).name(end-2:end), 'wav')
     continue;
   end
   
   fname = d(i).name(1:end-4);
   
   [s, Fs] = audioread(sprintf('%s/%s', dataset_dir, d(i).name));
   
   dur = length(s)/Fs;
   chk_len = round(chk_dur * Fs);
   
   nb_chk = ceil(length(s) / chk_len);
   warning('off')
   for c = 1:nb_chk
       
       filename = sprintf('%s_chk%d', fname, c);
       i0 = (c-1)*chk_len + 1;
       i1 = min(length(s), i0+chk_len-1);
       
       stmp = s(i0:i1);
       
       %% normalize and center signal
       stmp = stmp-mean(stmp);
       stmp = stmp / max(abs(stmp));
       
       
       wav_file = sprintf('%s/%s.wav', output_dir,filename);
       audiowrite(wav_file, stmp, Fs);
       
      %% 1 STFT
       [ Sw, ~, ~] = my_stft(stmp, w, rec, 0, 0);
       % reconstruction
       %s_hat = my_inv_stft(Sw, ones(1,N), rec, length(s));
       %SNR(s(N:end-N),s_hat(N:end-N).')
   
       % save the STFT of the signal in .mat file
       save(sprintf('%s/%s_stft.mat', dataset_outdir, filename), 'Sw', 'Fs', 'N', 'rec');
       
       
      %% TTB          
      if (Fs ~= fs_target)  % invalid sampling rate
       wav_file2 = './tmpfile.wav';
       system(sprintf('sox "%s" -r %d "%s"', wav_file, fs_target, wav_file2));
       fprintf(1, 'resampling %s...\n', wav_file);
       wav_file = wav_file2;
      end

      %% MFCC
      [ mfcc, t ] = my_mfcc( stmp, Fs );
      
      
      
      
      tic
      desc_tmp = Gget_desc_onefile(wav_file,do_s, config_s);
      %desc_tmp = Gget_desc_onefile(wav_file, do_s, config_s);
      modeling_tmp = Gget_temporalmodeling_onefile(desc_tmp);
      ttb_vec = real(struct2array(modeling_tmp));
      toc
      feature_name = fieldnames(modeling_tmp);
      save(sprintf('%s/%s_ttb.mat', dataset_outdir, filename), 'ttb_vec', 'feature_name', 'Fs');
 
   end
   warning('on')

   if regexpi(d(i).name,'quack')
     quack.dur = quack.dur + dur;
     quack.nb  = quack.nb + 1;
     qc(quack.nb).name = d(i).name(1:end-4);
     qc(quack.nb).dur  = dur;
   else
     toot.dur = toot.dur + dur;
     toot.nb  = toot.nb + 1; 
     tt(toot.nb).name = d(i).name(1:end-4);
     tt(toot.nb).dur  = dur;
   end
    
end


toot
quack
Fs

figure
bar([qc.dur])
xlabel('file index')
ylabel('duration [s]')
title('Quacking')

figure
bar([tt.dur])
xlabel('file index')
ylabel('duration [s]')
title('Tooting')






