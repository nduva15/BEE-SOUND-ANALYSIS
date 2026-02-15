function [ x, y, Fs ] = loaddat( folds, dataset_dir, features, da, DA_SNR )

if ~exist('features', 'var')
  features = 1; 
end

if ~exist('da', 'var')
  da = 0; 
end

if ~exist('DA_SNR', 'var') %% SNR for data augmentation
  DA_SNR = 30; 
end


recompute_all = false;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


da = round(da);

  x = [];
  y = [];
  index = 0;
  
  d = dir(dataset_dir);
  
  for i = 1:length(d)
   if d(i).name(1) == '.' || length(d(i).name) < 5
      continue;  
   end
   if ~strcmpi(d(i).name(end-2:end), 'wav')
     continue;
   end
   
   fname = d(i).name(1:end-4);
   
   tmp = regexpi(fname, '_', 'split');
   fn = sprintf('%s_%s', tmp{1}, tmp{2});
   %fprintf(1, '%s\n', fn);
   %isFold( fn, folds )
   %% check if is in fold
   if ~isFold( fn, folds )
      %fprintf(1, 'skip %s...\n', fn);
      continue;
   end

   index = index+1;
   
   if ~isempty(regexpi(fname, 'toot', 'match'))
     y(index) = 1; %% toot
   else
     y(index) = 2;  %% quack
   end
    [s, Fs] = audioread(sprintf('%s/%s', dataset_dir, d(i).name));
    
    %normalize signal
    s = s-mean(s);
    s = s / (1.2*max(abs(s)));  
    
   if features == 1 %% signal
      feats = s;
      x(index,:) = feats;
   elseif features == 2 %% MFCC
      %[s, Fs] = audioread(sprintf('%s/%s', dataset_dir, d(i).name));
      [ mfcc, t ] = my_mfcc( s, Fs );
      feats = mfcc(:,1:end).'; %reshape(mfcc.', [1 numelem(mfcc)]);
      x(index,:,:) = feats;
   elseif features == 3 %% STFT
      M = 1025;
      H = 512;
      w = hann(M, 'periodic');
      rec = 2;
      %[s, Fs] = audioread(sprintf('%s/%s', dataset_dir, d(i).name));
      [ Sw, ~, ~] = my_stft(s, w, rec);
      
      feats = abs(Sw(1:H,:)).^0.5;
      [nf, nt] = size(feats);
      
      x(index,:,:) = feats;
   elseif features == 4 %% TTB
       
       %%% load file to improve speed
       ttbfile = sprintf('%s/ttb/%s_ttb.mat', dataset_dir, fname);
       if exist(ttbfile, 'file') && ~recompute_all
          tmp = load(ttbfile);
          feats = tmp.ttb_vec;
       else
          warning('off')
          %% TTB   
          wav_file = './tmpfile2.wav';
          audiowrite(wav_file, s, Fs);
          if (Fs ~= fs_target)  % invalid sampling rate
           
           wav_file2 = './tmpfile.wav';
           system(sprintf('sox "%s" -r %d "%s"', wav_file, fs_target, wav_file2));
           %fprintf(1, 'resampling %s...\n', wav_file);
           wav_file = wav_file2;
          end   

          desc_tmp = Gget_desc_onefile(wav_file,do_s, config_s);
          modeling_tmp = Gget_temporalmodeling_onefile(desc_tmp);
          ttb_vec = real(struct2array(modeling_tmp));
          feature_name = fieldnames(modeling_tmp);
          
          save(ttbfile, 'ttb_vec', 'feature_name', 'Fs');
          feats = ttb_vec;
          warning('on')
       end
       x(index,:) = feats;
   end 
   

   %%%%%%%%%%% DATA AUGMENTATION
   if da > 0
     %fprintf(1, 'data augmentation...\n')
     lbl = y(index);  %% same label
     
     %% ref signal
     s0 = s;
     
     for ii = 1:da
       %% add noise
       s = sigmerge(s0, randn(size(s)), DA_SNR);
 
       %% random circshift
       s = circshift(s, randi(length(s)));

       
       %normalize signal
       s = s-mean(s);
       s = s / (1.2*max(abs(s)));  
    
       if features == 1 %% signal
        feats = s;
        x(index,:) = feats;
       elseif features == 2 %% MFCC
        [ mfcc, t ] = my_mfcc( s, Fs );
        feats = mfcc(:,1:end).'; %reshape(mfcc.', [1 numelem(mfcc)]);
        x(index,:,:) = feats;
       elseif features == 3 %% STFT
        M = 1025;
        H = 512;
        w = hann(M, 'periodic');
        rec = 2;
        [ Sw, ~, ~] = my_stft(s, w, rec);
      
        feats = abs(Sw(1:H,:)).^0.5;
        [nf, nt] = size(feats);
      
        x(index,:,:) = feats;
        
       elseif features == 4 %% TTB
       
        %%% load file to improve speed
        ttbfile = sprintf('%s/ttb/%s_ttb_d%d.mat', dataset_dir, fname, ii);
        if exist(ttbfile, 'file') && ~recompute_all
          tmp = load(ttbfile);
          feats = tmp.ttb_vec;
        else 
          warning('off')
          %% TTB
          wav_file = './tmpfile2.wav';
          audiowrite(wav_file, s, Fs);
          if (Fs ~= fs_target)  % invalid sampling rate
           wav_file2 = './tmpfile.wav';
           system(sprintf('sox "%s" -r %d "%s"', wav_file, fs_target, wav_file2));
           %fprintf(1, 'resampling %s...\n', wav_file);
           wav_file = wav_file2;
          end   

          desc_tmp = Gget_desc_onefile(wav_file,do_s, config_s);
          modeling_tmp = Gget_temporalmodeling_onefile(desc_tmp);
          ttb_vec = real(struct2array(modeling_tmp));
          feature_name = fieldnames(modeling_tmp);
          
          save(ttbfile, 'ttb_vec', 'feature_name', 'Fs');
          feats = ttb_vec;
          warning('on')
        end
        x(index,:) = feats;
       end % features
              
     end %% da
   
   end
    
  end


end

