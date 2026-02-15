function [ mfcc, t ] = my_mfcc( s, Fs )
% [ mfcc, t ] = my_mfcc( s, Fs )
%
%
%

if strcmpi(computer,'GLNX86')   %% linux 32
 mfcc_exec = './mfcc/mirex_mfccsfm_extractor-1.4';
else    %% linux 64
 mfcc_exec = './mfcc/mirex_mfccsfm_extractor-1.3';
end

[err, errstr] = system(sprintf('%s', mfcc_exec));
if  err ~= 0
  error(errstr);
end

if size(s,1) < size(s,2)
 s = s.';    
end

file_tmpwav = '_____tmp.wav';
mfcc_file   = '____mfcc.dat';

audiowrite(file_tmpwav, s, Fs);
%wavwrite(s, Fs, file_tmpwav);

[a,b] = system(sprintf('%s --file %s --mfccsfminst_out %s', mfcc_exec, file_tmpwav, mfcc_file));

[mfcc, t] = parsermfcc(mfcc_file);
t = t(:,1);

delete(file_tmpwav);

end

