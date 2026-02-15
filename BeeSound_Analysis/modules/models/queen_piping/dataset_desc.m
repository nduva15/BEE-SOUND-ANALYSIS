clear
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              Description of the piping dataset content
%
% Author : D. Fourer (dominique@fourer.fr)
% Date : 01-oct-2021

dataset_dir = './dataset/piping_database';

quack.dur = 0;
quack.nb  = 0;

toot.dur = 0;
toot.nb  = 0;

qc = {};
tt = {};

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
   
    %% 1 STFT (test reconstruction)
%     N = 1024;rec=2;w = hann(N,'periodic');
%    [ Sw, ~, ~] = my_stft(s, w, rec);
%    % reconstruction
%    s_hat = my_inv_stft(Sw, ones(1,N), rec, length(s));
%    SNR(s(N:end-N),s_hat(N:end-N).')
%    pause
   
   dur = length(s)/Fs;
 
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

%%%%%%%%%%%%

% affect Yt video index 
yt = {};

%% video1 : https://www.youtube.com/watch?v=9naKEy1v6Lw
% toot 1
% quack 1-2
idx = 1;
yt(idx).tt_idx = 1;
yt(idx).qc_idx = 1:2;


%% video2 : https://www.youtube.com/watch?v=x-YYhWR_i8s&t=42s
% 2
% toot 2 3
idx = 2;
yt(idx).tt_idx = 2:3;
yt(idx).qc_idx = [];



%% video3 : https://www.youtube.com/watch?v=NNsmbAmknbI
% 3
% toot 4,5,6
idx = 3;
yt(idx).tt_idx = 4:6;
yt(idx).qc_idx = [];


%% video4 : https://www.youtube.com/watch?v=x-YYhWR_i8s&t=42s
% toot 7 - 11
% quack 3-5
idx = 4;
yt(idx).tt_idx = 7:11;
yt(idx).qc_idx = 3:5;


%% video 5 : https://www.youtube.com/watch?v=PjJmsWrc_Gk&t=27s
% toot 12-18
idx = 5;
yt(idx).tt_idx = 12:18;
yt(idx).qc_idx = [];


%% video 6 : https://www.youtube.com/watch?v=AYecvVhkpKI&t=10s
% toot 19-20
idx = 6;
yt(idx).tt_idx = 19:20;

%% video 7 : https://www.youtube.com/watch?v=oaudlgBhOQQ
% toot 21-24
idx = 7;
yt(idx).tt_idx = 21:24;

%% video 8 : https://www.youtube.com/watch?v=egJu0bpGXgk&t=56s
% toot 25-26
idx = 8;
yt(idx).tt_idx = 25:26;


%% video 9 : https://www.youtube.com/watch?v=RBF2pGeyvXY
% toot 27
idx = 9;
yt(idx).tt_idx = 27;
yt(idx).qc_idx = [];

%% video 10 : https://www.youtube.com/watch?v=FFFGLjncAbk&t=121s
% toot 28-30
idx = 10;
yt(idx).tt_idx = 28:30;
yt(idx).qc_idx = [];

%% video 11 : https://www.youtube.com/watch?v=Kk3Jz6xJufU
% quack 6-8
% toot 31
idx = 11;
yt(idx).tt_idx = 31;
yt(idx).qc_idx = 6:8;


%% video 12 : https://www.youtube.com/watch?v=xK8fEPsAc6U
%# toot 32
idx = 12;
yt(idx).tt_idx = 32;
yt(idx).qc_idx = [];


%% video 13 : https://www.youtube.com/watch?v=7qwUslas1zI
% toot 33-35
idx = 13;
yt(idx).tt_idx = 33:35;
yt(idx).qc_idx = [];

%# video 14 : https://www.youtube.com/watch?v=BsoSHLJSgA0
% toot 36
idx = 14;
yt(idx).tt_idx = 36;
yt(idx).qc_idx = [];

for i = 1:14
  yt(i).tt_dur = 0;
  yt(i).qc_dur = 0;
  
  for j = yt(i).tt_idx
    tt(j).yt = i;
    yt(i).tt_dur = yt(i).tt_dur + tt(j).dur;
  end
  for j = yt(i).qc_idx
    qc(j).yt = i;
    yt(i).qc_dur = yt(i).qc_dur + qc(j).dur;
  end   
end

%% Fold composition
fold1 = [1:3 7 10];
fold2 = [4 8 12 14];
fold3 = [5 6 9 11 13];

figure
ttdur_vec = [sum([yt(fold1).tt_dur]) sum([yt(fold2).tt_dur]) sum([yt(fold3).tt_dur]) ];
bar(ttdur_vec)
xlabel('folds')
ylabel('duration [s]')
title('Tooting')

figure
qcdur_vec = [sum([yt(fold1).qc_dur]) sum([yt(fold2).qc_dur]) sum([yt(fold3).qc_dur]) ];
bar(qcdur_vec)
xlabel('folds')
ylabel('duration [s]')
title('Quacking')

figure
bar(ttdur_vec+qcdur_vec)
xlabel('folds')
title('Total duration')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fp = fopen('fold1.txt', 'w+t');
for i = fold1
 for t = yt(i).tt_idx, fprintf(fp, '%s\n', tt(t).name); end
 for q = yt(i).qc_idx, fprintf(fp, '%s\n', qc(q).name); end
end
fclose(fp);

fp = fopen('fold2.txt', 'w+t');
for i = fold2
 for t = yt(i).tt_idx, fprintf(fp, '%s\n', tt(t).name); end
 for q = yt(i).qc_idx, fprintf(fp, '%s\n', qc(q).name); end
end
fclose(fp);

fp = fopen('fold3.txt', 'w+t');
for i = fold3
 for t = yt(i).tt_idx, fprintf(fp, '%s\n', tt(t).name); end
 for q = yt(i).qc_idx, fprintf(fp, '%s\n', qc(q).name); end
end
fclose(fp);
