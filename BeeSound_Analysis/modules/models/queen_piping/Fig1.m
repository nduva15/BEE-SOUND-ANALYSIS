clear 
close all

addpath('./synchrosqueezedSTFT/')
output = 'figs';

mkdir(output);

dossier= 'dataset/piping_database/';

file1 = 'toot_1.wav';
file2 = 'quack_1.wav';

N = 512;

[s1, Fs] = audioread(sprintf('%s/%s', dossier, file1));
[s2, ~] = audioread(sprintf('%s/%s', dossier, file2));

t1 = ((1:length(s1))-1)/Fs;
t2 = ((1:length(s2))-1)/Fs;


s1 = s1 ./ max(abs(s1));
s1_rms = rms_env(s1, N);

s2 = s2 ./ max(abs(s2));
s2_rms = rms_env(s2, N);


M = 512;
L = 45; %30

%% compute TFRs
[tfr1] = tfrgab2(s1, M, L);
[tfr2] = tfrgab2(s2, M, L);

%% estimate pitch
[ps1,ts1,ss1] = swipep(s1,Fs);
I1 = find(ss1 > 0.25);
[ps2,ts2,ss2] = swipep(s2,Fs);
I2 = find(ss2 > 0.25);

f = ((1:M)-1)/M * Fs;
H = floor(M/2);

figure(1)
subplot(211)
imagesc(t1,f(1:H), abs(tfr1(1:H,:)).^0.4);
ylim([0 8000])
yticks([0 500 1000 2000 4000 6000 8000]);
hold on
h = plot(ts1(I1),ps1(I1), 'r-x');
legend(h, 'estimated F0');
xticks('')
set(gca,'YDir','normal');
pos = get(gca, 'Position');
pos(2) = pos(2)-0.13;
pos(4) = pos(4)+0.14;
set(gca, 'Position', pos)
colormap gray;
colormap(flipud(colormap));
%set(gca,'YScale','log')
ylabel('frequency [Hz]');
subplot(212)
plot(t1, s1);
ylim([-1 0.99])
hold on
h = plot(t1, s1_rms, 'r-.');
xlabel('time [s]');
legend(h, 'RMS envelope');
saveas(gcf, sprintf('%s/toot1.eps', output), 'epsc');

%pause

figure(2)
subplot(211)
imagesc(t2,f(1:H), abs(tfr2(1:H,:)).^0.4);
ylim([0 8000])
yticks([0 500 1000 2000 4000 6000 8000]);
hold on
h = plot(ts2(I2),ps2(I2), 'r-x');
legend(h, 'estimated F0');
xticks('')
set(gca,'YDir','normal')
%set(gca,'YScale','log')
pos = get(gca, 'Position');
pos(2) = pos(2)-0.13;
pos(4) = pos(4)+0.14;
set(gca, 'Position', pos)
colormap gray;
colormap(flipud(colormap));
ylabel('frequency [Hz]');
subplot(212)
plot(t2, s2);
ylim([-1 0.99])
hold on
h = plot(t2, s2_rms, 'r-.');
xlabel('time [s]');
legend(h, 'RMS envelope');
saveas(gcf, sprintf('%s/quack1.eps', output), 'epsc');

eps2pdf('./figs');