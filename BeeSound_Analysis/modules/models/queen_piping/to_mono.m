function [s, nb_channel, len] = to_mono(s)

nb_channel = min(size(s));
len = max(size(s));

if size(s,1)>size(s,2)
 s = s';
end

%%convert to mono
if nb_channel > 1
 s = sum(s)/nb_channel;
end

s = s-mean(s);
s = s / (1.2*max(abs(s)));   %normalize