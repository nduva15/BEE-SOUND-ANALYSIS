function [ s_hat ] = rms_env( s, N )

s_hat = zeros(1, length(s));

H = floor(N/2);

for n = 1:length(s)
  i0 = max(1,n-H+1);
  i1 = min(length(s),i0+N-1);
  
  s_hat(n) = sqrt(mean(abs(s(i0:i1)).^2));
end

end

