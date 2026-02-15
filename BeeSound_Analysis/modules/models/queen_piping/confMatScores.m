function [FMeasure, Recall, Precision, Accuracy] = confMatScores(conf_mat)
% [FMeasure, Recall, Precision, Accuracy] = confMatScores(conf_mat)
%
% return the FMeasure, Recall, Precision, Accuracy scores from a given
% confusion matrix
%
% confmat rows are true classes / cols are predicted

[nb_class, nb_class2] = size(conf_mat);

if (nb_class ~= nb_class2) || nb_class < 2 || nb_class2 < 2
 error('Invalid confusion matrix');    
end

FMeasure  = zeros(1, nb_class);
Recall    = zeros(1, nb_class);
Precision = zeros(1, nb_class);

for i = 1:nb_class
  Recall(i)    = conf_mat(i,i) / sum(conf_mat(i,:));
  Recall(isnan(Recall)) = 0;
  
  Precision(i) = conf_mat(i,i) / sum(conf_mat(:,i));
  Precision(isnan(Precision)) = 0;
  
  FMeasure(i)  = 2 * Recall(i) * Precision(i) / (Recall(i) + Precision(i));
  FMeasure(isnan(FMeasure)) = 0;
end

Accuracy = sum(diag(conf_mat))/ sum(sum(conf_mat));