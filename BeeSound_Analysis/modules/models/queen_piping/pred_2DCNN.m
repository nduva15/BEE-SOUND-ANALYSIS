function [ y_test ] = pred_1DCNN( x_test, model_name )

filemat = '___tmp.mat';
filemat2 = '___tmp2.mat';

if ~exist('model_name', 'var')
   model_name = 'tmp_model';
end

save(filemat, 'x_test', 'model_name')

err = system(sprintf('python ./pred_2DCNN.py %s', model_name));
pause(0.1)

load(filemat2);

delete(filemat);
delete(filemat2);
end

