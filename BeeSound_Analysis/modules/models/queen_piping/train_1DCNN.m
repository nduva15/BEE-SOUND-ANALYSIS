function [ err,model_name ] = train_1DCNN( x_train, y_train, model_name )

filemat = '___tmp.mat';

if ~exist('model_name', 'var')
   model_name = 'tmp_model';
end

save(filemat, 'x_train', 'y_train', 'model_name');

err = system('python train_1DCNN.py');

%delete(filemat);

end

