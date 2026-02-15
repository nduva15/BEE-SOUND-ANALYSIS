clear
close all

%% methods_to_use
addpath('./MSVMpack1.5/matlab');

dataset_dir = './dataset/output/wav';

methods = [1 2 3 4];
method_name = {'1D CNN' 'MFCC + CNN' 'STFT+CNN' 'TTB + SVM' };


fold{1} = loadfold('./fold1.txt');
fold{2} = loadfold('./fold2.txt');
fold{3} = loadfold('./fold3.txt');

da       = 3;  %% data augmentation ratio
da_snr   = 25; %% data augmentation noise snr
nb_class = 2;

fp = fopen('./resultats.txt', 'w+t');

for m = 1:length(methods)
    
    conf_mat = zeros(nb_class,nb_class,3);
    
    mm = methods(m);
    fprintf(1,'++++++++++++++++++++++++  Method : %s \n', method_name{mm});
    for n_test  = 1:3
      
      n_train = setdiff(1:3, n_test);
      
      fprintf(1, 'test: %d \n', n_test);
      fprintf(1, 'train: ');
      fprintf(1,'%d ', n_train); %d \n', n_train);
      fprintf(1, '\n');
      
      fprintf(1,'prepare data...');
      tic
      [x_train, y_train] = loaddat([fold{n_train}], dataset_dir, mm , da, da_snr);  %% data augmentation
      [x_test,  y_test]  = loaddat(fold{n_test},    dataset_dir, mm,0);
      toc
     
      
      if mm == 1 %% 1d cnn
          %% train
          %train_1DCNN(x_train, y_train)
          tmp = abs(fft(x_train,[],2));
          x_train = tmp(:,1:11025);
          train_1DCNN( x_train, y_train, '1dcnn'); % model_name
          
          %% test
          tmp = abs(fft(x_test,[],2));
          x_test = tmp(:,1:11025);
          
          ytmp = pred_1DCNN(x_test, '1dcnn');
          ytmp(ytmp==1) = 3;
          ytmp(ytmp==2) = 1;
          ytmp(ytmp==3) = 2;
          y_hat = ytmp;
          
      elseif mm == 2 %% MFCC + CNN
          %% train
          train_2DCNN( x_train, y_train, 'mfcc-cnn');
          
          %% test
          y_hat = pred_2DCNN(x_test, 'mfcc-cnn');
          
      elseif mm == 3 %% STFT + CNN %(tested)
          %% train
          train_2DCNN( x_train, y_train, 'stft-cnn');
          
          %% test
          y_hat = pred_2DCNN(x_test, 'stft-cnn');
          
      elseif mm == 4 %(tested)
          
          %% Z-score dataset
          x_train = zscore(x_train);
          x_test = zscore(x_test);
          
          %% use SVM
          %% train
          par_svm = '-m MSVM2 -k 2 -a 0.99 -q -n '; %-n
          m1 = trainmsvm(x_train, y_train, par_svm); %-q   
          
          % test  
          y_hat = predmsvm(m1, x_test);
          
          %% use LDA
%           indices = zeros(2,2);
%           indices(1,1) = min(find(y_train == 1));
%           indices(1,2) = max(find(y_train == 1));
%           indices(2,1) = min(find(y_train == 2));
%           indices(2,2) = max(find(y_train == 2));
%           
%           [Vect, repr, repr2] = train_lda(x_train, indices, 8);
%           
%           [gr1, gr2, p1, p2] = pred_lda(x_test, Vect, repr, repr2);
%           
%           y_hat = gr1;
      end 
      
      
      % Fill the confusion matrix
      for l = 1:nb_class   %% true class
        I = find( y_test == l);  %% position of the reference class l
        for ll = 1:nb_class      %% estimated class
          conf_mat(l,ll, n_test) = conf_mat(l,ll, n_test) + length(find(y_hat(I) == ll));
        end
      end
      
      %% display results
     fprintf(1,'\nConfusion matrix : Method : %s\n', method_name{mm});
     conf_mat(:,:,n_test)
%     [FMeasure, Recall, Precision, Accuracy] = confMatScores(conf_mat(:,:,n_test))
    
    end  %% folds

    fprintf(1, '++++++++   Overall Results ++++++++\n');
    fprintf(1,'\nConfusion matrix : Method : %s\n', method_name{mm});
    overall_confmat = sum(conf_mat(:,:,:),3)
    [FMeasure, Recall, Precision, Accuracy] = confMatScores(overall_confmat)
    
    fprintf(fp,'\nConfusion matrix : Method : %s\n', method_name{mm});
    fprintf(fp,'ConfMat:\n%d %d\n%d %d\n\n', overall_confmat(:,:))
    fprintf(fp,'Recall:\t\t %.2f %.2f\n', Recall(1), Recall(2));
    fprintf(fp,'Precision:\t %.2f %.2f\n', Precision(1), Precision(2));
    fprintf(fp,'F-Measure:\t %.2f %.2f\n', FMeasure(1), FMeasure(2));
    fprintf(fp,'Accuracy:\t %.2f\n\n', Accuracy);
end


fclose(fp);


