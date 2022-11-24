% DIY_CCPCA_SVM
%% setting 1
acc1 = zeros(1,9);
for SL = 2:10
    clear -regexp [^SL,^acc1,^acc2]
    clc
    load A.mat
    load index.mat
    Source=A{1,1};
    Target=A{1,SL};
    Source_label=(vec2ind(index{1,1}))';
    Target_label=(vec2ind(index{1,SL}))';
    %归一化
    Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
    Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);

    % CCPCA
    [pc,score,latent,tsquare] = pca(Source);
    P=pc(:,1);
    Re_Target = Target-Target*P*P';

    Re_Source = Source;
    
    error1=0;
    counter=0;
    for m=-4:4
        c=10^m;
        for n=-4:4
             gama=10^n;
             counter=counter+1;
            try
                % SVM:
                     cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
                     model = svmtrain(Source_label,Re_Source,cmd);  
                     [predict_label_test,] = svmpredict(Target_label,Re_Target, model);
                     d=diff([predict_label_test';Target_label']);
                     N = numel(find(d==0));
                     accur_test=N/size(Re_Target,1);
            catch
                  error1 =error1 + 1;
                  accur_test = 0;
            end
             result(counter,1)=m;
             result(counter,2)=n;
             result(counter,3)=accur_test;                            
        end                 
    end
    [max,index]=max(result(:,3));
    acc1(SL-1) = max;
end
acc1_mean = mean(acc1)

%% setting 2
acc2 = zeros(1,9);
for SL = 2:10
    clear -regexp [^SL,^acc1,^acc2]
    clc
    load A.mat
    load index.mat
    Source = A{1,SL-1};
    Target = A{1,SL};
    Source_label=(vec2ind(index{1,SL-1}))';
    Target_label=(vec2ind(index{1,SL}))';
    %归一化
    Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
    Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
    
    % CCPCA
    [pc,score,latent,tsquare] = pca(Target);
    P=pc(:,1);
    Re_Target = Target-Target*P*P';

    error1=0;
    counter=0;
    for m=-4:4
        c=10^m;
        for n=-4:4
             gama=10^n;
             counter=counter+1;
            try
                % SVM:
                     cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
                     model = svmtrain(Source_label,Source,cmd);  
                     [predict_label_test,] = svmpredict(Target_label,Re_Target, model);
                     d=diff([predict_label_test';Target_label']);
                     N = numel(find(d==0));
                     accur_test=N/size(Re_Target,1);
            catch
                  error1 =error1 + 1;
                  accur_test = 0;
            end
             result(counter,1)=m;
             result(counter,2)=n;
             result(counter,3)=accur_test;                            
        end                 
    end
    [max,index]=max(result(:,3));
    acc2(SL-1) = max;
end
acc2_mean = mean(acc2)

