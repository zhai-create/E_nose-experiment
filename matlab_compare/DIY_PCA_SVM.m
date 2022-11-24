% DIY_PCA_SVM
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
    
    % PCA
    [COEFF_train,SCORE_train,latent,tsquare] = pca(Source);
    %各成分的方差贡献率
    latent_percent=latent./sum(latent).*100;
    sumcounts=1;
    sum_score=latent_percent(sumcounts,1);
    while(sum_score<99)
        sumcounts=sumcounts+1;
        sum_score=sum_score+latent_percent(sumcounts,1);
    end
    npc=sumcounts; %npc表示要取的主成分维数
    Re_Source=SCORE_train(:,1:npc);%训练样本进行PCA       
    %测试样本进行PCA投影
    SCORE_test= bsxfun(@minus,Target,mean(Source,1))*COEFF_train;    %测试样本投影到训练样本确定的PCA空间
    Re_Target=SCORE_test(:,1:npc);

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
    
    % PCA
    [COEFF_train,SCORE_train,latent,tsquare] = pca(Source);
    %各成分的方差贡献率
    latent_percent=latent./sum(latent).*100;
    sumcounts=1;
    sum_score=latent_percent(sumcounts,1);
    while(sum_score<99)
        sumcounts=sumcounts+1;
        sum_score=sum_score+latent_percent(sumcounts,1);
    end
    npc=sumcounts; %npc表示要取的主成分维数
    Re_Source=SCORE_train(:,1:npc);%训练样本进行PCA       
    %测试样本进行PCA投影
    SCORE_test= bsxfun(@minus,Target,mean(Source,1))*COEFF_train;    %测试样本投影到训练样本确定的PCA空间
    Re_Target=SCORE_test(:,1:npc);

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
    acc2(SL-1) = max;
end
acc2_mean = mean(acc2)

