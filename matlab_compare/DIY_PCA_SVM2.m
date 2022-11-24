% DIY_PCA_SVM dataset 2
%% setting 1
name_class={'1','2','3','4','5','6'};
acc1 = zeros(1,2);
for SL = 2:3
    clear -regexp [^SL,^acc1,^acc2]
    clc
    load('D:\A_\Enose_datasets\4个月原始数据\datasetB_prep.mat')
    Source = batch1;
    if SL == 2
        Target = batch2;
        Tbatch_label = batch2_label;
    elseif SL == 3
        Target = batch3;
        Tbatch_label = batch3_label;
    end
    Source_label=(vec2ind(batch1_label'))';
    Target_label=(vec2ind(Tbatch_label'))';
    
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
    for m=-5:0.2:5
        c=10^m;
        for n=-5:0.2:5
             gama=10^n;
             counter=counter+1;
            try
                % SVM:
                     cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
                     model = svmtrain(Source_label,Re_Source,cmd);  
                     [predict_label_test,] = svmpredict(Target_label,Re_Target, model);
                     [confusion_matrix]=compute_confusion_matrix(Target_label,num_in_class,name_class);
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
acc2 = zeros(1,2);
for SL = 2:3
    clear -regexp [^SL,^acc1,^acc2,^acc1_mean]
    clc
    load('D:\A_\Enose_datasets\4个月原始数据\datasetB_prep.mat')
    if SL-1 == 1
        Source = batch1;
        Sbatch_label = batch1_label;
    elseif SL-1 == 2
        Source = batch2;
        Sbatch_label = batch2_label;
    end
    if SL == 2
        Target = batch2;
        Tbatch_label = batch2_label;
    elseif SL == 3
        Target = batch3;
        Tbatch_label = batch3_label;
    end
    Source_label=(vec2ind(Sbatch_label'))';
    Target_label=(vec2ind(Tbatch_label'))';
    
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
    for m=-6:0.2:6
        c=10^m;
        for n=-6:0.2:6
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

