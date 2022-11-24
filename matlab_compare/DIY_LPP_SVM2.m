% DIY_LPP_SVM dataset 2
%% setting 1
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
    
    % 调用constructW函数计算近邻，为lpp提供近邻关系矩阵
    fea = Source;                      %待处理矩阵输入，每一行是一个样本点
    options = [];
    options.NeighborMode = 'KNN';               %选择计算近邻的方法，此处用的是K近邻法                     
    options.WeightMode = 'HeatKernel';          %根据样本点之间的距离关系计算W的函数类型,本程序中选择的是“热核”
    options.k = 5;                              %近邻的个数
    options.t = 0.1;                              %热核参数,不能太小，太小导致计算出来的W都趋近于0，导致LPP报错
    W = constructW(fea,options);                
    % LPP算法进行降维
    npc=17;
    options.PCARatio = 1;                %The percentage of principal
                                         %component kept in【 the PCA
                                         %step. The percentage is
                                         %calculated based on the
                                         %eigenvalue. Default is 1
                                         %(100%, all the non-zero
                                         %eigenvalues will be kept.
    options.ReducedDim = npc;            %目标空间的维数，与options.PCARatio的值有关，当options.PCARatio=0.8时<=3维，当options.PCARatio=1时<=10
    [eigvector, eigvalue] = LPP(W, options, fea);
    Re_Source= fea*eigvector;    
    input_test = Target*eigvector;%测试样本进行LPP投影
    Re_Target = input_test;
    
%     % from CDSL other methods
%     batchS =Source;
%     batchT =Target;
%     [mappedA, mapping] = compute_mapping(batchS,'LPP',4);
%     W=mapping.M;
%     Re_Source=batchS*W;
%     Re_Target=batchT*W;
    
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
    
    % 调用constructW函数计算近邻，为lpp提供近邻关系矩阵
    fea = Source;                      %待处理矩阵输入，每一行是一个样本点
    options = [];
    options.NeighborMode = 'KNN';               %选择计算近邻的方法，此处用的是K近邻法                     
    options.WeightMode = 'HeatKernel';          %根据样本点之间的距离关系计算W的函数类型,本程序中选择的是“热核”
    options.k = 5;                              %近邻的个数
    options.t = 0.1;                              %热核参数,不能太小，太小导致计算出来的W都趋近于0，导致LPP报错
    W = constructW(fea,options);                
    % LPP算法进行降维
    npc=17;
    options.PCARatio = 1;                %The percentage of principal
                                         %component kept in【 the PCA
                                         %step. The percentage is
                                         %calculated based on the
                                         %eigenvalue. Default is 1
                                         %(100%, all the non-zero
                                         %eigenvalues will be kept.
    options.ReducedDim = npc;            %目标空间的维数，与options.PCARatio的值有关，当options.PCARatio=0.8时<=3维，当options.PCARatio=1时<=10
    [eigvector, eigvalue] = LPP(W, options, fea);
    Re_Source= fea*eigvector;    
    input_test = Target*eigvector;%测试样本进行LPP投影
    Re_Target = input_test;

%     % from CDSL other methods
%     batchS =Source;
%     batchT =Target;
%     [mappedA, mapping] = compute_mapping(batchS,'LPP',4);
%     W=mapping.M;
%     Re_Source=batchS*W;
%     Re_Target=batchT*W;

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

