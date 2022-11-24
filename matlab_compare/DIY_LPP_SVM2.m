% DIY_LPP_SVM dataset 2
%% setting 1
acc1 = zeros(1,2);
for SL = 2:3
    clear -regexp [^SL,^acc1,^acc2]
    clc
    load('D:\A_\Enose_datasets\4����ԭʼ����\datasetB_prep.mat')
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
    %��һ��
    Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
    Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
    
    % ����constructW����������ڣ�Ϊlpp�ṩ���ڹ�ϵ����
    fea = Source;                      %������������룬ÿһ����һ��������
    options = [];
    options.NeighborMode = 'KNN';               %ѡ�������ڵķ������˴��õ���K���ڷ�                     
    options.WeightMode = 'HeatKernel';          %����������֮��ľ����ϵ����W�ĺ�������,��������ѡ����ǡ��Ⱥˡ�
    options.k = 5;                              %���ڵĸ���
    options.t = 0.1;                              %�Ⱥ˲���,����̫С��̫С���¼��������W��������0������LPP����
    W = constructW(fea,options);                
    % LPP�㷨���н�ά
    npc=17;
    options.PCARatio = 1;                %The percentage of principal
                                         %component kept in�� the PCA
                                         %step. The percentage is
                                         %calculated based on the
                                         %eigenvalue. Default is 1
                                         %(100%, all the non-zero
                                         %eigenvalues will be kept.
    options.ReducedDim = npc;            %Ŀ��ռ��ά������options.PCARatio��ֵ�йأ���options.PCARatio=0.8ʱ<=3ά����options.PCARatio=1ʱ<=10
    [eigvector, eigvalue] = LPP(W, options, fea);
    Re_Source= fea*eigvector;    
    input_test = Target*eigvector;%������������LPPͶӰ
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
                     cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain����
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
    load('D:\A_\Enose_datasets\4����ԭʼ����\datasetB_prep.mat')
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
    %��һ��
    Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
    Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
    
    % ����constructW����������ڣ�Ϊlpp�ṩ���ڹ�ϵ����
    fea = Source;                      %������������룬ÿһ����һ��������
    options = [];
    options.NeighborMode = 'KNN';               %ѡ�������ڵķ������˴��õ���K���ڷ�                     
    options.WeightMode = 'HeatKernel';          %����������֮��ľ����ϵ����W�ĺ�������,��������ѡ����ǡ��Ⱥˡ�
    options.k = 5;                              %���ڵĸ���
    options.t = 0.1;                              %�Ⱥ˲���,����̫С��̫С���¼��������W��������0������LPP����
    W = constructW(fea,options);                
    % LPP�㷨���н�ά
    npc=17;
    options.PCARatio = 1;                %The percentage of principal
                                         %component kept in�� the PCA
                                         %step. The percentage is
                                         %calculated based on the
                                         %eigenvalue. Default is 1
                                         %(100%, all the non-zero
                                         %eigenvalues will be kept.
    options.ReducedDim = npc;            %Ŀ��ռ��ά������options.PCARatio��ֵ�йأ���options.PCARatio=0.8ʱ<=3ά����options.PCARatio=1ʱ<=10
    [eigvector, eigvalue] = LPP(W, options, fea);
    Re_Source= fea*eigvector;    
    input_test = Target*eigvector;%������������LPPͶӰ
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
                     cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain����
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

