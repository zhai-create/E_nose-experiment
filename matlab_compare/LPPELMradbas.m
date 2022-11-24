%%
clear all
clc
load A.mat
load index.mat

SL=2; %�Ӱ�
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';
%��һ��
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);


%% ����constructW����������ڣ�Ϊlpp�ṩ���ڹ�ϵ����
fea = Source;                      %������������룬ÿһ����һ��������
options = [];
options.NeighborMode = 'KNN';               %ѡ�������ڵķ������˴��õ���K���ڷ�                     
options.WeightMode = 'HeatKernel';          %����������֮��ľ����ϵ����W�ĺ�������,��������ѡ����ǡ��Ⱥˡ�
options.k = 5;                              %���ڵĸ���
options.t = 0.1;                              %�Ⱥ˲���,����̫С��̫С���¼��������W��������0������LPP����
W = constructW(fea,options);                
%% LPP�㷨���н�ά
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
input_train= fea*eigvector;    
input_test = Target*eigvector;%������������LPPͶӰ
for cishu=1:10
    clear max;
    clear result;
%% ELM
error2=0;
counter=0;
class_num=[1;5;10;15;20];
            ActivationFunction='radbas';
                    for t=1:5
                        Hnumbt=class_num(t,1);
                        for v=0:20
                            counter=counter+1;
                            Cv=10^v;
                            try
                            [TrainingAccuracy, TestingAccuracy] = fast_elm(input_train,Source_label,input_test,Target_label,1, Hnumbt, ActivationFunction,Cv);
                            catch
                                error2 = error2 +1;
                                TestingAccuracy =0;
                            end
                            testaccracy=TestingAccuracy;
                             result(counter,1)=Hnumbt;
                             result(counter,2)=v;
                             result(counter,3)=TestingAccuracy; 
                        end                 
                    end
[max,index]=max(result(:,3));  
maxpara=result(index,:);
newresult(cishu,:)=maxpara;
end
save LPPelm_radbas(2)

%%
clear all
clc
load A.mat
load index.mat

SL=3; %�Ӱ�
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';
%��һ��
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);


%% ����constructW����������ڣ�Ϊlpp�ṩ���ڹ�ϵ����
fea = Source;                      %������������룬ÿһ����һ��������
options = [];
options.NeighborMode = 'KNN';               %ѡ�������ڵķ������˴��õ���K���ڷ�                     
options.WeightMode = 'HeatKernel';          %����������֮��ľ����ϵ����W�ĺ�������,��������ѡ����ǡ��Ⱥˡ�
options.k = 5;                              %���ڵĸ���
options.t = 0.1;                              %�Ⱥ˲���,����̫С��̫С���¼��������W��������0������LPP����
W = constructW(fea,options);                
%% LPP�㷨���н�ά
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
input_train= fea*eigvector;    
input_test = Target*eigvector;%������������LPPͶӰ
for cishu=1:10
    clear max;
    clear result;
%% ELM
error2=0;
counter=0;
class_num=[1;5;10;15;20];
            ActivationFunction='radbas';
                    for t=1:5
                        Hnumbt=class_num(t,1);
                        for v=0:20
                            counter=counter+1;
                            Cv=10^v;
                            try
                            [TrainingAccuracy, TestingAccuracy] = fast_elm(input_train,Source_label,input_test,Target_label,1, Hnumbt, ActivationFunction,Cv);
                            catch
                                error2 = error2 +1;
                                TestingAccuracy =0;
                            end
                            testaccracy=TestingAccuracy;
                             result(counter,1)=Hnumbt;
                             result(counter,2)=v;
                             result(counter,3)=TestingAccuracy; 
                        end                 
                    end
[max,index]=max(result(:,3));  
maxpara=result(index,:);
newresult(cishu,:)=maxpara;
end
save LPPelm_radbas(3)

%%
clear all
clc
load A.mat
load index.mat

SL=4; %�Ӱ�
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';
%��һ��
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);


%% ����constructW����������ڣ�Ϊlpp�ṩ���ڹ�ϵ����
fea = Source;                      %������������룬ÿһ����һ��������
options = [];
options.NeighborMode = 'KNN';               %ѡ�������ڵķ������˴��õ���K���ڷ�                     
options.WeightMode = 'HeatKernel';          %����������֮��ľ����ϵ����W�ĺ�������,��������ѡ����ǡ��Ⱥˡ�
options.k = 5;                              %���ڵĸ���
options.t = 0.1;                              %�Ⱥ˲���,����̫С��̫С���¼��������W��������0������LPP����
W = constructW(fea,options);                
%% LPP�㷨���н�ά
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
input_train= fea*eigvector;    
input_test = Target*eigvector;%������������LPPͶӰ
for cishu=1:10
    clear max;
    clear result;
%% ELM
error2=0;
counter=0;
class_num=[1;5;10;15;20];
            ActivationFunction='radbas';
                    for t=1:5
                        Hnumbt=class_num(t,1);
                        for v=0:20
                            counter=counter+1;
                            Cv=10^v;
                            try
                            [TrainingAccuracy, TestingAccuracy] = fast_elm(input_train,Source_label,input_test,Target_label,1, Hnumbt, ActivationFunction,Cv);
                            catch
                                error2 = error2 +1;
                                TestingAccuracy =0;
                            end
                            testaccracy=TestingAccuracy;
                             result(counter,1)=Hnumbt;
                             result(counter,2)=v;
                             result(counter,3)=TestingAccuracy; 
                        end                 
                    end
[max,index]=max(result(:,3));  
maxpara=result(index,:);
newresult(cishu,:)=maxpara;
end
save LPPelm_radbas(4)

%%
clear all
clc
load A.mat
load index.mat

SL=5; %�Ӱ�
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';
%��һ��
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);


%% ����constructW����������ڣ�Ϊlpp�ṩ���ڹ�ϵ����
fea = Source;                      %������������룬ÿһ����һ��������
options = [];
options.NeighborMode = 'KNN';               %ѡ�������ڵķ������˴��õ���K���ڷ�                     
options.WeightMode = 'HeatKernel';          %����������֮��ľ����ϵ����W�ĺ�������,��������ѡ����ǡ��Ⱥˡ�
options.k = 5;                              %���ڵĸ���
options.t = 0.1;                              %�Ⱥ˲���,����̫С��̫С���¼��������W��������0������LPP����
W = constructW(fea,options);                
%% LPP�㷨���н�ά
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
input_train= fea*eigvector;    
input_test = Target*eigvector;%������������LPPͶӰ
for cishu=1:10
    clear max;
    clear result;
%% ELM
error2=0;
counter=0;
class_num=[1;5;10;15;20];
            ActivationFunction='radbas';
                    for t=1:5
                        Hnumbt=class_num(t,1);
                        for v=0:20
                            counter=counter+1;
                            Cv=10^v;
                            try
                            [TrainingAccuracy, TestingAccuracy] = fast_elm(input_train,Source_label,input_test,Target_label,1, Hnumbt, ActivationFunction,Cv);
                            catch
                                error2 = error2 +1;
                                TestingAccuracy =0;
                            end
                            testaccracy=TestingAccuracy;
                             result(counter,1)=Hnumbt;
                             result(counter,2)=v;
                             result(counter,3)=TestingAccuracy; 
                        end                 
                    end
[max,index]=max(result(:,3));  
maxpara=result(index,:);
newresult(cishu,:)=maxpara;
end
save LPPelm_radbas(5)

%%
clear all
clc
load A.mat
load index.mat

SL=6; %�Ӱ�
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';
%��һ��
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);


%% ����constructW����������ڣ�Ϊlpp�ṩ���ڹ�ϵ����
fea = Source;                      %������������룬ÿһ����һ��������
options = [];
options.NeighborMode = 'KNN';               %ѡ�������ڵķ������˴��õ���K���ڷ�                     
options.WeightMode = 'HeatKernel';          %����������֮��ľ����ϵ����W�ĺ�������,��������ѡ����ǡ��Ⱥˡ�
options.k = 5;                              %���ڵĸ���
options.t = 0.1;                              %�Ⱥ˲���,����̫С��̫С���¼��������W��������0������LPP����
W = constructW(fea,options);                
%% LPP�㷨���н�ά
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
input_train= fea*eigvector;    
input_test = Target*eigvector;%������������LPPͶӰ
for cishu=1:10
    clear max;
    clear result;
%% ELM
error2=0;
counter=0;
class_num=[1;5;10;15;20];
            ActivationFunction='radbas';
                    for t=1:5
                        Hnumbt=class_num(t,1);
                        for v=0:20
                            counter=counter+1;
                            Cv=10^v;
                            try
                            [TrainingAccuracy, TestingAccuracy] = fast_elm(input_train,Source_label,input_test,Target_label,1, Hnumbt, ActivationFunction,Cv);
                            catch
                                error2 = error2 +1;
                                TestingAccuracy =0;
                            end
                            testaccracy=TestingAccuracy;
                             result(counter,1)=Hnumbt;
                             result(counter,2)=v;
                             result(counter,3)=TestingAccuracy; 
                        end                 
                    end
[max,index]=max(result(:,3));  
maxpara=result(index,:);
newresult(cishu,:)=maxpara;
end
save LPPelm_radbas(6)

%%
clear all
clc
load A.mat
load index.mat

SL=7; %�Ӱ�
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';
%��һ��
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);


%% ����constructW����������ڣ�Ϊlpp�ṩ���ڹ�ϵ����
fea = Source;                      %������������룬ÿһ����һ��������
options = [];
options.NeighborMode = 'KNN';               %ѡ�������ڵķ������˴��õ���K���ڷ�                     
options.WeightMode = 'HeatKernel';          %����������֮��ľ����ϵ����W�ĺ�������,��������ѡ����ǡ��Ⱥˡ�
options.k = 5;                              %���ڵĸ���
options.t = 0.1;                              %�Ⱥ˲���,����̫С��̫С���¼��������W��������0������LPP����
W = constructW(fea,options);                
%% LPP�㷨���н�ά
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
input_train= fea*eigvector;    
input_test = Target*eigvector;%������������LPPͶӰ
for cishu=1:10
    clear max;
    clear result;
%% ELM
error2=0;
counter=0;
class_num=[1;5;10;15;20];
            ActivationFunction='radbas';
                    for t=1:5
                        Hnumbt=class_num(t,1);
                        for v=0:20
                            counter=counter+1;
                            Cv=10^v;
                            try
                            [TrainingAccuracy, TestingAccuracy] = fast_elm(input_train,Source_label,input_test,Target_label,1, Hnumbt, ActivationFunction,Cv);
                            catch
                                error2 = error2 +1;
                                TestingAccuracy =0;
                            end
                            testaccracy=TestingAccuracy;
                             result(counter,1)=Hnumbt;
                             result(counter,2)=v;
                             result(counter,3)=TestingAccuracy; 
                        end                 
                    end
[max,index]=max(result(:,3));  
maxpara=result(index,:);
newresult(cishu,:)=maxpara;
end
save LPPelm_radbas(7)

%%
clear all
clc
load A.mat
load index.mat

SL=8; %�Ӱ�
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';
%��һ��
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);


%% ����constructW����������ڣ�Ϊlpp�ṩ���ڹ�ϵ����
fea = Source;                      %������������룬ÿһ����һ��������
options = [];
options.NeighborMode = 'KNN';               %ѡ�������ڵķ������˴��õ���K���ڷ�                     
options.WeightMode = 'HeatKernel';          %����������֮��ľ����ϵ����W�ĺ�������,��������ѡ����ǡ��Ⱥˡ�
options.k = 5;                              %���ڵĸ���
options.t = 0.1;                              %�Ⱥ˲���,����̫С��̫С���¼��������W��������0������LPP����
W = constructW(fea,options);                
%% LPP�㷨���н�ά
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
input_train= fea*eigvector;    
input_test = Target*eigvector;%������������LPPͶӰ
for cishu=1:10
    clear max;
    clear result;
%% ELM
error2=0;
counter=0;
class_num=[1;5;10;15;20];
            ActivationFunction='radbas';
                    for t=1:5
                        Hnumbt=class_num(t,1);
                        for v=0:20
                            counter=counter+1;
                            Cv=10^v;
                            try
                            [TrainingAccuracy, TestingAccuracy] = fast_elm(input_train,Source_label,input_test,Target_label,1, Hnumbt, ActivationFunction,Cv);
                            catch
                                error2 = error2 +1;
                                TestingAccuracy =0;
                            end
                            testaccracy=TestingAccuracy;
                             result(counter,1)=Hnumbt;
                             result(counter,2)=v;
                             result(counter,3)=TestingAccuracy; 
                        end                 
                    end
[max,index]=max(result(:,3));  
maxpara=result(index,:);
newresult(cishu,:)=maxpara;
end
save LPPelm_radbas(8)

%%
clear all
clc
load A.mat
load index.mat

SL=9; %�Ӱ�
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';
%��һ��
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);


%% ����constructW����������ڣ�Ϊlpp�ṩ���ڹ�ϵ����
fea = Source;                      %������������룬ÿһ����һ��������
options = [];
options.NeighborMode = 'KNN';               %ѡ�������ڵķ������˴��õ���K���ڷ�                     
options.WeightMode = 'HeatKernel';          %����������֮��ľ����ϵ����W�ĺ�������,��������ѡ����ǡ��Ⱥˡ�
options.k = 5;                              %���ڵĸ���
options.t = 0.1;                              %�Ⱥ˲���,����̫С��̫С���¼��������W��������0������LPP����
W = constructW(fea,options);                
%% LPP�㷨���н�ά
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
input_train= fea*eigvector;    
input_test = Target*eigvector;%������������LPPͶӰ
for cishu=1:10
    clear max;
    clear result;
%% ELM
error2=0;
counter=0;
class_num=[1;5;10;15;20];
            ActivationFunction='radbas';
                    for t=1:5
                        Hnumbt=class_num(t,1);
                        for v=0:20
                            counter=counter+1;
                            Cv=10^v;
                            try
                            [TrainingAccuracy, TestingAccuracy] = fast_elm(input_train,Source_label,input_test,Target_label,1, Hnumbt, ActivationFunction,Cv);
                            catch
                                error2 = error2 +1;
                                TestingAccuracy =0;
                            end
                            testaccracy=TestingAccuracy;
                             result(counter,1)=Hnumbt;
                             result(counter,2)=v;
                             result(counter,3)=TestingAccuracy; 
                        end                 
                    end
[max,index]=max(result(:,3));  
maxpara=result(index,:);
newresult(cishu,:)=maxpara;
end
save LPPelm_radbas(9)

%%
clear all
clc
load A.mat
load index.mat

SL=10; %�Ӱ�
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';
%��һ��
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);


%% ����constructW����������ڣ�Ϊlpp�ṩ���ڹ�ϵ����
fea = Source;                      %������������룬ÿһ����һ��������
options = [];
options.NeighborMode = 'KNN';               %ѡ�������ڵķ������˴��õ���K���ڷ�                     
options.WeightMode = 'HeatKernel';          %����������֮��ľ����ϵ����W�ĺ�������,��������ѡ����ǡ��Ⱥˡ�
options.k = 5;                              %���ڵĸ���
options.t = 0.1;                              %�Ⱥ˲���,����̫С��̫С���¼��������W��������0������LPP����
W = constructW(fea,options);                
%% LPP�㷨���н�ά
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
input_train= fea*eigvector;    
input_test = Target*eigvector;%������������LPPͶӰ
for cishu=1:10
    clear max;
    clear result;
%% ELM
error2=0;
counter=0;
class_num=[1;5;10;15;20];
            ActivationFunction='radbas';
                    for t=1:5
                        Hnumbt=class_num(t,1);
                        for v=0:20
                            counter=counter+1;
                            Cv=10^v;
                            try
                            [TrainingAccuracy, TestingAccuracy] = fast_elm(input_train,Source_label,input_test,Target_label,1, Hnumbt, ActivationFunction,Cv);
                            catch
                                error2 = error2 +1;
                                TestingAccuracy =0;
                            end
                            testaccracy=TestingAccuracy;
                             result(counter,1)=Hnumbt;
                             result(counter,2)=v;
                             result(counter,3)=TestingAccuracy; 
                        end                 
                    end
[max,index]=max(result(:,3));  
maxpara=result(index,:);
newresult(cishu,:)=maxpara;
end
save LPPelm_radbas(10)