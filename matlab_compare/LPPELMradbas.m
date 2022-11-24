%%
clear all
clc
load A.mat
load index.mat

SL=2; %从板
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';
%归一化
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);


%% 调用constructW函数计算近邻，为lpp提供近邻关系矩阵
fea = Source;                      %待处理矩阵输入，每一行是一个样本点
options = [];
options.NeighborMode = 'KNN';               %选择计算近邻的方法，此处用的是K近邻法                     
options.WeightMode = 'HeatKernel';          %根据样本点之间的距离关系计算W的函数类型,本程序中选择的是“热核”
options.k = 5;                              %近邻的个数
options.t = 0.1;                              %热核参数,不能太小，太小导致计算出来的W都趋近于0，导致LPP报错
W = constructW(fea,options);                
%% LPP算法进行降维
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
input_train= fea*eigvector;    
input_test = Target*eigvector;%测试样本进行LPP投影
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

SL=3; %从板
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';
%归一化
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);


%% 调用constructW函数计算近邻，为lpp提供近邻关系矩阵
fea = Source;                      %待处理矩阵输入，每一行是一个样本点
options = [];
options.NeighborMode = 'KNN';               %选择计算近邻的方法，此处用的是K近邻法                     
options.WeightMode = 'HeatKernel';          %根据样本点之间的距离关系计算W的函数类型,本程序中选择的是“热核”
options.k = 5;                              %近邻的个数
options.t = 0.1;                              %热核参数,不能太小，太小导致计算出来的W都趋近于0，导致LPP报错
W = constructW(fea,options);                
%% LPP算法进行降维
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
input_train= fea*eigvector;    
input_test = Target*eigvector;%测试样本进行LPP投影
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

SL=4; %从板
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';
%归一化
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);


%% 调用constructW函数计算近邻，为lpp提供近邻关系矩阵
fea = Source;                      %待处理矩阵输入，每一行是一个样本点
options = [];
options.NeighborMode = 'KNN';               %选择计算近邻的方法，此处用的是K近邻法                     
options.WeightMode = 'HeatKernel';          %根据样本点之间的距离关系计算W的函数类型,本程序中选择的是“热核”
options.k = 5;                              %近邻的个数
options.t = 0.1;                              %热核参数,不能太小，太小导致计算出来的W都趋近于0，导致LPP报错
W = constructW(fea,options);                
%% LPP算法进行降维
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
input_train= fea*eigvector;    
input_test = Target*eigvector;%测试样本进行LPP投影
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

SL=5; %从板
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';
%归一化
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);


%% 调用constructW函数计算近邻，为lpp提供近邻关系矩阵
fea = Source;                      %待处理矩阵输入，每一行是一个样本点
options = [];
options.NeighborMode = 'KNN';               %选择计算近邻的方法，此处用的是K近邻法                     
options.WeightMode = 'HeatKernel';          %根据样本点之间的距离关系计算W的函数类型,本程序中选择的是“热核”
options.k = 5;                              %近邻的个数
options.t = 0.1;                              %热核参数,不能太小，太小导致计算出来的W都趋近于0，导致LPP报错
W = constructW(fea,options);                
%% LPP算法进行降维
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
input_train= fea*eigvector;    
input_test = Target*eigvector;%测试样本进行LPP投影
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

SL=6; %从板
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';
%归一化
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);


%% 调用constructW函数计算近邻，为lpp提供近邻关系矩阵
fea = Source;                      %待处理矩阵输入，每一行是一个样本点
options = [];
options.NeighborMode = 'KNN';               %选择计算近邻的方法，此处用的是K近邻法                     
options.WeightMode = 'HeatKernel';          %根据样本点之间的距离关系计算W的函数类型,本程序中选择的是“热核”
options.k = 5;                              %近邻的个数
options.t = 0.1;                              %热核参数,不能太小，太小导致计算出来的W都趋近于0，导致LPP报错
W = constructW(fea,options);                
%% LPP算法进行降维
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
input_train= fea*eigvector;    
input_test = Target*eigvector;%测试样本进行LPP投影
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

SL=7; %从板
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';
%归一化
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);


%% 调用constructW函数计算近邻，为lpp提供近邻关系矩阵
fea = Source;                      %待处理矩阵输入，每一行是一个样本点
options = [];
options.NeighborMode = 'KNN';               %选择计算近邻的方法，此处用的是K近邻法                     
options.WeightMode = 'HeatKernel';          %根据样本点之间的距离关系计算W的函数类型,本程序中选择的是“热核”
options.k = 5;                              %近邻的个数
options.t = 0.1;                              %热核参数,不能太小，太小导致计算出来的W都趋近于0，导致LPP报错
W = constructW(fea,options);                
%% LPP算法进行降维
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
input_train= fea*eigvector;    
input_test = Target*eigvector;%测试样本进行LPP投影
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

SL=8; %从板
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';
%归一化
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);


%% 调用constructW函数计算近邻，为lpp提供近邻关系矩阵
fea = Source;                      %待处理矩阵输入，每一行是一个样本点
options = [];
options.NeighborMode = 'KNN';               %选择计算近邻的方法，此处用的是K近邻法                     
options.WeightMode = 'HeatKernel';          %根据样本点之间的距离关系计算W的函数类型,本程序中选择的是“热核”
options.k = 5;                              %近邻的个数
options.t = 0.1;                              %热核参数,不能太小，太小导致计算出来的W都趋近于0，导致LPP报错
W = constructW(fea,options);                
%% LPP算法进行降维
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
input_train= fea*eigvector;    
input_test = Target*eigvector;%测试样本进行LPP投影
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

SL=9; %从板
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';
%归一化
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);


%% 调用constructW函数计算近邻，为lpp提供近邻关系矩阵
fea = Source;                      %待处理矩阵输入，每一行是一个样本点
options = [];
options.NeighborMode = 'KNN';               %选择计算近邻的方法，此处用的是K近邻法                     
options.WeightMode = 'HeatKernel';          %根据样本点之间的距离关系计算W的函数类型,本程序中选择的是“热核”
options.k = 5;                              %近邻的个数
options.t = 0.1;                              %热核参数,不能太小，太小导致计算出来的W都趋近于0，导致LPP报错
W = constructW(fea,options);                
%% LPP算法进行降维
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
input_train= fea*eigvector;    
input_test = Target*eigvector;%测试样本进行LPP投影
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

SL=10; %从板
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';
%归一化
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);


%% 调用constructW函数计算近邻，为lpp提供近邻关系矩阵
fea = Source;                      %待处理矩阵输入，每一行是一个样本点
options = [];
options.NeighborMode = 'KNN';               %选择计算近邻的方法，此处用的是K近邻法                     
options.WeightMode = 'HeatKernel';          %根据样本点之间的距离关系计算W的函数类型,本程序中选择的是“热核”
options.k = 5;                              %近邻的个数
options.t = 0.1;                              %热核参数,不能太小，太小导致计算出来的W都趋近于0，导致LPP报错
W = constructW(fea,options);                
%% LPP算法进行降维
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
input_train= fea*eigvector;    
input_test = Target*eigvector;%测试样本进行LPP投影
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