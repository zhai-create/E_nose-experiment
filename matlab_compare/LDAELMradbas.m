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



%% 使用LDA进行数据降维
options = [];
options.Regu=1;
options.Fisherface = 1;
[eigvector, eigvalue] = LDA(Source_label,options,Source);

input_train=Source*eigvector;      
input_test=Target*eigvector;   
              
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
save LDAelm_radbas(2)

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



%% 使用LDA进行数据降维
options = [];
options.Regu=1;
options.Fisherface = 1;
[eigvector, eigvalue] = LDA(Source_label,options,Source);

input_train=Source*eigvector;      
input_test=Target*eigvector;   
              
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
save LDAelm_radbas(3)

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



%% 使用LDA进行数据降维
options = [];
options.Regu=1;
options.Fisherface = 1;
[eigvector, eigvalue] = LDA(Source_label,options,Source);

input_train=Source*eigvector;      
input_test=Target*eigvector;   
              
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
save LDAelm_radbas(4)

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



%% 使用LDA进行数据降维
options = [];
options.Regu=1;
options.Fisherface = 1;
[eigvector, eigvalue] = LDA(Source_label,options,Source);

input_train=Source*eigvector;      
input_test=Target*eigvector;   
              
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
save LDAelm_radbas(5)

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



%% 使用LDA进行数据降维
options = [];
options.Regu=1;
options.Fisherface = 1;
[eigvector, eigvalue] = LDA(Source_label,options,Source);

input_train=Source*eigvector;      
input_test=Target*eigvector;   
              
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
save LDAelm_radbas(6)

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



%% 使用LDA进行数据降维
options = [];
options.Regu=1;
options.Fisherface = 1;
[eigvector, eigvalue] = LDA(Source_label,options,Source);

input_train=Source*eigvector;      
input_test=Target*eigvector;   
              
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
save LDAelm_radbas(7)

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



%% 使用LDA进行数据降维
options = [];
options.Regu=1;
options.Fisherface = 1;
[eigvector, eigvalue] = LDA(Source_label,options,Source);

input_train=Source*eigvector;      
input_test=Target*eigvector;   
              
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
save LDAelm_radbas(8)

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



%% 使用LDA进行数据降维
options = [];
options.Regu=1;
options.Fisherface = 1;
[eigvector, eigvalue] = LDA(Source_label,options,Source);

input_train=Source*eigvector;      
input_test=Target*eigvector;   
              
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
save LDAelm_radbas(9)

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



%% 使用LDA进行数据降维
options = [];
options.Regu=1;
options.Fisherface = 1;
[eigvector, eigvalue] = LDA(Source_label,options,Source);

input_train=Source*eigvector;      
input_test=Target*eigvector;   
              
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
save LDAelm_radbas(10)