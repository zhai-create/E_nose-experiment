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

%% 使用PCA进行数据降维
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
input_train=SCORE_train(:,1:npc);%训练样本进行PCA       

%测试样本进行PCA投影
SCORE_test= bsxfun(@minus,Target,mean(Source,1))*COEFF_train;    %测试样本投影到训练样本确定的PCA空间
input_test=SCORE_test(:,1:npc); 

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
                             result(counter,4)=npc; 
                        end                 
                    end
[max,index]=max(result(:,3));  
maxpara=result(index,:);
newresult(cishu,:)=maxpara;
end
save pcaelm_radbas(2)

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

%% 使用PCA进行数据降维
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
input_train=SCORE_train(:,1:npc);%训练样本进行PCA       

%测试样本进行PCA投影
SCORE_test= bsxfun(@minus,Target,mean(Source,1))*COEFF_train;    %测试样本投影到训练样本确定的PCA空间
input_test=SCORE_test(:,1:npc); 
               
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
                             result(counter,4)=npc; 
                        end                 
                    end
[max,index]=max(result(:,3));  
maxpara=result(index,:);
newresult(cishu,:)=maxpara;
end
save pcaelm_radbas(3)

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

%% 使用PCA进行数据降维
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
input_train=SCORE_train(:,1:npc);%训练样本进行PCA       

%测试样本进行PCA投影
SCORE_test= bsxfun(@minus,Target,mean(Source,1))*COEFF_train;    %测试样本投影到训练样本确定的PCA空间
input_test=SCORE_test(:,1:npc); 
               
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
                             result(counter,4)=npc; 
                        end                 
                    end
[max,index]=max(result(:,3));  
maxpara=result(index,:);
newresult(cishu,:)=maxpara;
end
save pcaelm_radbas(4)

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

%% 使用PCA进行数据降维
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
input_train=SCORE_train(:,1:npc);%训练样本进行PCA       

%测试样本进行PCA投影
SCORE_test= bsxfun(@minus,Target,mean(Source,1))*COEFF_train;    %测试样本投影到训练样本确定的PCA空间
input_test=SCORE_test(:,1:npc); 
               
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
                             result(counter,4)=npc; 
                        end                 
                    end
[max,index]=max(result(:,3));  
maxpara=result(index,:);
newresult(cishu,:)=maxpara;
end
save pcaelm_radbas(5)

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

%% 使用PCA进行数据降维
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
input_train=SCORE_train(:,1:npc);%训练样本进行PCA       

%测试样本进行PCA投影
SCORE_test= bsxfun(@minus,Target,mean(Source,1))*COEFF_train;    %测试样本投影到训练样本确定的PCA空间
input_test=SCORE_test(:,1:npc); 
               
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
                             result(counter,4)=npc; 
                        end                 
                    end
[max,index]=max(result(:,3));  
maxpara=result(index,:);
newresult(cishu,:)=maxpara;
end
save pcaelm_radbas(6)

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

%% 使用PCA进行数据降维
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
input_train=SCORE_train(:,1:npc);%训练样本进行PCA       

%测试样本进行PCA投影
SCORE_test= bsxfun(@minus,Target,mean(Source,1))*COEFF_train;    %测试样本投影到训练样本确定的PCA空间
input_test=SCORE_test(:,1:npc); 
               
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
                             result(counter,4)=npc; 
                        end                 
                    end
[max,index]=max(result(:,3));  
maxpara=result(index,:);
newresult(cishu,:)=maxpara;
end
save pcaelm_radbas(7)

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

%% 使用PCA进行数据降维
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
input_train=SCORE_train(:,1:npc);%训练样本进行PCA       

%测试样本进行PCA投影
SCORE_test= bsxfun(@minus,Target,mean(Source,1))*COEFF_train;    %测试样本投影到训练样本确定的PCA空间
input_test=SCORE_test(:,1:npc); 
               
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
                             result(counter,4)=npc; 
                        end                 
                    end
[max,index]=max(result(:,3));  
maxpara=result(index,:);
newresult(cishu,:)=maxpara;
end
save pcaelm_radbas(8)

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

%% 使用PCA进行数据降维
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
input_train=SCORE_train(:,1:npc);%训练样本进行PCA       

%测试样本进行PCA投影
SCORE_test= bsxfun(@minus,Target,mean(Source,1))*COEFF_train;    %测试样本投影到训练样本确定的PCA空间
input_test=SCORE_test(:,1:npc); 
               
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
                             result(counter,4)=npc; 
                        end                 
                    end
[max,index]=max(result(:,3));  
maxpara=result(index,:);
newresult(cishu,:)=maxpara;
end
save pcaelm_radbas(9)

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

%% 使用PCA进行数据降维
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
input_train=SCORE_train(:,1:npc);%训练样本进行PCA       

%测试样本进行PCA投影
SCORE_test= bsxfun(@minus,Target,mean(Source,1))*COEFF_train;    %测试样本投影到训练样本确定的PCA空间
input_test=SCORE_test(:,1:npc); 
               
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
                             result(counter,4)=npc; 
                        end                 
                    end
[max,index]=max(result(:,3));  
maxpara=result(index,:);
newresult(cishu,:)=maxpara;
end
save pcaelm_radbas(10)