%%10块板子数据，核函数：radbas
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

[pc,score,latent,tsquare] = pca(Target);
P=pc(:,1);
Re_Target = Target-Target*P*P';

class_num=[1;5;10;15;20];
error1=0;
counter=0;
ActivationFunction='radbas';
for t=1:5
    Hnumbt=class_num(t,1);
    for v=0:20
        counter=counter+1;
        Cv=10^v;
        try
         [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,Re_Target,Target_label,1, Hnumbt, ActivationFunction,Cv);
        catch
              error1 =error1 + 1;
              TestingAccuracy = 0;
        end
         result(counter,1)=t;
         result(counter,2)=v;
         result(counter,3)=TestingAccuracy;                            
    end                 
end
[max,index]=max(result(:,3));
save CCPCAradbas(1-2)

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

[pc,score,latent,tsquare] = pca(Target);
P=pc(:,1);
Re_Target = Target-Target*P*P';
class_num=[1;5;10;15;20];
error1=0;
counter=0;
ActivationFunction='radbas';
for t=1:5
    Hnumbt=class_num(t,1);
    for v=0:20
        counter=counter+1;
        Cv=10^v;
        try
         [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,Re_Target,Target_label,1, Hnumbt, ActivationFunction,Cv);
        catch
              error1 =error1 + 1;
              TestingAccuracy = 0;
        end
         result(counter,1)=t;
         result(counter,2)=v;
         result(counter,3)=TestingAccuracy;                            
    end                 
end
[max,index]=max(result(:,3));
save CCPCAradbas(1-3)

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

[pc,score,latent,tsquare] = pca(Target);
P=pc(:,1);
Re_Target = Target-Target*P*P';

class_num=[1;5;10;15;20];
error1=0;
counter=0;
ActivationFunction='radbas';
for t=1:5
    Hnumbt=class_num(t,1);
    for v=0:20
        counter=counter+1;
        Cv=10^v;
        try
         [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,Re_Target,Target_label,1, Hnumbt, ActivationFunction,Cv);
        catch
              error1 =error1 + 1;
              TestingAccuracy = 0;
        end
         result(counter,1)=t;
         result(counter,2)=v;
         result(counter,3)=TestingAccuracy;                            
    end                 
end
[max,index]=max(result(:,3));
save CCPCAradbas(1-4)

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

[pc,score,latent,tsquare] = pca(Target);
P=pc(:,1);
Re_Target = Target-Target*P*P';

class_num=[1;5;10;15;20];
error1=0;
counter=0;
ActivationFunction='radbas';
for t=1:5
    Hnumbt=class_num(t,1);
    for v=0:20
        counter=counter+1;
        Cv=10^v;
        try
         [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,Re_Target,Target_label,1, Hnumbt, ActivationFunction,Cv);
        catch
              error1 =error1 + 1;
              TestingAccuracy = 0;
        end
         result(counter,1)=t;
         result(counter,2)=v;
         result(counter,3)=TestingAccuracy;                            
    end                 
end
[max,index]=max(result(:,3));
save CCPCAradbas(1-5)

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

[pc,score,latent,tsquare] = pca(Target);
P=pc(:,1);
Re_Target = Target-Target*P*P';

class_num=[1;5;10;15;20];
error1=0;
counter=0;
ActivationFunction='radbas';
for t=1:5
    Hnumbt=class_num(t,1);
    for v=0:20
        counter=counter+1;
        Cv=10^v;
        try
         [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,Re_Target,Target_label,1, Hnumbt, ActivationFunction,Cv);
        catch
              error1 =error1 + 1;
              TestingAccuracy = 0;
        end
         result(counter,1)=t;
         result(counter,2)=v;
         result(counter,3)=TestingAccuracy;                            
    end                 
end
[max,index]=max(result(:,3));
save CCPCAradbas(1-6)

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

[pc,score,latent,tsquare] = pca(Target);
P=pc(:,1);
Re_Target = Target-Target*P*P';

class_num=[1;5;10;15;20];
error1=0;
counter=0;
ActivationFunction='radbas';
for t=1:5
    Hnumbt=class_num(t,1);
    for v=0:20
        counter=counter+1;
        Cv=10^v;
        try
         [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,Re_Target,Target_label,1, Hnumbt, ActivationFunction,Cv);
        catch
              error1 =error1 + 1;
              TestingAccuracy = 0;
        end
         result(counter,1)=t;
         result(counter,2)=v;
         result(counter,3)=TestingAccuracy;                            
    end                 
end
[max,index]=max(result(:,3));
save CCPCAradbas(1-7)

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

[pc,score,latent,tsquare] = pca(Target);
P=pc(:,1);
Re_Target = Target-Target*P*P';

class_num=[1;5;10;15;20];
error1=0;
counter=0;
ActivationFunction='radbas';
for t=1:5
    Hnumbt=class_num(t,1);
    for v=0:20
        counter=counter+1;
        Cv=10^v;
        try
         [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,Re_Target,Target_label,1, Hnumbt, ActivationFunction,Cv);
        catch
              error1 =error1 + 1;
              TestingAccuracy = 0;
        end
         result(counter,1)=t;
         result(counter,2)=v;
         result(counter,3)=TestingAccuracy;                            
    end                 
end
[max,index]=max(result(:,3));
save CCPCAradbas(1-8)

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

[pc,score,latent,tsquare] = pca(Target);
P=pc(:,1);
Re_Target = Target-Target*P*P';

class_num=[1;5;10;15;20];
error1=0;
counter=0;
ActivationFunction='radbas';
for t=1:5
    Hnumbt=class_num(t,1);
    for v=0:20
        counter=counter+1;
        Cv=10^v;
        try
         [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,Re_Target,Target_label,1, Hnumbt, ActivationFunction,Cv);
        catch
              error1 =error1 + 1;
              TestingAccuracy = 0;
        end
         result(counter,1)=t;
         result(counter,2)=v;
         result(counter,3)=TestingAccuracy;                            
    end                 
end
[max,index]=max(result(:,3));
save CCPCAradbas(1-9)
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

[pc,score,latent,tsquare] = pca(Target);
P=pc(:,1);
Re_Target = Target-Target*P*P';

class_num=[1;5;10;15;20];
error1=0;
counter=0;
ActivationFunction='radbas';
for t=1:5
    Hnumbt=class_num(t,1);
    for v=0:20
        counter=counter+1;
        Cv=10^v;
        try
         [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,Re_Target,Target_label,1, Hnumbt, ActivationFunction,Cv);
        catch
              error1 =error1 + 1;
              TestingAccuracy = 0;
        end
         result(counter,1)=t;
         result(counter,2)=v;
         result(counter,3)=TestingAccuracy;                            
    end                 
end
[max,index]=max(result(:,3));
save CCPCAradbas(1-10)
