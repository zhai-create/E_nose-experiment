%% setting1
%% 
clear all
clc
%% 导入数据
load A.mat
load index.mat
SL=2; %从板
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';

%% 数据归一化
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
%中心化
for i=1: size(Source,1)
    Xmaster(i,:) = Source (i,:) - mean(Source);
end

for i=1: size(Target,1)
    Xslave(i,:) = Target (i,:) - mean(Target);
end
%% 样本子集选择
Num=30;
Hmaster=Xmaster*Xmaster';
Ht_master=diag(Hmaster);
[Amaster,Bmaster]=sort(Ht_master,'descend' );
Smaster_select=Xmaster(Bmaster(1:Num,:),:);

Hslave=Xslave*Xslave';
Ht_slave=diag(Hslave);
[Aslave,Bslave]=sort(Ht_slave,'descend' );
Sslave_select=Xslave(Bslave(1:Num,:),:);

%%  GLSW  
A = (Smaster_select - Sslave_select);
[U,S,V] = svd(A);
D2 = S'*S;
W = sqrt(D2./1 + eye(size(Sslave_select,2),size(Sslave_select,2)));
G=V*inv(W)*V';
T = Target*G;

%% ELM
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
         [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,T,Target_label,1, Hnumbt, ActivationFunction,Cv);
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
save radbas(1-2)

%% 
clear all
clc
%% 导入数据
load A.mat
load index.mat
SL=3; %从板
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';

%% 数据归一化
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
%中心化
for i=1: size(Source,1)
    Xmaster(i,:) = Source (i,:) - mean(Source);
end

for i=1: size(Target,1)
    Xslave(i,:) = Target (i,:) - mean(Target);
end
%% 样本子集选择
Num=30;
Hmaster=Xmaster*Xmaster';
Ht_master=diag(Hmaster);
[Amaster,Bmaster]=sort(Ht_master,'descend' );
Smaster_select=Xmaster(Bmaster(1:Num,:),:);

Hslave=Xslave*Xslave';
Ht_slave=diag(Hslave);
[Aslave,Bslave]=sort(Ht_slave,'descend' );
Sslave_select=Xslave(Bslave(1:Num,:),:);

%%  GLSW  
A = (Smaster_select - Sslave_select);
[U,S,V] = svd(A);
D2 = S'*S;
W = sqrt(D2./1 + eye(size(Sslave_select,2),size(Sslave_select,2)));
G=V*inv(W)*V';
T = Target*G;

%% ELM
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
         [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,T,Target_label,1, Hnumbt, ActivationFunction,Cv);
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
save radbas(1-3)


%% 
clear all
clc
%% 导入数据
load A.mat
load index.mat
SL=4; %从板
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';

%% 数据归一化
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
%中心化
for i=1: size(Source,1)
    Xmaster(i,:) = Source (i,:) - mean(Source);
end

for i=1: size(Target,1)
    Xslave(i,:) = Target (i,:) - mean(Target);
end
%% 样本子集选择
Num=30;
Hmaster=Xmaster*Xmaster';
Ht_master=diag(Hmaster);
[Amaster,Bmaster]=sort(Ht_master,'descend' );
Smaster_select=Xmaster(Bmaster(1:Num,:),:);

Hslave=Xslave*Xslave';
Ht_slave=diag(Hslave);
[Aslave,Bslave]=sort(Ht_slave,'descend' );
Sslave_select=Xslave(Bslave(1:Num,:),:);

%%  GLSW  
A = (Smaster_select - Sslave_select);
[U,S,V] = svd(A);
D2 = S'*S;
W = sqrt(D2./1 + eye(size(Sslave_select,2),size(Sslave_select,2)));
G=V*inv(W)*V';
T = Target*G;

%% ELM
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
         [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,T,Target_label,1, Hnumbt, ActivationFunction,Cv);
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
save radbas(1-4)

%% 
clear all
clc
%% 导入数据
load A.mat
load index.mat
SL=5; %从板
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';

%% 数据归一化
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
%中心化
for i=1: size(Source,1)
    Xmaster(i,:) = Source (i,:) - mean(Source);
end

for i=1: size(Target,1)
    Xslave(i,:) = Target (i,:) - mean(Target);
end
%% 样本子集选择
Num=30;
Hmaster=Xmaster*Xmaster';
Ht_master=diag(Hmaster);
[Amaster,Bmaster]=sort(Ht_master,'descend' );
Smaster_select=Xmaster(Bmaster(1:Num,:),:);

Hslave=Xslave*Xslave';
Ht_slave=diag(Hslave);
[Aslave,Bslave]=sort(Ht_slave,'descend' );
Sslave_select=Xslave(Bslave(1:Num,:),:);

%%  GLSW  
A = (Smaster_select - Sslave_select);
[U,S,V] = svd(A);
D2 = S'*S;
W = sqrt(D2./1 + eye(size(Sslave_select,2),size(Sslave_select,2)));
G=V*inv(W)*V';
T = Target*G;

%% ELM
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
         [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,T,Target_label,1, Hnumbt, ActivationFunction,Cv);
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
save radbas(1-5)

%% 
clear all
clc
%% 导入数据
load A.mat
load index.mat
SL=6; %从板
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';

%% 数据归一化
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
%中心化
for i=1: size(Source,1)
    Xmaster(i,:) = Source (i,:) - mean(Source);
end

for i=1: size(Target,1)
    Xslave(i,:) = Target (i,:) - mean(Target);
end
%% 样本子集选择
Num=30;
Hmaster=Xmaster*Xmaster';
Ht_master=diag(Hmaster);
[Amaster,Bmaster]=sort(Ht_master,'descend' );
Smaster_select=Xmaster(Bmaster(1:Num,:),:);

Hslave=Xslave*Xslave';
Ht_slave=diag(Hslave);
[Aslave,Bslave]=sort(Ht_slave,'descend' );
Sslave_select=Xslave(Bslave(1:Num,:),:);

%%  GLSW  
A = (Smaster_select - Sslave_select);
[U,S,V] = svd(A);
D2 = S'*S;
W = sqrt(D2./1 + eye(size(Sslave_select,2),size(Sslave_select,2)));
G=V*inv(W)*V';
T = Target*G;

%% ELM
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
         [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,T,Target_label,1, Hnumbt, ActivationFunction,Cv);
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
save radbas(1-6)

%% 
clear all
clc
%% 导入数据
load A.mat
load index.mat
SL=7; %从板
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';

%% 数据归一化
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
%中心化
for i=1: size(Source,1)
    Xmaster(i,:) = Source (i,:) - mean(Source);
end

for i=1: size(Target,1)
    Xslave(i,:) = Target (i,:) - mean(Target);
end
%% 样本子集选择
Num=30;
Hmaster=Xmaster*Xmaster';
Ht_master=diag(Hmaster);
[Amaster,Bmaster]=sort(Ht_master,'descend' );
Smaster_select=Xmaster(Bmaster(1:Num,:),:);

Hslave=Xslave*Xslave';
Ht_slave=diag(Hslave);
[Aslave,Bslave]=sort(Ht_slave,'descend' );
Sslave_select=Xslave(Bslave(1:Num,:),:);

%%  GLSW  
A = (Smaster_select - Sslave_select);
[U,S,V] = svd(A);
D2 = S'*S;
W = sqrt(D2./1 + eye(size(Sslave_select,2),size(Sslave_select,2)));
G=V*inv(W)*V';
T = Target*G;

%% ELM
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
         [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,T,Target_label,1, Hnumbt, ActivationFunction,Cv);
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
save radbas(1-7)

%% 
clear all
clc
%% 导入数据
load A.mat
load index.mat
SL=8; %从板
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';

%% 数据归一化
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
%中心化
for i=1: size(Source,1)
    Xmaster(i,:) = Source (i,:) - mean(Source);
end

for i=1: size(Target,1)
    Xslave(i,:) = Target (i,:) - mean(Target);
end
%% 样本子集选择
Num=30;
Hmaster=Xmaster*Xmaster';
Ht_master=diag(Hmaster);
[Amaster,Bmaster]=sort(Ht_master,'descend' );
Smaster_select=Xmaster(Bmaster(1:Num,:),:);

Hslave=Xslave*Xslave';
Ht_slave=diag(Hslave);
[Aslave,Bslave]=sort(Ht_slave,'descend' );
Sslave_select=Xslave(Bslave(1:Num,:),:);

%%  GLSW  
A = (Smaster_select - Sslave_select);
[U,S,V] = svd(A);
D2 = S'*S;
W = sqrt(D2./1 + eye(size(Sslave_select,2),size(Sslave_select,2)));
G=V*inv(W)*V';
T = Target*G;

%% ELM
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
         [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,T,Target_label,1, Hnumbt, ActivationFunction,Cv);
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
save radbas(1-8)

%% 
clear all
clc
%% 导入数据
load A.mat
load index.mat
SL=9; %从板
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';

%% 数据归一化
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
%中心化
for i=1: size(Source,1)
    Xmaster(i,:) = Source (i,:) - mean(Source);
end

for i=1: size(Target,1)
    Xslave(i,:) = Target (i,:) - mean(Target);
end
%% 样本子集选择
Num=30;
Hmaster=Xmaster*Xmaster';
Ht_master=diag(Hmaster);
[Amaster,Bmaster]=sort(Ht_master,'descend' );
Smaster_select=Xmaster(Bmaster(1:Num,:),:);

Hslave=Xslave*Xslave';
Ht_slave=diag(Hslave);
[Aslave,Bslave]=sort(Ht_slave,'descend' );
Sslave_select=Xslave(Bslave(1:Num,:),:);

%%  GLSW  
A = (Smaster_select - Sslave_select);
[U,S,V] = svd(A);
D2 = S'*S;
W = sqrt(D2./1 + eye(size(Sslave_select,2),size(Sslave_select,2)));
G=V*inv(W)*V';
T = Target*G;

%% ELM
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
         [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,T,Target_label,1, Hnumbt, ActivationFunction,Cv);
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
save radbas(1-9)

%% 
clear all
clc
%% 导入数据
load A.mat
load index.mat
SL=10; %从板
Source=A{1,1};
Target=A{1,SL};
Source_label=(vec2ind(index{1,1}))';
Target_label=(vec2ind(index{1,SL}))';

%% 数据归一化
Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
%中心化
for i=1: size(Source,1)
    Xmaster(i,:) = Source (i,:) - mean(Source);
end

for i=1: size(Target,1)
    Xslave(i,:) = Target (i,:) - mean(Target);
end
%% 样本子集选择
Num=30;
Hmaster=Xmaster*Xmaster';
Ht_master=diag(Hmaster);
[Amaster,Bmaster]=sort(Ht_master,'descend' );
Smaster_select=Xmaster(Bmaster(1:Num,:),:);

Hslave=Xslave*Xslave';
Ht_slave=diag(Hslave);
[Aslave,Bslave]=sort(Ht_slave,'descend' );
Sslave_select=Xslave(Bslave(1:Num,:),:);

%%  GLSW  
A = (Smaster_select - Sslave_select);
[U,S,V] = svd(A);
D2 = S'*S;
W = sqrt(D2./1 + eye(size(Sslave_select,2),size(Sslave_select,2)));
G=V*inv(W)*V';
T = Target*G;
%% ELM
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
         [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,T,Target_label,1, Hnumbt, ActivationFunction,Cv);
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
save radbas(1-10)