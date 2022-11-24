%%10块板子数据
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

[pc,score,latent,tsquare] = pca(Source);
P=pc(:,1);
Re_Target = Target-Target*P*P';

error1=0;
counter=0;
for m=-5:5
    c=10^m;
    for n=-4:4
         gama=10^n;
         counter=counter+1;
        try
            %% SVM:
                 cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
                 model = svmtrain(Source_label,Source,cmd);  
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
save CCPCAsvm(1-2)

%% 10块板子数据
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

error1=0;
counter=0;
for m=-5:5
    c=10^m;
    for n=-4:4
         gama=10^n;
         counter=counter+1;
        try
            %% SVM:
                 cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
                 model = svmtrain(Source_label,Source,cmd);  
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
save CCPCAsvm(1-3)


%% 10块板子数据
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

error1=0;
counter=0;
for m=-5:5
    c=10^m;
    for n=-4:4
         gama=10^n;
         counter=counter+1;
        try
            %% SVM:
                 cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
                 model = svmtrain(Source_label,Source,cmd);  
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
save CCPCAsvm(1-4)


%% 10块板子数据
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

error1=0;
counter=0;
for m=-5:5
    c=10^m;
    for n=-4:4
         gama=10^n;
         counter=counter+1;
        try
            %% SVM:
                 cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
                 model = svmtrain(Source_label,Source,cmd);  
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
save CCPCAsvm(1-5)


%% 10块板子数据
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

error1=0;
counter=0;
for m=-5:5
    c=10^m;
    for n=-4:4
         gama=10^n;
         counter=counter+1;
        try
            %% SVM:
                 cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
                 model = svmtrain(Source_label,Source,cmd);  
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
save CCPCAsvm(1-6)


%% 10块板子数据
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

error1=0;
counter=0;
for m=-5:5
    c=10^m;
    for n=-4:4
         gama=10^n;
         counter=counter+1;
        try
            %% SVM:
                 cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
                 model = svmtrain(Source_label,Source,cmd);  
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
save CCPCAsvm(1-7)


%% 10块板子数据
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

error1=0;
counter=0;
for m=-5:5
    c=10^m;
    for n=-4:4
         gama=10^n;
         counter=counter+1;
        try
            %% SVM:
                 cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
                 model = svmtrain(Source_label,Source,cmd);  
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
save CCPCAsvm(1-8)

%% 10块板子数据
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

error1=0;
counter=0;
for m=-5:5
    c=10^m;
    for n=-4:4
         gama=10^n;
         counter=counter+1;
        try
            %% SVM:
                 cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
                 model = svmtrain(Source_label,Source,cmd);  
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
save CCPCAsvm(1-9)

%% 10块板子数据
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

error1=0;
counter=0;
for m=-5:5
    c=10^m;
    for n=-4:4
         gama=10^n;
         counter=counter+1;
        try
            %% SVM:
                 cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
                 model = svmtrain(Source_label,Source,cmd);  
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
save CCPCAsvm(1-10)


% %%10块板子数据
% clear all
% clc
% load A.mat
% load index.mat
% SL=3; %从板
% Source=[A{1,1};A{1,2}];
% Target=A{1,SL};
% Source_label=[(vec2ind(index{1,1}))';(vec2ind(index{1,2}))'];
% Target_label=(vec2ind(index{1,SL}))';
% %归一化
% Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
% Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
% 
% [pc,score,latent,tsquare] = pca(Target);
% P=pc(:,1);
% Re_Target = Target-Target*P*P';
% 
% error1=0;
% counter=0;
% for m=-5:5
%     c=10^m;
%     for n=-4:4
%          gama=10^n;
%          counter=counter+1;
%         try
%             %% SVM:
%                  cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
%                  model = svmtrain(Source_label,Source,cmd);  
%                  [predict_label_test,] = svmpredict(Target_label,Re_Target, model);
%                  d=diff([predict_label_test';Target_label']);
%                  N = numel(find(d==0));
%                  accur_test=N/size(Re_Target,1);
%         catch
%               error1 =error1 + 1;
%               accur_test = 0;
%         end
%          result(counter,1)=m;
%          result(counter,2)=n;
%          result(counter,3)=accur_test;                            
%     end                 
% end
% [max,index]=max(result(:,3));
% save CCPCAsvm(1-2to3)
% 
% %% 10块板子数据
% clear all
% clc
% load A.mat
% load index.mat
% SL=4; %从板
% Source=[A{1,1};A{1,2};A{1,3}];
% Target=A{1,SL};
% Source_label=[(vec2ind(index{1,1}))';(vec2ind(index{1,2}))';(vec2ind(index{1,3}))'];
% Target_label=(vec2ind(index{1,SL}))';
% %归一化
% Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
% Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
% 
% [pc,score,latent,tsquare] = pca(Target);
% P=pc(:,1);
% Re_Target = Target-Target*P*P';
% 
% error1=0;
% counter=0;
% for m=-5:5
%     c=10^m;
%     for n=-4:4
%          gama=10^n;
%          counter=counter+1;
%         try
%             %% SVM:
%                  cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
%                  model = svmtrain(Source_label,Source,cmd);  
%                  [predict_label_test,] = svmpredict(Target_label,Re_Target, model);
%                  d=diff([predict_label_test';Target_label']);
%                  N = numel(find(d==0));
%                  accur_test=N/size(Re_Target,1);
%         catch
%               error1 =error1 + 1;
%               accur_test = 0;
%         end
%          result(counter,1)=m;
%          result(counter,2)=n;
%          result(counter,3)=accur_test;                            
%     end                 
% end
% [max,index]=max(result(:,3));
% save CCPCAsvm(1-3to4)
% 
% %% 10块板子数据
% clear all
% clc
% load A.mat
% load index.mat
% SL=5; %从板
% Source=[A{1,1};A{1,2};A{1,3};A{1,4}];
% Target=A{1,SL};
% Source_label=[(vec2ind(index{1,1}))';(vec2ind(index{1,2}))';(vec2ind(index{1,3}))';(vec2ind(index{1,4}))'];
% Target_label=(vec2ind(index{1,SL}))';
% %归一化
% Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
% Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
% 
% [pc,score,latent,tsquare] = pca(Target);
% P=pc(:,1);
% Re_Target = Target-Target*P*P';
% 
% error1=0;
% counter=0;
% for m=-5:5
%     c=10^m;
%     for n=-4:4
%          gama=10^n;
%          counter=counter+1;
%         try
%             %% SVM:
%                  cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
%                  model = svmtrain(Source_label,Source,cmd);  
%                  [predict_label_test,] = svmpredict(Target_label,Re_Target, model);
%                  d=diff([predict_label_test';Target_label']);
%                  N = numel(find(d==0));
%                  accur_test=N/size(Re_Target,1);
%         catch
%               error1 =error1 + 1;
%               accur_test = 0;
%         end
%          result(counter,1)=m;
%          result(counter,2)=n;
%          result(counter,3)=accur_test;                            
%     end                 
% end
% [max,index]=max(result(:,3));
% save CCPCAsvm(1-4to5)
% 
% %% 10块板子数据
% clear all
% clc
% load A.mat
% load index.mat
% SL=6; %从板
% Source=[A{1,1};A{1,2};A{1,3};A{1,4};
%         A{1,5}];
% Target=A{1,SL};
% Source_label=[(vec2ind(index{1,1}))';(vec2ind(index{1,2}))';(vec2ind(index{1,3}))';(vec2ind(index{1,4}))';
%               (vec2ind(index{1,5}))'];
% Target_label=(vec2ind(index{1,SL}))';
% %归一化
% Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
% Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
% 
% [pc,score,latent,tsquare] = pca(Target);
% P=pc(:,1);
% Re_Target = Target-Target*P*P';
% 
% error1=0;
% counter=0;
% for m=-5:5
%     c=10^m;
%     for n=-4:4
%          gama=10^n;
%          counter=counter+1;
%         try
%             %% SVM:
%                  cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
%                  model = svmtrain(Source_label,Source,cmd);  
%                  [predict_label_test,] = svmpredict(Target_label,Re_Target, model);
%                  d=diff([predict_label_test';Target_label']);
%                  N = numel(find(d==0));
%                  accur_test=N/size(Re_Target,1);
%         catch
%               error1 =error1 + 1;
%               accur_test = 0;
%         end
%          result(counter,1)=m;
%          result(counter,2)=n;
%          result(counter,3)=accur_test;                            
%     end                 
% end
% [max,index]=max(result(:,3));
% save CCPCAsvm(1-5to6)
% 
% %% 10块板子数据
% clear all
% clc
% load A.mat
% load index.mat
% SL=7; %从板
% Source=[A{1,1};A{1,2};A{1,3};A{1,4};
%         A{1,5};A{1,6}];
% Target=A{1,SL};
% Source_label=[(vec2ind(index{1,1}))';(vec2ind(index{1,2}))';(vec2ind(index{1,3}))';(vec2ind(index{1,4}))';
%               (vec2ind(index{1,5}))';(vec2ind(index{1,6}))'];
% Target_label=(vec2ind(index{1,SL}))';
% %归一化
% Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
% Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
% 
% [pc,score,latent,tsquare] = pca(Target);
% P=pc(:,1);
% Re_Target = Target-Target*P*P';
% 
% error1=0;
% counter=0;
% for m=-5:5
%     c=10^m;
%     for n=-4:4
%          gama=10^n;
%          counter=counter+1;
%         try
%             %% SVM:
%                  cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
%                  model = svmtrain(Source_label,Source,cmd);  
%                  [predict_label_test,] = svmpredict(Target_label,Re_Target, model);
%                  d=diff([predict_label_test';Target_label']);
%                  N = numel(find(d==0));
%                  accur_test=N/size(Re_Target,1);
%         catch
%               error1 =error1 + 1;
%               accur_test = 0;
%         end
%          result(counter,1)=m;
%          result(counter,2)=n;
%          result(counter,3)=accur_test;                            
%     end                 
% end
% [max,index]=max(result(:,3));
% save CCPCAsvm(1-6to7)
% 
% %% 10块板子数据
% clear all
% clc
% load A.mat
% load index.mat
% SL=8; %从板
% Source=[A{1,1};A{1,2};A{1,3};A{1,4};
%         A{1,5};A{1,6};A{1,7}];
% Target=A{1,SL};
% Source_label=[(vec2ind(index{1,1}))';(vec2ind(index{1,2}))';(vec2ind(index{1,3}))';(vec2ind(index{1,4}))';
%               (vec2ind(index{1,5}))';(vec2ind(index{1,6}))';(vec2ind(index{1,7}))'];
% Target_label=(vec2ind(index{1,SL}))';
% %归一化
% Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
% Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
% 
% [pc,score,latent,tsquare] = pca(Target);
% P=pc(:,1);
% Re_Target = Target-Target*P*P';
% 
% error1=0;
% counter=0;
% for m=-5:5
%     c=10^m;
%     for n=-4:4
%          gama=10^n;
%          counter=counter+1;
%         try
%             %% SVM:
%                  cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
%                  model = svmtrain(Source_label,Source,cmd);  
%                  [predict_label_test,] = svmpredict(Target_label,Re_Target, model);
%                  d=diff([predict_label_test';Target_label']);
%                  N = numel(find(d==0));
%                  accur_test=N/size(Re_Target,1);
%         catch
%               error1 =error1 + 1;
%               accur_test = 0;
%         end
%          result(counter,1)=m;
%          result(counter,2)=n;
%          result(counter,3)=accur_test;                            
%     end                 
% end
% [max,index]=max(result(:,3));
% save CCPCAsvm(1-7to8)
% 
% %% 10块板子数据
% clear all
% clc
% load A.mat
% load index.mat
% SL=9; %从板
% Source=[A{1,1};A{1,2};A{1,3};A{1,4};
%         A{1,5};A{1,6};A{1,7};A{1,8}];
% Target=A{1,SL};
% Source_label=[(vec2ind(index{1,1}))';(vec2ind(index{1,2}))';(vec2ind(index{1,3}))';(vec2ind(index{1,4}))';
%               (vec2ind(index{1,5}))';(vec2ind(index{1,6}))';(vec2ind(index{1,7}))';(vec2ind(index{1,8}))'];
% Target_label=(vec2ind(index{1,SL}))';
% %归一化
% Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
% Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
% 
% [pc,score,latent,tsquare] = pca(Target);
% P=pc(:,1);
% Re_Target = Target-Target*P*P';
% 
% error1=0;
% counter=0;
% for m=-5:5
%     c=10^m;
%     for n=-4:4
%          gama=10^n;
%          counter=counter+1;
%         try
%             %% SVM:
%                  cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
%                  model = svmtrain(Source_label,Source,cmd);  
%                  [predict_label_test,] = svmpredict(Target_label,Re_Target, model);
%                  d=diff([predict_label_test';Target_label']);
%                  N = numel(find(d==0));
%                  accur_test=N/size(Re_Target,1);
%         ca
%         tch
%               error1 =error1 + 1;
%               accur_test = 0;
%         end
%          result(counter,1)=m;
%          result(counter,2)=n;
%          result(counter,3)=accur_test;                            
%     end                 
% end
% [max,index]=max(result(:,3));
% save CCPCAsvm(1-8to9)
% 
% %% 10块板子数据
% clear all
% clc
% load A.mat
% load index.mat
% SL=10; %从板
% Source=[A{1,1};A{1,2};A{1,3};A{1,4};
%         A{1,5};A{1,6};A{1,7};A{1,8};A{1,9}];
% Target=A{1,SL};
% Source_label=[(vec2ind(index{1,1}))';(vec2ind(index{1,2}))';(vec2ind(index{1,3}))';(vec2ind(index{1,4}))';
%               (vec2ind(index{1,5}))';(vec2ind(index{1,6}))';(vec2ind(index{1,7}))';(vec2ind(index{1,8}))';
%               (vec2ind(index{1,9}))'];
% Target_label=(vec2ind(index{1,SL}))';
% %归一化
% Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
% Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
% 
% [pc,score,latent,tsquare] = pca(Target);
% P=pc(:,1);
% Re_Target = Target-Target*P*P';
% 
% error1=0;
% counter=0;
% for m=-5:5
%     c=10^m;
%     for n=-4:4
%          gama=10^n;
%          counter=counter+1;
%         try
%             %% SVM:
%                  cmd=[' -c ',num2str(c),' -g ',num2str(gama)];  %svmtrain参数
%                  model = svmtrain(Source_label,Source,cmd);  
%                  [predict_label_test,] = svmpredict(Target_label,Re_Target, model);
%                  d=diff([predict_label_test';Target_label']);
%                  N = numel(find(d==0));
%                  accur_test=N/size(Re_Target,1);
%         catch
%               error1 =error1 + 1;
%               accur_test = 0;
%         end
%          result(counter,1)=m;
%          result(counter,2)=n;
%          result(counter,3)=accur_test;                            
%     end                 
% end
% [max,index]=max(result(:,3));
% save CCPCAsvm(1-9to11)