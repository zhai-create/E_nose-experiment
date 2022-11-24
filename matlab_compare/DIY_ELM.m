% DIY_ELM
%% setting 1
acc = zeros(1,9);
for SL = 2:10
    clear -regexp [^SL,^acc]
    clc
    load A.mat
    load index.mat
    Source=A{1,1};
    Target=A{1,SL};
    Source_label=(vec2ind(index{1,1}))';
    Target_label=(vec2ind(index{1,SL}))';
    %归一化
    Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
    Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
    
    for cishu=1:10
        clear max;
        clear result;
    % ELM
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
                                [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,Target,Target_label,1, Hnumbt, ActivationFunction,Cv);
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
    acc(SL-1) = max;
end

%% setting 2
acc = zeros(1,9);
for SL = 2:10
    clear -regexp [^SL,^acc]
    clc
    load A.mat
    load index.mat
    Source = A{1,SL-1};
    Target = A{1,SL};
    Source_label=(vec2ind(index{1,SL-1}))';
    Target_label=(vec2ind(index{1,SL}))';
    %归一化
    Source=Source./repmat(sqrt(sum(Source.^2,1)),size(Source,1),1);
    Target=Target./repmat(sqrt(sum(Target.^2,1)),size(Target,1),1);
    
    for cishu=1:10
        clear max;
        clear result;
    % ELM
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
                                [TrainingAccuracy, TestingAccuracy] = fast_elm(Source,Source_label,Target,Target_label,1, Hnumbt, ActivationFunction,Cv);
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
    acc(SL-1) = max;
end

