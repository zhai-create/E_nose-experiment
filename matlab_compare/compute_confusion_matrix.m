% 预测标签，每一类的数目，类别数目
function [confusion_matrix]=compute_confusion_matrix(predict_label,num_in_class,name_class)
% predict_label为一维行向量
% num_in_class代表每一类的个数
% name_class代表类名
num_class=length(num_in_class);
num_in_class=[0 num_in_class];

confusion_matrix=size(num_class,num_class);
 
for ci=1:num_class
    for cj=1:num_class
        summer=0;%统计对应标签个数  
        c_start=sum(num_in_class(1:ci))+1;
        c_end=sum(num_in_class(1:ci+1));
        summer=size(find(predict_label(c_start:c_end)==cj),2);  % 统计对应标签个数,注意此处的predict_label可能是数值组成的向量，不是字符串组成的向量
%         confusion_matrix(ci,cj)=summer/num_in_class(ci+1);%这个是显示正确率，
        confusion_matrix(ci,cj)=summer;%这个是显示图片有多少张，
    end
end
 
draw_cm(confusion_matrix,name_class,num_class);
 
end
