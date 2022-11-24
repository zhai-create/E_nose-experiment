% auto run

run('DIY_SVM2.m');
acc = [acc1_mean;acc2_mean];
save ACC_DIY_SVM2.mat acc

run('DIY_PCA_SVM2.m');
acc = [acc1_mean;acc2_mean];
save ACC_DIY_PCA_SVM2.mat acc

run('DIY_CCPCA_SVM2.m');
acc = [acc1_mean;acc2_mean];
save ACC_DIY_CCPCA_SVM2.mat acc

run('DIY_OSC_SVM2.m');
acc = [acc1_mean;acc2_mean];
save ACC_DIY_OSC_SVM2.mat acc

run('DIY_LDA_SVM2.m');
acc = [acc1_mean;acc2_mean];
save ACC_DIY_LDA_SVM2.mat acc

run('DIY_DS_SVM2.m');
acc = [acc1_mean;acc2_mean];
save ACC_DIY_DS_SVM2.mat acc

run('DIY_LPP_SVM2.m');
acc = [acc1_mean;acc2_mean];
save ACC_DIY_LPP_SVM2.mat acc

run('DIY_GLSW_SVM2.m');
acc = [acc1_mean;acc2_mean];
save ACC_DIY_GLSW_SVM2.mat acc
