svm = fitcecoc(BOW_tr,labels_tr);
predicted = svm.predict(BOW_val);
CM = confusionmat(labels_val,predicted);
CM = CM./repmat(sum(CM,2),1,size(CM,2));
accuracy = mean(diag(CM));