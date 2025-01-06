%{
load("trainds.mat")
load("valds.mat")
disp("Making BOWs")
[BOW_tr,labels_tr,BOW_val,labels_val] = BOWCreator(trainds,valds,20,200,["SURF"],150);
disp("Training")
tb = TreeBagger(200,BOW_tr, labels_tr);
svm = fitcecoc(BOW_tr,labels_tr);
disp("Predicting")
predicted = cellfun(@str2double,tb.predict(BOW_val));
%}
CM = confusionmat(labels_val,categorical(predicted));
CM = CM./repmat(sum(CM,2),1,size(CM,2));
accuracy = mean(diag(CM));
figure(1)
confusionchart(CM)
title(["Accuracy:", num2str(accuracy), "20,200,[SURF],150, TreeBagger200"])

predicted = svm.predict(BOW_val);
CM = confusionmat(labels_val,predicted);
CM = CM./repmat(sum(CM,2),1,size(CM,2));
accuracy = mean(diag(CM));

figure(2)
confusionchart(CM)
title(["Accuracy:", num2str(accuracy), "20,200,[SURF],150, SVM"])