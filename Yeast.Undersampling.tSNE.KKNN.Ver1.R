setwd("C:/Users/faisal/OneDrive - eSevens/HD Backup/My University/Study/Kanazawa University/Riset/Code/R/201703")
rm(list = ls())

library(Rtsne)
library(kknn)
library(ROCR)
library(caret)
library(unbalanced)

if(exists("result_classification_all_auc")){
  rm(result_classification_all_auc)
}

if(exists("result_classification_all_sens")){
  rm(result_classification_all_sens)
}

if(exists("result_classification_all_spec")){
  rm(result_classification_all_spec)
}

dim_value = 2
k_value = 5
cross_num = 5
perplexity_value = 50
positive_label = "ME3"
negative_label = "BIG"

positive_label_us = "1"
negative_label_us = "0"

filename_save = "yeast"
filename_data = "data/yeast2class.csv"

main_data = read.csv(filename_data, stringsAsFactors = FALSE)
colnames(main_data)[ncol(main_data)] = "class_label"
main_data[which(main_data$class_label == positive_label), ncol(main_data)] = 1
main_data[which(main_data$class_label == negative_label), ncol(main_data)] = 0
main_data[,ncol(main_data)] = as.factor(main_data[,ncol(main_data)])

main_data.positive = main_data[which(main_data$class_label == positive_label_us),]
main_data.negative = main_data[which(main_data$class_label == negative_label_us),]

cross_positive = round(nrow(main_data.positive)/cross_num)
cross_negative = round(nrow(main_data.negative)/cross_num)


#cross-validation ============================================================== - start
for(cross_i in 1:cross_num){
  start_positive_i = cross_i
  start_negative_i = cross_i
  
  if(cross_i == 1){
    end_positive_i = cross_i*cross_positive
    end_negative_i = cross_i*cross_negative
  }
  
  if(cross_i > 1){
    start_positive_i = end_positive_i+1
    start_negative_i = end_negative_i+1
    
    end_positive_i = (cross_i)*cross_positive
    end_negative_i = (cross_i)*cross_negative
  }
  
  if(cross_i == cross_num){
    end_positive_i = nrow(main_data.positive)
    end_negative_i = nrow(main_data.negative)
  }
  
  main_data.test = rbind.data.frame(main_data.positive[c(start_positive_i:end_positive_i),], main_data.negative[c(start_negative_i:end_negative_i),])
  main_data.train = rbind.data.frame(main_data.positive[-c(start_positive_i:end_positive_i),], main_data.negative[-c(start_negative_i:end_negative_i),])
  
  if(exists("result_classification_auc")){
    rm(result_classification_auc)
  }
  
  if(exists("result_classification_sens")){
    rm(result_classification_sens)
  }
  
  if(exists("result_classification_spec")){
    rm(result_classification_spec)
  }
  
  #exp3 u.s -> tsne
  #==============================================================
  exp3_data.train = main_data.train
  exp3_data.test = main_data.test
  exp3_data.train.positive = exp3_data.train[which(exp3_data.train$class_label == positive_label_us),] 
  exp3_data.train.negative = exp3_data.train[which(exp3_data.train$class_label == negative_label_us),] 
  exp3_ir = round(nrow(exp3_data.train.negative)/nrow(exp3_data.train.positive))
  exp3_perc = (nrow(exp3_data.train.positive)/(exp3_ir*nrow(exp3_data.train.positive)))*100
  init_ir = exp3_ir -1
  
  exp3_cross_i = 1
  while(exp3_perc <= 50){
    #random U.S
    output = exp3_data.train$class_label
    input = exp3_data.train[, -(ncol(exp3_data.train))]
    exp3_data.train.us = ubUnder(X=input, Y=output, perc = exp3_perc, method = "percPos")
    exp3_data.train = cbind.data.frame(exp3_data.train.us$X, exp3_data.train.us$Y)
    colnames(exp3_data.train)[ncol(exp3_data.train)] = "class_label"
    main_data.train = exp3_data.train
    #print(table(exp3_data.train.us$Y))
    
    #tsne
    main_data.process = rbind.data.frame(main_data.train, main_data.test)
    main_data.train.count = nrow(main_data.train)
    main_data.matrix = as.matrix(main_data.process[,-(ncol(main_data.process))])
    main_data.tsne = Rtsne(main_data.matrix,dims=dim_value,PCA=T,max_iter=5000, perplexity=perplexity_value, check_duplicates=FALSE)
    main_data.tsne.result = cbind.data.frame(main_data.tsne$Y, main_data.process[,ncol(main_data.process)])
    colnames(main_data.tsne.result)[ncol(main_data.tsne.result)] = "class_label"
    
    main_data.process.train = main_data.tsne.result[1:main_data.train.count,]
    main_data.process.test = main_data.tsne.result[(main_data.train.count+1):nrow(main_data.process),]
    
    #classification
    classification_model = kknn(class_label~., main_data.process.train, main_data.process.test[,-ncol(main_data.process.test)], k = k_value, distance = 1, kernel = "triangular")
    roc.prediction = ROCR::prediction(as.numeric(fitted(classification_model)), as.numeric(main_data.process.test[,ncol(main_data.process.test)]))
    roc.tpr.fpr = ROCR::performance(roc.prediction,"tpr","fpr")
    roc.auc = ROCR::performance(roc.prediction,"auc")
    #exp2_auc[cross_i] = as.numeric(roc.auc@y.values)
    
    if(!exists("result_classification_auc")){
      assign("result_classification_auc", as.numeric(roc.auc@y.values))
    } else {
      result_classification_auc = rbind.data.frame(result_classification_auc, as.numeric(roc.auc@y.values))
    }
    
    matrix.sens.spec = confusionMatrix(fitted(classification_model), main_data.process.test[,ncol(main_data.process.test)], positive = positive_label_us)
    #exp2_sens[cross_i] = as.numeric(matrix.sens.spec$byClass[1])
    #exp2_spec[cross_i] = as.numeric(matrix.sens.spec$byClass[2])
    
    if(!exists("result_classification_sens")){
      assign("result_classification_sens", as.numeric(matrix.sens.spec$byClass[1]))
    } else {
      result_classification_sens = rbind.data.frame(result_classification_sens, as.numeric(matrix.sens.spec$byClass[1]))
    }
    
    if(!exists("result_classification_spec")){
      assign("result_classification_spec", as.numeric(matrix.sens.spec$byClass[2]))
    } else {
      result_classification_spec = rbind.data.frame(result_classification_spec, as.numeric(matrix.sens.spec$byClass[2]))
    }
    
    #update IR
    exp3_ir = exp3_ir - 1
    exp3_perc = (nrow(exp3_data.train.positive)/(exp3_ir*nrow(exp3_data.train.positive)))*100
  }
  
  colnames(result_classification_auc)[1] = paste0("V", cross_i)
  if(!exists("result_classification_all_auc")){
    assign("result_classification_all_auc", result_classification_auc)
  } else {
    result_classification_all_auc = cbind.data.frame(result_classification_all_auc, result_classification_auc)
  }
  
  colnames(result_classification_sens)[1] = paste0("V", cross_i)
  if(!exists("result_classification_all_sens")){
    assign("result_classification_all_sens", result_classification_sens)
  } else {
    result_classification_all_sens = cbind.data.frame(result_classification_all_sens, result_classification_sens)
  }
  
  colnames(result_classification_spec)[1] = paste0("V", cross_i)
  if(!exists("result_classification_all_spec")){
    assign("result_classification_all_spec", result_classification_spec)
  } else {
    result_classification_all_spec = cbind.data.frame(result_classification_all_spec, result_classification_spec)
  }
  
}
#cross-validation ============================================================== - end


#show result
#==============================================================
print("AUC--------------------------")
result_classification_all_auc = cbind.data.frame(result_classification_all_auc, cbind(rowMeans(result_classification_all_auc)))
colnames(result_classification_all_auc)[ncol(result_classification_all_auc)] = "average"
result_classification_all_auc=cbind.data.frame(cbind(c(init_ir:exp3_ir)), result_classification_all_auc)
colnames(result_classification_all_auc)[1] = "IR"
print(result_classification_all_auc)
write.csv(result_classification_all_auc, paste0("result/",filename_save,"_auc.csv"), row.names = FALSE, quote = FALSE)

print("Sensitivity--------------------------")
result_classification_all_sens = cbind.data.frame(result_classification_all_sens, cbind(rowMeans(result_classification_all_sens)))
colnames(result_classification_all_sens)[ncol(result_classification_all_sens)] = "average"
result_classification_all_sens=cbind.data.frame(cbind(c(init_ir:exp3_ir)), result_classification_all_sens)
colnames(result_classification_all_sens)[1] = "IR"
print(result_classification_all_sens)
write.csv(result_classification_all_sens, paste0("result/",filename_save,"_sens.csv"), row.names = FALSE, quote = FALSE)

print("Specificity--------------------------")
result_classification_all_spec = cbind.data.frame(result_classification_all_spec, cbind(rowMeans(result_classification_all_spec)))
colnames(result_classification_all_spec)[ncol(result_classification_all_spec)] = "average"
result_classification_all_spec=cbind.data.frame(cbind(c(init_ir:exp3_ir)), result_classification_all_spec)
colnames(result_classification_all_spec)[1] = "IR"
print(result_classification_all_spec)
write.csv(result_classification_all_spec, paste0("result/",filename_save,"_spec.csv"), row.names = FALSE, quote = FALSE)