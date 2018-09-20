####This is a first script for DAS####

#Load packages for model development
install.packages('caret')
library(caret)



#lines for reading files

#excel
file1 <- read_excel("filepath!")
View(file)

#csv
file2 <- read.csv('filepath', header = True/False, sep = ';/,')

#samenvoegen twee tabellen


merged_table <- merge(file1, file2, by = 'shared column', all.x=TRUE (#all cases of dataframe x are preserved))

                      
#doorsnijdingen of aggregaties

agg_data <- aggregate(merged_table, by=list(merged_table$column_name), 
          FUN = function(x)c(Mean = mean(x), SD = sd(X)), na.rm=TRUE)

#make a bar chart

summary.matrix <- data.frame(Fieldname=levels(as.factor(agg_data$aggregation_based_fieldname)),
  AggregationFunctionName=tapply(merged_data$field, merged_table$field, mean))                    
                      
ggplot(summary.matrix, aes(x = factor(Fieldname), y = AggregationFunctionName)) + geom_bar(stat = "identity")

