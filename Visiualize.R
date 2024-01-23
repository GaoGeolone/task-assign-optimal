library(ggforce)
#install.packages("ggforce")
library(dplyr)
library(tidyr)
library(hrbrthemes)
library(viridis)
library(patchwork)
library(gghighlight)
#install.packages("HyperG")
library(HyperG)
#install.packages("gghighlight")
library("colorspace")
# Read data.csv
getwd()
setwd('g:/15-GEngine/advanced-architecture-scheduler/OptimizerAnalyze/Log2x2')

# Section Exp and discussion subsec1
df<-read.csv('data.csv')
# 按照 category 列进行分组
#test <- df[df$Type != "FunAssigned" | df$SolutionQuality != "non-optimal", ]
da_filter <- df[df$Xp != "[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]" ,]
#da_filter <- da_filter[da_filter$Type != "Random" , ]
data_w0 <- da_filter %>%
  group_by(omegaf0,Type) %>%
  summarise(
    median = median(obj),  # 中位数
    q1 = quantile(obj, 0.25),  # 下四分位数
    q3 = quantile(obj, 0.75),  # 上四分位数
    upper = min(q3, max(obj)),  # 上边缘
    lower = max(q1, min(obj))  # 下边缘
  )
data_w1 <- da_filter %>%
  group_by(omegag0,Type) %>%
  summarise(
    median = median(obj),  # 中位数
    q1 = quantile(obj, 0.25),  # 下四分位数
    q3 = quantile(obj, 0.75),  # 上四分位数
    upper = min(q3, max(obj)),  # 上边缘
    lower = max(q1, min(obj))  # 下边缘
  )
data_w2 <- da_filter %>%
  group_by(omegal0,Type) %>%
  summarise(
    median = median(obj),  # 中位数
    q1 = quantile(obj, 0.25),  # 下四分位数
    q3 = quantile(obj, 0.75),  # 上四分位数
    upper = min(q3, max(obj)),  # 上边缘
    lower = max(q1, min(obj))  # 下边缘
  )
# Represent it
P1 <- data_w0 %>%
  ggplot( aes(x=omegaf0, y=median, group=Type, color=Type)) +
  geom_line() +
  #scale_color_viridis(discrete = TRUE) +
  #ggtitle("Performance Comparison of Different Scheduling Mechanisms") +
  #theme_ipsum() +
  scale_y_continuous(limits = c(0, 300)) +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill=Type), alpha = 0.255)+
  ylab("Minmax Workload")+
  xlab(expression(paste("Frequency of function ",f[0]," as ", omega[0])))+
  scale_fill_discrete_qualitative(palette = "Dark 3")+
  theme(
    plot.title = element_text(hjust = 0.5),  # 设置标题居中
    axis.title.y = element_text(hjust = 0.5), # 设置y轴标签居中
    axis.title.x = element_text(hjust = 0.5)  # 如果需要也可以设置x轴标签居中
  )+ theme(legend.position = "none")
   +gghighlight::gghighlight(Type = FunAssigned)
P2 <- data_w1 %>%
  ggplot( aes(x=omegag0, y=median, group=Type, color=Type)) +
  geom_line() +
  #scale_color_viridis(discrete = TRUE) +
  #ggtitle("Performance Comparison of Different Scheduling Mechanisms") +
  #theme_ipsum() +
  scale_y_continuous(limits = c(0, 300)) +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill=Type), alpha = 0.255)+
  ylab("Minmax Workload")+
  xlab(expression(paste("Frequency of function ",g[0]," as ", omega[1])))+
  scale_fill_discrete_qualitative(palette = "Dark 3")+
  theme(
    plot.title = element_text(hjust = 0.5),  # 设置标题居中
    axis.title.y = element_text(hjust = 0.5), # 设置y轴标签居中
    axis.title.x = element_text(hjust = 0.5)  # 如果需要也可以设置x轴标签居中
  )+ theme(legend.position = "none")
P3 <- data_w2 %>%
  ggplot( aes(x=omegal0, y=median, group=Type, color=Type)) +
  geom_line() +
  #scale_color_viridis(discrete = TRUE) +
  #ggtitle("Performance Comparison of Different Scheduling Mechanisms") +
  #theme_ipsum() +
  scale_y_continuous(limits = c(0, 300)) +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill=Type), alpha = 0.255)+
  ylab("Minmax Workload")+
  xlab(expression(paste("Frequency of function ",l[0]," as ", omega[2])))+
  scale_fill_discrete_qualitative(palette = "Dark 3")+
  theme(
    plot.title = element_text(hjust = 0.5),  # 设置标题居中
    axis.title.y = element_text(hjust = 0.5), # 设置y轴标签居中
    axis.title.x = element_text(hjust = 0.5)  # 如果需要也可以设置x轴标签居中
  )
combined_plot <- P1 + P2 + P3 +
  plot_layout(ncol = 3)

combined_plot

# Exp and discussion subsec1
# 显著性检验定义自定义函数
da_test <- da_filter %>%
  pivot_wider(
    id_cols = c(omegaf0, omegag0, omegaf1, omegag1, omegal0),
    names_from = Type,
    values_from = obj)
print(da_test)
error<-da_test[da_test$FunAssigned>da_test$EntityAssigned,]
print(na.omit(error))
# 单样本 t 检验test_2_1 <- na.omit(da_test$EntityAssigned - da_test$FunAssigned)
test_2_1 <- na.omit(da_test$EntityAssigned - da_test$FunAssigned)
test_3_1 <- na.omit(da_test$FOnlyCU - da_test$FunAssigned)
test_4_1 <- na.omit(da_test$EOnlyCu - da_test$FunAssigned)
test_5_1 <- na.omit(da_test$FOnlyLK - da_test$FunAssigned)
test_6_1 <- na.omit(da_test$EOnlyLK - da_test$FunAssigned)
test_7_1 <- na.omit(da_test$Random - da_test$FunAssigned)
result1 <- t.test(test_2_1, mu = 0, alternative = "greater")
result2 <- t.test(test_3_1, mu = 0, alternative = "greater")
result3 <- t.test(test_4_1, mu = 0, alternative = "greater")
result4 <- t.test(test_5_1, mu = 0, alternative = "greater")
result5 <- t.test(test_6_1, mu = 0, alternative = "greater")
result6 <- t.test(test_7_1, mu = 0, alternative = "greater")
print(result1)
print(result2)
print(result3)
print(result4)
print(result5)
print(result6)
typeof(test_2_1)
ggplot(data=test_2_1)
min(test_4_1)

# Discussion subsec3
# 找到Funassigned 显著发挥作用的解，对比Xp和Xp*的区别，看看耦合性是如何影响最优的分配结果的
# test_2_1 取出其中的四分位点
quartiles <- unname(quantile(test_2_1, probs = c(0.25, 0.5, 0.75)))
# 将四分位数赋值给q1、q2、q3
q1 <- quartiles[1]
q2 <- quartiles[2]
q3 <- quartiles[3]
Group1 <- da_test[da_test$EntityAssigned - da_test$FunAssigned == q1,]
Group2 <- da_test[da_test$EntityAssigned - da_test$FunAssigned == q2,]
Group3 <- da_test[da_test$EntityAssigned - da_test$FunAssigned > q3,]
#group1 <- Group1[,c('omegaf0','omegag0','omegal0')]
#Group2[,c('omegaf0','omegag0','omegal0')]
#Group3[,c('omegaf0','omegag0','omegal0')]
group1[group1$omegag0==group1$omegal0,]
# 创建条件向量
condition_o <- (da_test$omegaf0 == 20) & (da_test$omegag0 == 20) & (da_test$omegal0 == 20)
condition_f <- (da_test$omegaf0 == 45) & (da_test$omegag0 == 20) & (da_test$omegal0 == 20)
condition_g <- (da_test$omegaf0 == 20) & (da_test$omegag0 == 45) & (da_test$omegal0 == 20)
condition_l <- (da_test$omegaf0 == 20) & (da_test$omegag0 == 20) & (da_test$omegal0 == 45)
# 使用条件向量过滤数据框
selected_rows <- da_test[condition_o, ]
selected_rowsf <- da_test[condition_f, ]
selected_rowsg <- da_test[condition_g, ]
selected_rowsl <- da_test[condition_l, ]
selected_rows
selected_rowsf
selected_rowsg
selected_rowsl
dcondition_o <- (df$omegaf0 == 10) & (df$omegag0 == 10) & (df$omegal0 == 10)
dcondition_f <- (df$omegaf0 == 50) & (df$omegag0 == 10) & (df$omegal0 == 10)
dcondition_g <- (df$omegaf0 == 10) & (df$omegag0 == 50) & (df$omegal0 == 10)
dcondition_l <- (df$omegaf0 == 10) & (df$omegag0 == 10) & (df$omegal0 == 50)
dselected_rows <- df[dcondition_o,]
dselected_rowsf <- df[dcondition_f, ]
dselected_rowsg <- df[dcondition_g, ]
dselected_rowsl <- df[dcondition_l, ]

DrawHyperGraphOfFunctions <- function(dselected){
  # 转换为矩阵
  stringOfmat <- dselected[dselected$Type=='FunAssigned',]$Xp
  print(dselected[dselected$Type=='FunAssigned',]$obj)
  # 删除多余的空格和换行符
  stringOfmat <- gsub("\\[|\\]", "", stringOfmat)
  stringOfmat <- gsub("\\s+", " ", stringOfmat)
  stringOfmat <- gsub("\\n", "", stringOfmat)
  # 将字符串转换为向量
  vec <- unlist(strsplit(stringOfmat, " "))
  vec <- as.numeric(vec)
  # 将向量转换为矩阵
  mat <- matrix(vec, nrow=8, byrow=TRUE)
  # 显示转换后的矩阵
  print(mat)
  # 获取矩阵的行数和列数
  nrow <- nrow(mat)
  ncol <- ncol(mat)
  funName <- c('e0-f0','e1-f0','e0-g0','e1-g0','e0-l0','e1-l0','e2-f1','e3-f1','e2-g1','e3-g1','e2-l0','e3-l0')
  # 使用 for 循环遍历矩阵的行和列
  result_list <- list()
  
  for (i in 1:nrow(mat)) {
    row_indices <- which(mat[i,] == 1)  # 寻找值为1的元素的列号
    #print(row_indices)
    if (length(row_indices)>0){
      row_names <- funName[c(row_indices)]  # 在funName中寻找对应的名称
      #print(row_names)
      result_list <- append(result_list,list(row_names))  # 将对应名称放入列表中
    }#TODO: Incorrect
  }
  return(result_list)
}


h0 <- hypergraph_from_edgelist(DrawHyperGraphOfFunctions(dselected_rows))
h1 <- hypergraph_from_edgelist(DrawHyperGraphOfFunctions(dselected_rowsf))
h2 <- hypergraph_from_edgelist(DrawHyperGraphOfFunctions(dselected_rowsg))
h3 <- hypergraph_from_edgelist(DrawHyperGraphOfFunctions(dselected_rowsl))
#h2 <- hypergraph.add.edges(h,list(c('e0-f0','e0-g0','e0-l0'),c('e1-f0','e1-g0','e1-l0'),c('e2-f1','e2-g1','e2-l0'),c('e3-f1','e3-g1','e3-l0')))
layout_fr <- layout_with_graphopt(hypergraph2graph(h0),mass = 5,spring.length = 15,spring.constant=0.01)
plot(h0, vertex.color="lightblue", vertex.frame.color="darkblue", vertex.label.color="black",
     edge_colors='red', vertex.label.cex=1.5, vertex.label.dist=-3, vertex.size=20, margin=0.0025,layout=layout_fr)
layout_fr <- layout_with_graphopt(hypergraph2graph(h1),mass = 15,spring.length = 15,spring.constant=0.01)
plot(h1, vertex.color="lightblue", vertex.frame.color="darkblue", vertex.label.color="black",
     edge_colors='red', vertex.label.cex=1.5, vertex.label.dist=-3, vertex.size=20, margin=0.0025,layout=layout_fr)
layout_fr <- layout_with_graphopt(hypergraph2graph(h2),mass = 5,spring.length = 15,spring.constant=0.01)
plot(h2, vertex.color="lightblue", vertex.frame.color="darkblue", vertex.label.color="black",
     edge_colors='red', vertex.label.cex=1.5, vertex.label.dist=-3, vertex.size=20, margin=0.0025,layout=layout_fr)
layout_fr <- layout_with_graphopt(hypergraph2graph(h3),mass = 5,spring.length = 15,spring.constant=0.01)
plot(h3, vertex.color="lightblue", vertex.frame.color="darkblue", vertex.label.color="black",
     edge_colors='red', vertex.label.cex=1.5, vertex.label.dist=-3, vertex.size=20, margin=0.0025,layout=layout_fr)

#
# Discussion subsec1
setwd('g:/15-GEngine/advanced-architecture-scheduler/OptimizerAnalyze')

dfwf<-read.csv('datawf_exp.csv')
# 按照 category 列进行分组
#test <- df[df$Type != "FunAssigned" | df$SolutionQuality != "non-optimal", ]
da_filter <- dfwf[dfwf$Xp != "[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]" ,]
#da_filter <- da_filter[da_filter$Type != "Random" , ]
data_wf0 <- da_filter %>%
  group_by(workloadf0,Type) %>%
  summarise(
    median = median(obj),  # 中位数
    q1 = quantile(obj, 0.25),  # 下四分位数
    q3 = quantile(obj, 0.75),  # 上四分位数
    upper = min(q3, max(obj)),  # 上边缘
    lower = max(q1, min(obj))  # 下边缘
  )
data_wf1 <- da_filter %>%
  group_by(workloadg0,Type) %>%
  summarise(
    median = median(obj),  # 中位数
    q1 = quantile(obj, 0.25),  # 下四分位数
    q3 = quantile(obj, 0.75),  # 上四分位数
    upper = min(q3, max(obj)),  # 上边缘
    lower = max(q1, min(obj))  # 下边缘
  )
data_wf2 <- da_filter %>%
  group_by(workloadl0,Type) %>%
  summarise(
    median = median(obj),  # 中位数
    q1 = quantile(obj, 0.25),  # 下四分位数
    q3 = quantile(obj, 0.75),  # 上四分位数
    upper = min(q3, max(obj)),  # 上边缘
    lower = max(q1, min(obj))  # 下边缘
  )
# Represent it
P1 <- data_wf0 %>%
  ggplot( aes(x=workloadf0, y=median, group=Type, color=Type)) +
  geom_line() +
  #scale_color_viridis(discrete = TRUE) +
  #ggtitle("Performance Comparison of Different Scheduling Mechanisms") +
  #theme_ipsum() +
  scale_y_continuous(limits = c(0, 350)) +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill=Type), alpha = 0.255)+
  ylab("Minmax Workload")+
  xlab(expression(paste("Computational complexity of function ",f[0]," as ", Wf[0])))+
  scale_fill_discrete_qualitative(palette = "Dark 3")+
  theme(
    plot.title = element_text(hjust = 0.5),  # 设置标题居中
    axis.title.y = element_text(hjust = 0.5), # 设置y轴标签居中
    axis.title.x = element_text(hjust = 0.5)  # 如果需要也可以设置x轴标签居中
  )+ theme(legend.position = "none")
P2 <- data_wf1 %>%
  ggplot( aes(x=workloadg0, y=median, group=Type, color=Type)) +
  geom_line() +
  #scale_color_viridis(discrete = TRUE) +
  #ggtitle("Performance Comparison of Different Scheduling Mechanisms") +
  #theme_ipsum() +
  scale_y_continuous(limits = c(0, 350)) +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill=Type), alpha = 0.255)+
  ylab("Minmax Workload")+
  xlab(expression(paste("Computational complexity of function ",g[0]," as ", wg[0])))+
  scale_fill_discrete_qualitative(palette = "Dark 3")+
  theme(
    plot.title = element_text(hjust = 0.5),  # 设置标题居中
    axis.title.y = element_text(hjust = 0.5), # 设置y轴标签居中
    axis.title.x = element_text(hjust = 0.5)  # 如果需要也可以设置x轴标签居中
  )+ theme(legend.position = "none")
P3 <- data_wf2 %>%
  ggplot( aes(x=workloadl0, y=median, group=Type, color=Type)) +
  geom_line() +
  #scale_color_viridis(discrete = TRUE) +
  #ggtitle("Performance Comparison of Different Scheduling Mechanisms") +
  #theme_ipsum() +
  scale_y_continuous(limits = c(0, 350)) +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill=Type), alpha = 0.255)+
  ylab("Minmax Workload")+
  xlab(expression(paste("Computational complexity of function ",l[0]," as ", wl[0])))+
  scale_fill_discrete_qualitative(palette = "Dark 3")+
  theme(
    plot.title = element_text(hjust = 0.5),  # 设置标题居中
    axis.title.y = element_text(hjust = 0.5), # 设置y轴标签居中
    axis.title.x = element_text(hjust = 0.5)  # 如果需要也可以设置x轴标签居中
  )
combined_plot <- P1 + P2 + P3 +
  plot_layout(ncol = 3)

combined_plot

# 同样也分析一下计算任务变化时，任务分配方案在内聚上的表现
# Discussion subsec3
# 找到Funassigned 显著发挥作用的解，对比Xp和Xp*的区别，看看耦合性是如何影响最优的分配结果的

# 创建条件向量
dcondition_o <- (dfwf$workloadf0 == 100) & (dfwf$workloadg0 == 100) & (dfwf$workloadl0 == 100)
dcondition_f <- (dfwf$workloadf0 == 550) & (dfwf$workloadg0 == 100) & (dfwf$workloadl0 == 100)
dcondition_g <- (dfwf$workloadf0 == 100) & (dfwf$workloadg0 == 550) & (dfwf$workloadl0 == 100)
dcondition_l <- (dfwf$workloadf0 == 100) & (dfwf$workloadg0 == 100) & (dfwf$workloadl0 == 550)
dselected_rows <- dfwf[dcondition_o,]
dselected_rowsf <- dfwf[dcondition_f, ]
dselected_rowsg <- dfwf[dcondition_g, ]
dselected_rowsl <- dfwf[dcondition_l, ]

h0 <- hypergraph_from_edgelist(DrawHyperGraphOfFunctions(dselected_rows))
h1 <- hypergraph_from_edgelist(DrawHyperGraphOfFunctions(dselected_rowsf))
h2 <- hypergraph_from_edgelist(DrawHyperGraphOfFunctions(dselected_rowsg))
h3 <- hypergraph_from_edgelist(DrawHyperGraphOfFunctions(dselected_rowsl))
#h2 <- hypergraph.add.edges(h,list(c('e0-f0','e0-g0','e0-l0'),c('e1-f0','e1-g0','e1-l0'),c('e2-f1','e2-g1','e2-l0'),c('e3-f1','e3-g1','e3-l0')))
layout_fr <- layout_with_graphopt(hypergraph2graph(h0),mass = 15,spring.length = 15,spring.constant=0.01)
plot(h0, vertex.color="lightblue", vertex.frame.color="darkblue", vertex.label.color="black",
     edge_colors='red', vertex.label.cex=1.5, vertex.label.dist=-3, vertex.size=20, margin=0.0025,layout=layout_fr)
layout_fr <- layout_with_graphopt(hypergraph2graph(h1),mass = 15,spring.length = 15,spring.constant=0.01)
plot(h1, vertex.color="lightblue", vertex.frame.color="darkblue", vertex.label.color="black",
     edge_colors='red', vertex.label.cex=1.5, vertex.label.dist=-3, vertex.size=20, margin=0.0025,layout=layout_fr)
layout_fr <- layout_with_graphopt(hypergraph2graph(h2),mass = 5,spring.length = 15,spring.constant=0.01)
plot(h2, vertex.color="lightblue", vertex.frame.color="darkblue", vertex.label.color="black",
     edge_colors='red', vertex.label.cex=1.5, vertex.label.dist=-3, vertex.size=20, margin=0.0025,layout=layout_fr)
layout_fr <- layout_with_graphopt(hypergraph2graph(h3),mass = 5,spring.length = 15,spring.constant=0.01)
plot(h3, vertex.color="lightblue", vertex.frame.color="darkblue", vertex.label.color="black",
     edge_colors='red', vertex.label.cex=1.5, vertex.label.dist=-3, vertex.size=20, margin=0.0025,layout=layout_fr)


# 显著性检验定义自定义函数
da_test <- da_filter %>%
  pivot_wider(
    id_cols = c(workloadf0, workloadg0, workloadl0),
    names_from = Type,
    values_from = obj)
print(da_test)
error<-da_test[da_test$FunAssigned>da_test$EntityAssigned,]
print(na.omit(error))
# 单样本 t 检验test_2_1 <- na.omit(da_test$EntityAssigned - da_test$FunAssigned)
test_2_1 <- na.omit(da_test$EntityAssigned - da_test$FunAssigned)
test_3_1 <- na.omit(da_test$FOnlyCU - da_test$FunAssigned)
test_4_1 <- na.omit(da_test$EOnlyCU - da_test$FunAssigned)
test_5_1 <- na.omit(da_test$FOnlyLK - da_test$FunAssigned)
test_6_1 <- na.omit(da_test$EOnlyLK - da_test$FunAssigned)
test_7_1 <- na.omit(da_test$Random - da_test$FunAssigned)
result1 <- t.test(test_2_1, mu = 0, alternative = "greater")
result2 <- t.test(test_3_1, mu = 0, alternative = "greater")
result3 <- t.test(test_4_1, mu = 0, alternative = "greater")
result4 <- t.test(test_5_1, mu = 0, alternative = "greater")
result5 <- t.test(test_6_1, mu = 0, alternative = "greater")
result6 <- t.test(test_7_1, mu = 0, alternative = "greater")
print(result1)
print(result2)
print(result3)
print(result4)
print(result5)
print(result6)


# Dispose of N data
setwd('g:/15-GEngine/advanced-architecture-scheduler/OptimizerAnalyze')
df<-read.csv('dataN.csv')
#da_filter <- df[df$Xp != "[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]" ,] #因为规模变化，没有办法滤除
da_filter <- df[df$obj < 6000,]
data_n0 <- da_filter %>%
  group_by(n0,Type) %>%
  summarise(
    median = median(obj),  # 中位数
    q1 = quantile(obj, 0.25),  # 下四分位数
    q3 = quantile(obj, 0.75),  # 上四分位数
    upper = min(q3, max(obj)),  # 上边缘
    lower = max(q1, min(obj))  # 下边缘
  )
data_n1 <- da_filter %>%
  group_by(n1,Type) %>%
  summarise(
    median = median(obj),  # 中位数
    q1 = quantile(obj, 0.25),  # 下四分位数
    q3 = quantile(obj, 0.75),  # 上四分位数
    upper = min(q3, max(obj)),  # 上边缘
    lower = max(q1, min(obj))  # 下边缘
  )
# Represent it
P1 <- data_n0 %>%
  ggplot( aes(x=n0, y=median, group=Type, color=Type)) +
  geom_line() +
  #scale_color_viridis(discrete = TRUE) +
  #ggtitle("Performance Comparison of Different Scheduling Mechanisms") +
  #theme_ipsum() +
  scale_y_continuous(limits = c(0, 1300)) +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill=Type), alpha = 0.255)+
  ylab("Minmax Workload")+
  xlab(expression(paste("Frequency of function ",f[0]," as ", omega[0])))+
  scale_fill_discrete_qualitative(palette = "Dark 3")+
  theme(
    plot.title = element_text(hjust = 0.5),  # 设置标题居中
    axis.title.y = element_text(hjust = 0.5), # 设置y轴标签居中
    axis.title.x = element_text(hjust = 0.5)  # 如果需要也可以设置x轴标签居中
  )#+ theme(legend.position = "none")
P2 <- data_n1 %>%
  ggplot( aes(x=n1, y=median, group=Type, color=Type)) +
  geom_line() +
  #scale_color_viridis(discrete = TRUE) +
  #ggtitle("Performance Comparison of Different Scheduling Mechanisms") +
  #theme_ipsum() +
  #scale_y_continuous(limits = c(0, 2300)) +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill=Type), alpha = 0.255)+
  ylab("Minmax Workload")+
  xlab(expression(paste("Frequency of function ",g[0]," as ", omega[1])))+
  scale_fill_discrete_qualitative(palette = "Dark 3")+
  theme(
    plot.title = element_text(hjust = 0.5),  # 设置标题居中
    axis.title.y = element_text(hjust = 0.5), # 设置y轴标签居中
    axis.title.x = element_text(hjust = 0.5)  # 如果需要也可以设置x轴标签居中
  )#+ theme(legend.position = "none")
P1
