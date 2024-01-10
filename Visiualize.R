library(ggforce)
#install.packages("ggforce")
library(dplyr)
library(hrbrthemes)
library(viridis)
# Read data.csv
getwd()
setwd('g:/15-GEngine/advanced-architecture-scheduler/OptimizerAnalyze/Log2x2More')
df<-read.csv('data.csv')
# 按照 category 列进行分组
test <- df[df$Type == "FunAssigned" & df$SolutionQuality == "non-optimal", ]
da_filter <- df[df$Type != "FunAssigned" | df$obj != 1.234568, ]
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
  scale_color_viridis(discrete = TRUE) +
  ggtitle("Performance Comparison of Different Scheduling Mechanisms") +
  theme_ipsum() +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill=Type), alpha = 0.3)+
  ylab("Maxmin Workload of Compute Unit and Transport Node")+
  xlab(expression(paste("Frequency of function ",f[0]," as ", omega[0])))+
  theme(
    plot.title = element_text(hjust = 0.5),  # 设置标题居中
    axis.title.y = element_text(hjust = 0.5), # 设置y轴标签居中
    axis.title.x = element_text(hjust = 0.5)  # 如果需要也可以设置x轴标签居中
  )
P2 <- data_w1 %>%
  ggplot( aes(x=omegag0, y=median, group=Type, color=Type)) +
  geom_line() +
  scale_color_viridis(discrete = TRUE) +
  ggtitle("Performance Comparison of Different Scheduling Mechanisms") +
  theme_ipsum() +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill=Type), alpha = 0.3)+
  ylab("Maxmin Workload of Compute Unit and Transport Node")+
  xlab(expression(paste("Frequency of function ",f[0]," as ", omega[0])))+
  theme(
    plot.title = element_text(hjust = 0.5),  # 设置标题居中
    axis.title.y = element_text(hjust = 0.5), # 设置y轴标签居中
    axis.title.x = element_text(hjust = 0.5)  # 如果需要也可以设置x轴标签居中
  )
P3 <- data_w2 %>%
  ggplot( aes(x=omegal0, y=median, group=Type, color=Type)) +
  geom_line() +
  scale_color_viridis(discrete = TRUE) +
  ggtitle("Performance Comparison of Different Scheduling Mechanisms") +
  theme_ipsum() +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill=Type), alpha = 0.3)+
  ylab("Maxmin Workload of Compute Unit and Transport Node")+
  xlab(expression(paste("Frequency of function ",f[0]," as ", omega[0])))+
  theme(
    plot.title = element_text(hjust = 0.5),  # 设置标题居中
    axis.title.y = element_text(hjust = 0.5), # 设置y轴标签居中
    axis.title.x = element_text(hjust = 0.5)  # 如果需要也可以设置x轴标签居中
  )
P3

P1 / P2 / P3
# 使用 gridExtra 包将原始图和局部放大图排列布局
library(gridExtra)

# 创建一个2 x 2的布局
layout <- rbind(c(1, 2, NA), c(3, 3, NA))

Zoom <- data_w2 %>%
  ggplot( aes(x=omegal0, y=median, group=Type, color=Type)) +
  geom_line() +
  scale_color_viridis(discrete = TRUE) +
  theme_ipsum() +
  #geom_ribbon(aes(ymin = lower, ymax = upper, fill=Type), alpha = 0.3)+
  coord_cartesian(xlim = c(20, 40), ylim = c(200, 250)) +
  annotate("rect", xmin = 3, xmax = 8, ymin = 4, ymax = 12, fill = "grey", alpha = 0.2)

# 使用 grid.arrange() 函数进行布局
grid.arrange(P3, Zoom, layout_matrix = layout)+
  ggtitle("Performance Comparison of Different Scheduling Mechanisms")
