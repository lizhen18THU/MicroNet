# MicroNet
MicroNet Recurrence


------------------------------------------------------------------------------------------------------------------------
model                         params              FLOPs            TOP1            TOP5                    comment
------------------------------------------------------------------------------------------------------------------------
0.1×GhostNet              0.29M              6.87M            56.85             81.94                   width=0.1

MicroNet_M0_no1       1.07M              6.03M           54.54              79.31            非线性环节dynamic-shift-max的reduction取4，并且depthwise_conv和bn后面没有加任何非线性环节，
								论文中这里没有加dynamic-shift-max，也没有提及是否要加非线性环节
MicroNet_M0_no2       1.05M              5.98M           53.88              78.87             非线性环节dynamic-shift-max的reduction使用的是16，并调整了网络最后全连接层的bottleneck结构
								以满足6MFLOPs要求，同时将缺少非线性环节的depthwise_conv和bn后面加上了Relu
MicroNet_M0_no3       1.05M              5.98M           59.20              82.76             在no2的基础上，参考论文二作的文章dynamic-ReLU将dynamic-shift-max加上了alphas和
								lambdas等固定参数，其他的settings相对于no2没有改变
