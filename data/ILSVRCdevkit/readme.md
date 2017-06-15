处理imagenet各种问题

先getpositive，把正样本图片输出到positive.txt

然后get_sizes，读图片大小。这个代码得改最后几行，设置是train还是test

然后replace_data，把宽高比太大的图片加pad，输出到data_to_replace文件夹。但是这个文件夹好像得先手动建好？？

然后需要对train集再跑一遍get_sizes。。。因为加了pad，需要更新，里边会自己去读data_to_replace

用之前最好看一遍代码，尤其是里边的路径。。。可能是VID的路径，因为后来VID也是这个代码跑的