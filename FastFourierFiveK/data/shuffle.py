import os
import random
out = open("/ghome/huangjie/Continous/Baseline/groups_train_mixReFive.txt",'w')
lines=[]
with open("/ghome/huangjie/Continous/Baseline/mix.txt", 'r') as infile:
     for line in infile:
         lines.append(line)
random.shuffle(lines)
for line in lines:
    out.write(line)

infile.close()
out.close()

