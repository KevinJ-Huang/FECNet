import os
import random
out = open("/home/jieh/Projects/ExposureFrequency/FastFourierExp1/data/groups_train_mixexposure.txt",'w')
lines=[]
with open("/home/jieh/Projects/ExposureFrequency/FastFourierExp1/data/exposure.txt", 'r') as infile:
     for line in infile:
         lines.append(line)
random.shuffle(lines)
for line in lines:
    out.write(line)

infile.close()
out.close()

