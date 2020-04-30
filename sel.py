import os
import random
import shutil

imgl=os.listdir('caiyang/')
sps=random.sample(imgl, 1000)
for sp in sps:
    print(sp)
    shutil.move('caiyang/'+sp,'train/'+sp)