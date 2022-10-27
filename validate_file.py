import os
file = open('/content/drive/MyDrive/tongji_test/train_label.txt')
txt=file.readlines()
a=[]
for w in txt:
    w=w.replace('\n','')
    a.append(w.split(' ')[0])
filelist = os.listdir('/content/drive/MyDrive/tongji_test/train')
for item in a:
  if item not in filelist:
    print(item)
