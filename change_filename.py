import os
path = r"C:\Users\VIPLAB\Desktop\Yan\Lee's Family Reunion labeled\L\\"
files=os.listdir(path)
print(files)

n=0
for i in files:
    oldname = path + files[n]
    newname = path +"L_"+ str(n) + '.jpg'
    os.rename(oldname,newname)
    print(oldname + '>>>'+ newname)
    n=n+1