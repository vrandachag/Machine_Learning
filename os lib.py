import os
import fnmatch
for root,dirs,files in os.walk('C:/Users/chagv/Downloads/archive'):
  for n in files:
    images = os.listdir('C:/Users/chagv/Downloads/archive')
    if fnmatch.fnmatch(n,'*.jpg'):
      print(n)
print(images)
