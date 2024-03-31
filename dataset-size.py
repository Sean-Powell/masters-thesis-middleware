import os

DATASET = "C:/Users/seanp/Desktop/MIT2000/Stimuli/"
min_size = float('inf')
max_size = float('-inf')
for f in os.listdir(DATASET):
    print(f)
    for j in os.listdir(DATASET + "/" + f):
        fileSize = os.path.getsize (DATASET + "/" + f + "/" + j)
        if fileSize >= max_size:
            max_size = fileSize
        
        if fileSize <= min_size:
            min_size = fileSize


print("Max: " + str(max_size / 1000))
print("Min: " + str(min_size / 1000))