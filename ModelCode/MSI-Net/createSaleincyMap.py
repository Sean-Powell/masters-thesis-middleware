import shutil
import os
import subprocess

DATASET = "CAT2000"
TEST_DIR = "test-dir"

for f in os.listdir(DATASET):
    print(f)
    shutil.copy(DATASET + "/" + f, TEST_DIR + "/" + f)
    command = "python main.py test -d cat2000 -p " + TEST_DIR
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    os.remove(TEST_DIR + "/" + f)