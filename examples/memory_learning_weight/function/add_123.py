from os import listdir
from os.path import isfile, join
import os

mypath = join("..", "result","assistment")


for fileName in listdir(mypath):
    current_file = join(mypath, fileName)
    if isfile( current_file ):
        if "(" not in fileName:
            os.rename(current_file, join(mypath, fileName+"(3)"))
