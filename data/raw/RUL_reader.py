import pandas as pd
import numpy as np

# loading funtion 
dirs = "C:\\Users\\vinayak tyagi\\Desktop\\CMAPSSData\\"
dirs_2 = "F:\\Python Projects\\nasa_PMM\\"

def readfile(filename):
        list_of_lists = []
        path = dirs + filename 
        file = open(path ,'r')
        txt = file.read()
        print(txt.splitlines())
        print(len(txt.splitlines()))
##        for i in txt.splitlines():
##            print(i)


readfile("RUL_FD001.txt")
