from os import listdir
from os.path import isfile, join
import ast

mypath = join("..", "result")

thread_auc = 0.87
allFileCount = 0
for fileName in listdir(mypath):
    current_file = join(mypath, fileName)
    if isfile( current_file ):
        f_result = open(current_file)
        # test_auc / train_auc / test_loss / train_loss / test_accuracy / train_accuracy
        count = 0
        for line in f_result:
            if line[0] == "{" and count < 1:
                allFileCount += 1
                line = line.strip("\n")
                dic = ast.literal_eval(line)
                #print line
                count += 1
                one_result_max = 0
                one_result_epoch = 0
                for i in dic:
                    #print i, type(i), dic[i], type(dic[i])
                    test_auc = dic[i]
                    if test_auc > one_result_max:
                        one_result_max = test_auc
                        one_result_epoch = i
                if one_result_max > thread_auc:
                    print fileName," \t ===> one_result_epoch",one_result_epoch, "one_result_max", one_result_max
        #print "~~~~~~"
        f_result.close()

print allFileCount
