import pandas as pd
from argparse import ArgumentParser
from main import Framework
from measures.metrics import feedbackdata
from main import filedirect
import os
#direct = os.getcwd()
def getfileNames():
    '''
    Function to Return Names of all the datasets
    '''
    import os
    direct = os.getcwd()
    direct = direct + '/Datasets'
    os.chdir(direct)
    datalist = list((os.listdir()))
    DatasetNames = []

    for items in datalist:
        Name, _ = os.path.splitext(items)
        DatasetNames.append(Name)

    return (DatasetNames)

def differentiator(datalist):
    # Extraction of Datasets
    datalistd = datalist
    global filedirect
    direct = filedirect
    print(direct, 'DataExtraction')
    directl = direct + '/savedresult'
    os.chdir(directl)
    datalistl = list((os.listdir()))
    # Extraction of Saved Results
    '''
    directl = direct + '/savedresult'
    os.chdir(directl)
    datalist = list((os.listdir()))
    '''

    # Listing Names of DataSets Uploaded
    datadone = []  # Empty list for data already done
    newdatalist = []  # empty list for data to be done

    # To collect names of datadone
    for i in range(len(datalistl)):
        alg, namedata = datalistl[i].split('-')
        name, _ = namedata.split('.')
        datadone.append(name)
    print(datadone)
    # loop for distinguishing the data done and to be done
    for i in range(len(datalistd)):
        name, formatx = datalistd[i].split('.')
        if name not in datadone:
            newdatalist.append(datalistd[i])

    # loop to distinguish if the dataset has already been run
    if len(newdatalist) == 0:
        return('EMPTY')
    return newdatalist



def DataExtraction():
    '''
    Function To Return list of all datasets in the framework
    '''
    import os
    import pandas as pd
    global filedirect

    direct = filedirect
    print(direct, 'DataExtraction')
    direct = direct + '/Datasets'
    os.chdir(direct)
    datalist = list((os.listdir()))
    datalist = differentiator(datalist)
    if datalist == 'EMPTY':
        return False
    Datasets = []
    DatasetNames = []
    for items in datalist:
        File = direct + '/' + items
        df = pd.read_csv(File)
        Datasets.append(df)

    for items in datalist:
        Name, _ = os.path.splitext(items)
        DatasetNames.append(Name)
    #print(type(Datasets))
    #print(type(DatasetNames))
    #print(DatasetNames)

    dictsets = {k:v for k,v in zip(DatasetNames, Datasets)}

    return dictsets


def sampling(data):
  '''
  Input: Dataset; DataFrame
  Output: Dataframe with random but sorted rows
  Randomly selects 75% of the original dataset to form
  final dataset but in Ascending order.
  '''
  import random
  import pandas as pd
  import numpy as np
  boot = data.shape[0]
  p = set()
  bootr = int(boot*0.90)
  print(bootr)
  #p = []
  for x in range(bootr):
    p.add(random.randint(0,boot-1))
  p = list(p)
  p = np.sort(p)
  final = data.iloc[p,:]
  return final


#Does extraction of dataset, combines with name as key and data in the valueset
Datas = DataExtraction()


parser = ArgumentParser()
parser.add_argument('Option', help='Option')
parser.add_argument('Runs', help='Runs')

args = parser.parse_args()
selector = str(args.Option)
Runs = int(args.Runs)
if Datas ==False:
    selector = 'NULL'
print(Runs)
print(Datas)
#Main if loop to either train or show the results of the framework
if selector == 'Train':
    freshlist = [[[0 for col in range(Runs)] for col in range(7)] for col in range(4)]

    for keys in Datas.keys():
        iterator = 0
        datasetframe = pd.DataFrame(Datas[keys])
        for i in range(Runs):
            print(datasetframe.shape , ' FIRST')
            datasetframe1 = datasetframe #sampling(datasetframe)
            print(datasetframe1.shape, 'SECONDDDD')
            first = Framework(datasetframe1, keys,freshlist,iterator,Runs)
            #first.Iocsvm()
            first.C_AE()
            #first.Ae()
            #first.SVM()
            print(keys)
            iterator+=1


if selector == 'Show':
    import os

    global filedirect
    filer = filedirect
    os.chdir(filer)
    print(filer, 'here')
    print(os.listdir())
    myCmd = 'export FLASK_APP=flaskblog.py; flask run'
    # myCmd2 = 'flask run'
    import webbrowser

    # Assigning URL to be opened
    strURL = "http://localhost:5000/"
    # Open url in default browser
    webbrowser.open(strURL, new=2)
    os.system(myCmd)
    # os.system(myCmd2)
#print(Datas[0])
#First = Framework(Datas[0])
#a,b = First.Ae()
#tn, fp, fn, tp, d, r = feedbackdata(a,b)
#print(tn, fp, fn, tp, d, r )

