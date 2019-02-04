from os import listdir
from os.path import isfile, join

header = "class,picNum,picType"
lines = []
lines.append(header)
onlyfiles = [f for f in listdir('test') if isfile(join('test', f))]
for i in onlyfiles:

    if i[0] == 'W':
        i = i.strip('.jpg')
        string = "{},{},{}".format('0',i,'Wenche')
    elif i[0] == 'G':
        i = i.strip('.jpg')
        string = "{},{},{}".format('1',i,'Gabbi')
    elif i[0] == 'B':
        i = i.strip('.JPG')
        string = "{},{},{}".format('2',i,'Bjarke')
    elif i[0] == 'M':
        i = i.strip('.JPG')
        string = "{},{},{}".format('3',i,'Monica')
    elif i[0] == 'A':
        i = i.strip('.jpg')
        string = "{},{},{}".format('4',i,'Alex')
    else:
        string = "{},{},{}".format('error',i,'error')
    lines.append(string)

myfile = open('classifications.csv', 'w')
for line in lines:
    myfile.write(line +'\n')

myfile.close()