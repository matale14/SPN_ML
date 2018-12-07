import os, pathlib


def rename(path, name):
    count = 0
    for filename in os.listdir(path):
        if filename != "imname.py":
            i = path+"\\"+filename
            e = "{}\\{} ({}).{}".format(path, name, count, filename[-3:])
            print(i)
            print(e)
            os.rename(i, e)
            count += 1
    print("changed",count, "file(s)!")

if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))

    name = "W"  #CHANGE THIS

    rename(path, name)
