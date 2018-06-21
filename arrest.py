//日付map
def date(x):
    b=int(str(x).split('-')[1])
    if  str(x).split('-')[0]=='2013':
        a=0
    elif str(x).split('-')[0]=='2014':
        a=12
    elif str(x).split('-')[0]=='2015':
        a=24
    return a+b

//時間map
def time(x):
    return str(x).split(':')[0]

//年齢map
def age(x):
    return  str(int(int(x)/10))
		
//文字列変換   
def toStr(x):
    return str(x)

//文字列を分離して個数をリターン		
def splitStr(x):
    return len(str(x).split(','))
