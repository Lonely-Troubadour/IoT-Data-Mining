dict_ = dict()

dict_[frozenset(['a'])] = 1
dict_[frozenset(['p'])] = 3
dict_[frozenset(['f'])] = 4
dict_[frozenset(['m'])] = 2
dict_[frozenset(['c'])] = 3
dict_[frozenset(['b'])] = 2

print(dict_.keys())

d = dict.fromkeys(dict_.keys(), list())

for key, value in d.items():
    print(key, value)

i = 0
for item in dict_.keys():
    print(item)
    d[item].append(['abc'])
    i += 1
    if i == 2:
        break
    

print(d)