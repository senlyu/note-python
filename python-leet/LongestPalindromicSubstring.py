maxn=0
list3=[]
for i in range(len(s)):
    for j in range(i,len(s)):
        list1=[]
        for k in range(i,j):
            list1.append(s[k])
            list2=list1.reverse
            print(list1,i,j,k)
            if list1==list2 and k-i+1>maxn:
                list3=list1
                print(list3)
