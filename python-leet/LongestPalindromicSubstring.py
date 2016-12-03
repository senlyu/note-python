s='babad'
maxn=0
list3=[]
for i in range(len(s)):
    for j in range(i,len(s)):
        list1=[]
        for k in range(i,j+1):
            list1.append(s[k])
            list2=list1[:]
            list2.reverse()
            print('print the list the ijk',list1,list2,i,j,k,k-i+1,maxn)
            if list1==list2 and k-i+1>maxn:
                list3=list1
                maxn=k-i+1
                print('print the list3',list3)
