s='bbaab'

def pppp(i,j):
    if i==j:
        return True
    elif i==j-1 and s[i]==s[j]:
        return True
    elif s[i]!=s[j]:
        return False
    elif i<j:
        return pppp(i+1,j-1) and s[i]==s[j]
    else:
        return False

for i in range(len(s)):
    for j in range(len(s)):
        if pppp(i,j):
            print('Yes',i,j)


