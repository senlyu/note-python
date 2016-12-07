
class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        ss=[]
        a=''
        for i in range(len(s)+1):
                ss.append('')
        if numRows >=len(s) or numRows ==1:
            return s
        elif numRows==2:
            for i in range(len(s)):
                if i%2==0:
                    ss[1]+=s[i]
                else:
                    ss[2]+=s[i]
            return ss[1]+ss[2]
        else:
            for i in range(len(s)):
                if i//(numRows-1) % 2 ==0 :
                    ss[(i+1)%(2*numRows-2)]+=s[i]
                else:
                    if (i+1)%(2*numRows-2)==0:
                        ss[2]+=s[i]
                    else:
                        ss[numRows-(((i+1) % (2*numRows-2))-numRows)]+=s[i]
            if s:
                for j in range(numRows+1):
                    a+=(ss[j])
            return a
