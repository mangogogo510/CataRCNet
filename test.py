def num2(s1,s2):
    #m = [[0 for i in range(len())]]
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    mmax = 0
   
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1,j+1]= m[i,j]+1
                if m[i+1,j+1]>mmax:
                    
  return mmax

while True:
    
    try:
        a,b = input(),input()
        print(num2(a,b))
    except:
        break
