hight = 1.75
weight = 80.5
BMI =  weight/(hight*hight)
print(BMI)
if BMI < 18.5:
    print('too thin')
elif BMI >= 18.5 and BMI < 25:
    print('normal')
elif BMI >= 25 and BMI < 28:
    print('overweight')
elif BMI >=28 and BMI <32:
    print('fat')
elif BMI >=32:
    print('obcity')



