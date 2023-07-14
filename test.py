import matplotlib.pyplot as plt
x=[1,2,3,3.5,2]
bins=[0.5,1.5, 2.5]
first_bin=[.5,1.5)
second_bin=[1.5, 2.5)
third_bin=[2.5,3.5]
third_bin = [3,4]
count, bins=plt.hist(x, bins=bins, label='lala')
print(count) = [1,2]
plt.legend()
#plt.show()
filename=filedir+filename
plt.savefig(filename)


