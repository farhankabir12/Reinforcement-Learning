L = [1.5, 4.3, 0,1]

# writing to file
file1 = open('tmp/run1/myfile' + str(1) + '.txt', 'w')
file1.writelines(str(L))

file1.close()