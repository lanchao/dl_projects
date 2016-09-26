input = '/Users/lanchao/Downloads/car_detection.txt'
outfile = '/Users/lanchao/Downloads/car_detection_out.txt'

with open(input) as file:
    with open(outfile, 'w') as out:
        for line in file:
            num = line.split(' ')
            re = 1 - float(num[0])
            newline = '{:.6f} {:s} {:s} {:s}'.format(re, num[1], num[2], num[3])
            out.write(newline)

