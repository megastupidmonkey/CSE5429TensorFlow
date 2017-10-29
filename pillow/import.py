#Basic structure for reading the list of pixel values


import os


inputFolder = "output"
inputExt = ".dat"


filenames = []
for filename in os.listdir(inputFolder):
	if filename.lower().endswith(inputExt):
		filenames.append(filename)


for filename in filenames:
	filePath = os.path.join(inputFolder, filename)
	print(filePath)

	data = []
	with open(filePath, "r") as file:
		for line in file:
			line = line.split()
			if line:
				line = [int(i) for i in line]
				data.extend(line)

	#print(data)
	print("len = %d" % len(data))

