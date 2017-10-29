import os
from PIL import Image


inputFolder = "input"
outputFolder = "output"
supportedFileTypes = [".png"]
outputExt = ".dat"
size = (400, 300)


if not os.path.isdir(outputFolder):
	os.mkdir(outputFolder)

filenames = []
for filename in os.listdir(inputFolder):
	for type in supportedFileTypes:
		if filename.lower().endswith(type):
			filenames.append(filename)
			break


for filename in filenames:
	inputPath = os.path.join(inputFolder, filename)
	outputPath = os.path.join(outputFolder, os.path.splitext(filename)[0] + outputExt)

	print(inputPath)
	print(outputPath)

	im = Image.open(inputPath)
	print(im.format, im.size, im.mode)

	im = im.convert("L")
	print(im.format, im.size, im.mode)

	im.thumbnail(size)
	print(im.format, im.size, im.mode)

	data = list(im.getdata())
	with open(outputPath, "w") as file:
		first = True
		for item in data:
			if first:
				file.write("%d" % item)
				first = False
			else:
				file.write(" %d" % item)

