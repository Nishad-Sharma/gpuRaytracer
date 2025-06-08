run the following to compile a metallib:

'''xcrun -sdk macosx metal -c add.metal -o add.air'''
'''xcrun -sdk macosx metallib add.air -o MyMetalLib.metallib'''

then you can ```swift run```