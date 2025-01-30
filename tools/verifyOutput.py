import sys


def parseVector(line: str):
    newString = line.strip().replace('  ', ' ')
    return [float(i) for i in newString.split(" ")]

###Returns true if vector diff is less than epsilon
def vectorEqual(vector1, vector2, epsilon):
    for i in range(len(vector1)):
        if abs(vector1[i] - vector2[i]) > epsilon:
            return False

    return True

class fileData:
    def __init__(self, filePath):
        self.timesteps = list()
        self.nBody = 0
        self.nDim = 0
        self.filePath = filePath
        with open(filePath, 'r') as f:
            self.nBody = int(f.readline())
            self.nDim = int(f.readline())
            while True:
                self.timesteps.append(timestep(f, self.nBody))
                f.readline()  ##Reads and discards nbody and nDim.
                if f.readline() == '':
                    break

    def fileDataEqual(self, compareData):
        if self.nBody != compareData.nBody:
            return False
        if self.nDim != compareData.nDim:
            return False
        if len(self.timesteps) != len(compareData.timesteps):
            return False

        for i in range(len(self.timesteps)):
            if not self.timesteps[i].timestepEqual(self.nBody, self.nDim, compareData.timesteps[i]):
                return False


        return True

class timestep:

    def __init__(self, fileHandle, nbody):
        self.bodyMass = list()
        self.bodyCoord = list()
        self.bodyVelocities = list()
        self.timeStamp = float(fileHandle.readline())
        for i in range(nbody):
            self.bodyMass.append(float(fileHandle.readline()))

        for i in range(nbody):
            self.bodyCoord.append(parseVector(fileHandle.readline()))

        for i in range(nbody):
            self.bodyVelocities.append(parseVector(fileHandle.readline()))

    def timestepEqual(self, nbody, ndim, filestep):
        epsilon = 1e-5
        if abs(self.timeStamp - filestep.timeStamp) > epsilon:
            return False

        for i in range(nbody):
            if abs(self.bodyMass[i] - filestep.bodyMass[i]) > epsilon:
                return False
            if not vectorEqual(self.bodyCoord[i], filestep.bodyCoord[i], epsilon):
                return False
            if not vectorEqual(self.bodyVelocities[i], filestep.bodyVelocities[i], epsilon):
                return False

        return True

if len(sys.argv) != 3:
    print("Wrong amount of arguments")
print(f"Comparing file {sys.argv[1]} and {sys.argv[2]}")

fd = fileData(sys.argv[1])
fd2 = fileData(sys.argv[2])


equal = fd.fileDataEqual(fd2)
print(f"Data equal? {equal}")
if equal:
    sys.exit(0) ##Exit without error when equal.
else:
    sys.exit(1) ##Exit with error when not equal.

