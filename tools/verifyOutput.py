import os
import sys


def parseVector(line: str):
    newString = line.strip().replace('  ', ' ')
    returnList = [float(i) for i in newString.split(" ")]
    if len(returnList) == 1:
        print("Vector should probably not be length 1...")
    return returnList


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

        self.path = os.path.join(os.getcwd(), filePath)
        print(f"Loading data from: {self.path}")

        lines = None
        with open(self.path, 'r') as f:
            lines = f.readlines()

        self.nBody = int(lines[0])
        self.nDim = int(lines[1])
        index = 2
        while True:
            self.timesteps.append(timestep(lines, self.nBody, index))
            index += (1+3*self.nBody) #Set the index at the correct position. (1 for timestamp 3*nbody for mass, coordinates and velocities)
            index += 2 #Skip the next lines of nBody and nDim.
            if index >= len(lines): #Stop when end of lines found.
                break

    def fileDataEqual(self, compareData):
        if self.nBody != compareData.nBody:
            print(f"Fail: nBody not equal. {self.nBody} != {compareData.nBody}")
            return False
        if self.nDim != compareData.nDim:
            print(f"Fail: nDim not equal. {self.nDim} != {compareData.nDim}")
            return False
        if len(self.timesteps) != len(compareData.timesteps):
            print(f"Fail: Nr of timesteps not equal. {self.timesteps} != {compareData.timesteps}")
            return False

        for i in range(len(self.timesteps)):
            if not self.timesteps[i].timestepEqual(self.nBody, self.nDim, i, compareData.timesteps[i]):
                return False

        return True


class timestep:

    def __init__(self, lines, nbody, index):
        self.bodyMass = list()
        self.bodyCoord = list()
        self.bodyVelocities = list()
        localIdx = index #Just in case I misunderstand python pass by reference things.
        self.timeStamp = float(lines[localIdx])
        localIdx += 1
        try:
            for i in range(nbody):
                self.bodyMass.append(float(lines[localIdx]))
                localIdx += 1
            for i in range(nbody):
                self.bodyCoord.append(parseVector(lines[localIdx]))
                localIdx += 1
            for i in range(nbody):
                self.bodyVelocities.append(parseVector(lines[localIdx]))
                localIdx += 1
        except:
            print(f"Error while parsing timestep at line nr {localIdx+1}")
            sys.exit(2)


    def timestepEqual(self, nbody, ndim, timestepIdx ,filestep):
        epsilon = 1e-3
        if abs(self.timeStamp - filestep.timeStamp) > epsilon:
            print(f"Fail: Timestamp not equal. TimeIndx {timestepIdx} {self.timeStamp} != {filestep.timeStamp}")
            return False

        for i in range(nbody):
            if abs(self.bodyMass[i] - filestep.bodyMass[i]) > epsilon:
                print(f"Fail: Masses not equal. TimeIndx {timestepIdx} {self.bodyMass[i]} != {filestep.bodyMass[i]}")
                return False
            if not vectorEqual(self.bodyCoord[i], filestep.bodyCoord[i], epsilon):
                print(f"Fail: Coordiantes not equal. TimeIndx {timestepIdx} {self.bodyCoord[i]} != {filestep.bodyCoord[i]}")
                return False
            if not vectorEqual(self.bodyVelocities[i], filestep.bodyVelocities[i], epsilon):
                print(f"Fail: Velocity not equal. TimeIndx {timestepIdx} {self.bodyVelocities[i]} != {filestep.bodyVelocities[i]}")
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
    sys.exit(0)  ##Exit without error when equal.
else:
    sys.exit(1)  ##Exit with error when not equal.
