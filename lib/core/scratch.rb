require_relative 'Utilities'
require_relative 'DataSet'

################## ExampleList ROTATION of 2-D Inputs ###########################

include ExampleDistribution

#def createRotationMatrix(angleInDegrees) # for CLOCKWISE Rotations!!
#  angleInRadians = angleInDegrees * ((2.0*Math::PI)/360.0)
#  s = Math.sin(angleInRadians)
#  c = Math.cos(angleInRadians)
#  # puts "angleInRadians= #{angleInRadians}\ts= #{s}\tc= #{c}\t    "
#  rotationMatrix = Matrix[[c, -s], [s, c]]
#  # puts "rotationMatrix= #{rotationMatrix} "
#  return rotationMatrix
#end
#
#def reconstructExampleListUsingNewInputs(examples, rotatedMatrix)
#  arrayOfInputs = rotatedMatrix.to_a
#  examples.each do |example|
#    example[:inputs] = arrayOfInputs[example[:exampleNumber]]
#  end
#  return examples
#end
#
#
#def extractArrayOfExampleInputVectors(examples)
#  arrayOfInputRows = []
#  examples.each do | anExampleRow |
#    inputRow = anExampleRow[:inputs]
#    arrayOfInputRows << inputRow
#  end
#  return arrayOfInputRows
#end


def createTrainingSet
  xStart = [-1.0, 1.0, -1.0, 1.0]
  # xStart = [0.0, 2.0, 0.0, 2.0]
  # xStart = [1.0, 3.0, 1.0, 3.0]

  yStart = [1.0, 1.0, -1.0, -1.0]
  # yStart = [4.0, 4.0, 0.0, 0.0]

  xInc = [0.0, 0.0, 0.0, 0.0]
  xInc = [0.0, 0.0, 0.0, 0.0]

  yInc = [0.0, 0.0, -0.0, -0.0]
  # yInc = [0.2, 0.2, -0.2, -0.2]
  yInc = [0.0, 0.0, -0.0, -0.0]

  numberOfClasses = xStart.length
  numberOfExamples = 4
  numberOfExamplesInEachClass = numberOfExamples / numberOfClasses
  exampleNumber = 0
  examples = []
  numberOfClasses.times do |indexToClass|
    xS = xStart[indexToClass]
    xI = xInc[indexToClass]
    yS = yStart[indexToClass]
    yI = yInc[indexToClass]

    numberOfExamplesInEachClass.times do |classExNumb|
      x = xS + (xI * classExNumb)
      y = yS + (yI * classExNumb)
      aPoint = [x, y]
      desiredOutputs = [0.0, 0.0, 0.0, 0.0]
      desiredOutputs[indexToClass] = 1.0
      examples << {:inputs => aPoint, :targets => desiredOutputs, :exampleNumber => exampleNumber, :class => indexToClass}
      exampleNumber += 1
    end
  end
  STDERR.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
  puts examples
  examples
end


examples = createTrainingSet



angleOfClockwiseRotationOfInputData = 45.0

# Rotation CLOCKWISE:
rotationMatrix = createRotationMatrix(angleOfClockwiseRotationOfInputData)
matrixToBeRotated = Matrix.rows(extractArrayOfExampleInputVectors(examples))

puts matrixToBeRotated

rotatedMatrix = matrixToBeRotated * rotationMatrix

puts rotatedMatrix

examples = reconstructExampleListUsingNewInputs(examples, rotatedMatrix)

puts "result for rotated exampleList= #{examples} "

