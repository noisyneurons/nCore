### VERSION "nCore"
## ../nCore/lib/core/DataSet.rb

require_relative 'Utilities'
require_relative '../../../rubystats/lib/normal_distribution'


module ExampleDistribution

  def distributeDataToInputAndOutputNeurons(examples, arrayOfArraysOfInputAndOutputNeurons)
    arrayOfArraysOfInputAndOutputNeurons.each { |anArray| distributeDataToAnArrayOfNeurons(anArray, examples) }
  end

  def distributeDataToAnArrayOfNeurons(anArray, examples)
    anArray.each_with_index { |aNeuron, arrayIndexToExampleData| distributeSelectedDataToNeuron(aNeuron, examples, aNeuron.keyToExampleData, arrayIndexToExampleData) }
  end

  def distributeSelectedDataToNeuron(theNeuron, examples, keyToExampleData, arrayIndexToExampleData)
    theNeuron.arrayOfSelectedData = examples.collect { |anExample| anExample[keyToExampleData][arrayIndexToExampleData] }
  end

  def normalizeDataSet(examples)
    transformedExampleDataSet = examples.deep_clone
    arrayOfInputRows = extractArrayOfExampleInputVectors(examples)
    numberOfNetworkInputs = arrayOfInputRows[0].length
    std("numberOfNetworkInputs ", numberOfNetworkInputs)
    numberOfNetworkInputs.times do |inputNumber|
      std("inputNumber ", inputNumber)
      allExampleValuesForOneNetworkInput = extractArrayOfAllExamplesForJustOneNetworkInput(inputNumber, arrayOfInputRows)
      std("\tallExampleValuesForOneNetworkInput ", allExampleValuesForOneNetworkInput)
      normalizedExampleValuesForInput = allExampleValuesForOneNetworkInput.normalize
      std("\tnormalizedExampleValuesForInput ", normalizedExampleValuesForInput)
      insertExampleValuesForInput(transformedExampleDataSet, normalizedExampleValuesForInput, inputNumber)
    end
    transformedExampleDataSet
  end

  def extractArrayOfExampleInputVectors(examples)
    arrayOfInputRows = []
    examples.each do |anExample|
      inputRow = anExample[:inputs]
      arrayOfInputRows << inputRow
    end
    return arrayOfInputRows
  end

  def extractArrayOfAllExamplesForJustOneNetworkInput(inputNumber, arrayOfInputRows)
    arrayOfInputRows.collect { |aRow| aRow[inputNumber] }
  end

  def insertExampleValuesForInput(examples, normalizedExampleValuesForInput, inputNumber)
    examples.each_with_index do |anExample, exampleNumber|
      anExample[:inputs][inputNumber] = normalizedExampleValuesForInput[exampleNumber]
    end
  end

  ################## ExampleList ROTATION of 2-D Inputs ###########################

  def rotateClockwise(examples, angleOfClockwiseRotationOfInputData)
    rotationMatrix = createRotationMatrix(angleOfClockwiseRotationOfInputData)
    matrixToBeRotated = Matrix.rows(extractArrayOfExampleInputVectors(examples))
    rotatedMatrix = matrixToBeRotated * rotationMatrix
    reconstructExampleListUsingNewInputs(examples, rotatedMatrix)
  end

  def createRotationMatrix(angleInDegrees) # for CLOCKWISE Rotations!!
    angleInRadians = angleInDegrees * ((2.0*Math::PI)/360.0)
    s = Math.sin(angleInRadians)
    c = Math.cos(angleInRadians)
    # puts "angleInRadians= #{angleInRadians}\ts= #{s}\tc= #{c}\t    "
    rotationMatrix = Matrix[[c, -s], [s, c]]
    # puts "rotationMatrix= #{rotationMatrix} "
    return rotationMatrix
  end

  def reconstructExampleListUsingNewInputs(examples, rotatedMatrix)
    arrayOfInputs = rotatedMatrix.to_a
    examples.each do |example|
      example[:inputs] = arrayOfInputs[example[:exampleNumber]]
    end
    return examples
  end

  def extractArrayOfExampleInputVectors(examples)
    arrayOfInputRows = []
    examples.each do |anExampleRow|
      inputRow = anExampleRow[:inputs]
      arrayOfInputRows << inputRow
    end
    return arrayOfInputRows
  end
end

############################# Specialized DATA GENERATION Routines ###########################


class GenerateDataSet
  attr_accessor :args
  include ExampleDistribution

  def initialize(args)
    @args = args
  end
end

class XORDataGenerator < GenerateDataSet
  def generate(numberOfExamples=4, standardDeviationOfAddedGaussianNoise=0.0)
    examples = []
    examples << {:inputs => [0.0, 0.0], :targets => [0.0], :exampleNumber => 0, :class => 0}
    examples << {:inputs => [0.0, 1.0], :targets => [1.0], :exampleNumber => 1, :class => 1}
    examples << {:inputs => [1.0, 0.0], :targets => [1.0], :exampleNumber => 2, :class => 1}
    examples << {:inputs => [1.0, 1.0], :targets => [0.0], :exampleNumber => 3, :class => 0}
    return examples
  end
end


class Generate4ClassDataSet < GenerateDataSet
  def generate(numberOfExamples, standardDeviationOfAddedGaussianNoise)
    gaussianRandomNumberGenerator = NormalDistribution.new(meanOfGaussianNoise = 0.0, standardDeviationOfAddedGaussianNoise)

    vs = args[:verticalShift]
    hs = args[:horizontalShift]

    xStart = [(-1.0 + hs), (1.0 + hs), (-1.0 + hs), (1.0 + hs)]
    yStart = [(1.0 + vs), (1.0 + vs), (-1.0 + vs), (-1.0 + vs)]

    xInc = [0.0, 0.0, 0.0, 0.0]
    yInc = [0.0, 0.0, 0.0, 0.0]

    numberOfClasses = xStart.length
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
        x += gaussianRandomNumberGenerator.get_rng unless (standardDeviationOfAddedGaussianNoise < 1e-30)
        y = yS + (yI * classExNumb)
        y += gaussianRandomNumberGenerator.get_rng unless (standardDeviationOfAddedGaussianNoise < 1e-30)
        aPoint = [x, y]
        desiredOutputs = [0.0, 0.0, 0.0, 0.0]
        desiredOutputs[indexToClass] = 1.0
        examples << {:inputs => aPoint, :targets => desiredOutputs, :exampleNumber => exampleNumber, :class => indexToClass}
        exampleNumber += 1
      end
    end
    STDERR.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
    angleOfClockwiseRotationOfInputData = args[:angleOfClockwiseRotationOfInputData]
    examples = rotateClockwise(examples, angleOfClockwiseRotationOfInputData)
  end
end


class PreviousDataGenerator
  attr_reader :arrayOfExamples

  def initialize(arrayOf2DInputSeedArrays, numberOfExamplesPerClass = 1, increment = 1.0)
    numberOfClasses = arrayOf2DInputSeedArrays.length
    @arrayOfExamples = []
    arrayOf2DInputSeedArrays.each_with_index do |seedArray, indexToOutputDimensionToSet|
      anExample = nil
      inputs = seedArray.clone
      numberOfExamplesPerClass.times do |i|
        outputs = Array.new(numberOfClasses) { 0.0 }
        outputs[indexToOutputDimensionToSet] = 1.0
        anExample = Example.new(inputs, outputs)
        @arrayOfExamples << anExample
        inputs = inputs.clone
        inputs[1] += increment
      end
    end
  end

  def arrayOfExamples
    @arrayOfExamples.each_with_index do |anExample, exampleNumber|
      inputs = anExample[0]
      outputs = anExample[1]
      puts "#{exampleNumber}\tinputs= #{inputs};\toutputs= #{outputs};"
    end
    return @arrayOfExamples
  end
end




