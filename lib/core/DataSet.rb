### VERSION "nCore"
## ../nCore/lib/core/DataSet.rb

require_relative 'Utilities'

module ExampleDistribution

  def distributeDataToInputAndOutputNeurons(examples, arrayOfArraysOfInputAndOutputNeurons)
    arrayOfArraysOfInputAndOutputNeurons.each { |anArray| distributeDataToAnArrayOfNeurons(anArray, examples) }
  end


  def distributeDataToAnArrayOfNeurons(anArray, examples)
    anArray.each_with_index { |aNeuron, arrayIndexToExampleData| distributeSelectedDataToNeuron(aNeuron, examples, aNeuron.keyToExampleData, arrayIndexToExampleData) }
  end

  private

  def distributeSelectedDataToNeuron(theNeuron, examples, keyToExampleData, arrayIndexToExampleData)
    theNeuron.arrayOfSelectedData = examples.collect { |anExample| anExample[keyToExampleData][arrayIndexToExampleData] }
  end
end

############################# Specialized DATA GENERATION Routines ###########################

class DataGenerator
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


################## ExampleList ROTATION of 2-D Inputs ###########################

def extractArrayOfExampleInputVectors(exampleList)
  arrayOfInputRows = []
  exampleList.each do |inputTarget|
    inputRow = inputTarget[0]
    arrayOfInputRows << inputRow
  end
  return arrayOfInputRows
end

def reconstructExampleListUsingNewInputs(originalExampleList, rotatedMatrix)
  arrayOfInputs = rotatedMatrix.to_a
  originalExampleList.each_with_index do |example, exampleNumber|
    example[0] = arrayOfInputs[exampleNumber]
  end
  return originalExampleList
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




