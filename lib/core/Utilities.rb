# Utilities.rb

require 'mathn'
require 'matrix'


module OS
  def OS.windows?
    (/cygwin|mswin|mingw|bccwin|wince|emx/ =~ RUBY_PLATFORM) != nil
  end

  def OS.mac?
    (/darwin/ =~ RUBY_PLATFORM) != nil
  end

  def OS.unix?
    !OS.windows?
  end

  def OS.linux?
    OS.unix? and not OS.mac?
  end
end

############################# DATA GENERATION ###########################

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

############################# DEBUG UTILITY FUNCTIONS ###########################

def std(txt, x)
  STDOUT.puts "#{txt}\t#{x}"; STDOUT.flush
end

def qreport(dataArray, epochNumber, interval)
  if ((epochNumber-1).modulo(interval) == 0)
    STDOUT.print "Epoch Number=\t#{epochNumber} --\t"
    dataArray.each_with_index do |dataItem, indexToDataItem|
      STDOUT.print "dataItem #{indexToDataItem} =\t#{dataItem};\t"
    end
    STDOUT.flush
    STDOUT.puts
  end
end

def periodicallyDisplayContentsOfHash(hashWithData, epochNumber, interval)
  if ((epochNumber-1).modulo(interval) == 0)
    STDOUT.print "Epoch Number=\t#{epochNumber} -->\t"
    hashWithData.each do |key, value|
      STDOUT.print "#{key} =\t#{value};\t"
    end
    STDOUT.flush
    STDOUT.puts
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



#TODO Subclass Vector and add these methods to the subclass
class Vector
  # Calculates the distance to Point p
  def dist_to(p)
    return (self - p).r
  end
end

#	##########################  Array Extensions ##################
class Array
  def shuffle
    sort { rand <=> 0.5 }
  end

  def shuffle!
    replace shuffle
  end

  def mean
    sumOfArray = self.inject { |sum, n| sum + n }
    return (sumOfArray / self.length)
  end

  def standardError
    meanOfArray = self.mean
    sumOfSquares = self.inject { |sum, n| sum + ((n-meanOfArray)**2) }
    return Math.sqrt(sumOfSquares / self.length)
  end

  def normalize
    maximum = (self.max).to_f
    self.collect { |value| value / maximum }
  end

  def scaleValuesToSumToOne
    sumOfArray = self.inject(0.0) { |sum, value| sum + value }
    self.collect { |value| value / sumOfArray }
  end
end

#	##########################  Object Extensions ##################
class Object
  def blank?
    respond_to?(:empty?) ? empty? : !self
  end

  def present?
    !blank?
  end
end



