def getZeroXingExampleSet(theNeuron)
  minX1 = -4.0
  maxX1 = 4.0
  numberOfTrials = 20
  increment = (maxX1 - minX1) / numberOfTrials
  x0Array = []
  x1Array = []
  numberOfTrials.times do |index|
    x1 = minX1 + index * increment
    x0 = findZeroCrossing(x1, theNeuron)
    next if (x0.nil?)
    x1Array << x1
    x0Array << x0
  end
  [x0Array, x1Array]
end

def findZeroCrossing(x1, theNeuron)
  minX0 = -2.0
  maxX0 = 2.0
  numberOfTrials = 20
  increment = (maxX0 - minX0) / numberOfTrials
  oldX0 = minX0
  return nil if (ioFunctionFor2inputs(minX0, x1, theNeuron) > 0.5)
  numberOfTrials.times do |index|
    x0 = minX0 + (index * increment)
    output = ioFunctionFor2inputs(x0, x1, theNeuron)
    return ((x0 + oldX0) / 2.0) if (output >= 0.5)
    oldX0 = x0
  end
  return nil
end

#def ioFunctionFor2inputs(x0, x1, theNeuron)
#  anExample = [x0, x1]
#  distributeSetOfExamples(anExample)
#  allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(0) }
#  return theNeuron.output
#end

def ioFunctionFor2inputs(x0, x1, theNeuron)
  output = x0 + x1
end


xArray, yArray = getZeroXingExampleSet(nil)

puts "xArray=\t#{xArray}"
puts "yArray=\t#{yArray}"


