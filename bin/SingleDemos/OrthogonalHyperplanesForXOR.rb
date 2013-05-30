### VERSION "nCore"
## ../nCore/bin/SingleDemos/OrthogonalHyperplanesForXOR.rb# Purpose:  Attempt to learn orthogonal hyperplanes using flocking...
## TODO By using a significant amount of flocking during the later stages of this run, it would seem that the orthogonal hyperplanes could
## further strengthened and that the one non-orthogonal hyperplane could be diminished.

demoToPerform = "simpleLearningWithFlocking"
# Choices are:  simpleLearningWithFlocking, learningWithFocusedFlocking, learningWithMomentumAndFocusedFlocking

require_relative  '../../lib/core/DataSet'
require_relative  '../../lib/core/NeuralParts'
require_relative  '../../lib/core/NeuralPartsExtended'
require_relative  '../../lib/core/NetworkFactories'
require_relative  '../../lib/plot/CorePlottingCode'

####################### TEMP Experiment  #########################      N
class NeuronRecorder
  private

  def convertEachHashToAVector(anArray)
    return anArrayOfVectors = anArray.collect do |measuresForAnExample|
      #Vector[(measuresForAnExample[:netInput]), (measuresForAnExample[:error])]
      Vector[(measuresForAnExample[:netInput])]
    end
  end
end
####################### TEMP Experiment  #########################      N

def createMultiClassTrainingSet(numberOfExamples, rightShiftUpper2Classes = 0.0)
  include ExampleDistribution

  xStart = [-1.0+rightShiftUpper2Classes, 1.0+rightShiftUpper2Classes, -1.0, 1.0]    # assumes clockwise numbering of classes, from 10:30 being class 0
  yStart = [1.0, 1.0, -1.0, -1.0]


  xInc = [0.0, 0.0, 0.0, 0.0]
  yInc = [1.0, 1.0, -1.0, -1.0]

  # target = [0.9, 0.9, 0.9, 0.9] # potentially will use this example set for supervised learning.

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
      y = yS + (yI * classExNumb)
      aPoint = [x, y]
      targets = [0.0, 0.0, 0.0, 0.0]
      targets[indexToClass] = 1.0
      examples << {:inputs => aPoint, :targets => targets, :exampleNumber => exampleNumber}
      exampleNumber += 1
    end
  end
  STDERR.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
  examples
end

def  createXORTrainingSet(numberOfExamples, rightShiftUpper2Classes)
  include ExampleDistribution

  xStart = [-1.0+rightShiftUpper2Classes, 1.0+rightShiftUpper2Classes, 1.0, -1.0]    # assumes clockwise numbering of classes, from 10:30 being class 0
  yStart = [1.0, 1.0, -1.0, -1.0]


  xInc = [0.0, 0.0, 0.0, 0.0]
  yInc = [1.0, 1.0, -1.0, -1.0]

  # target = [0.9, 0.9, 0.9, 0.9] # potentially will use this example set for supervised learning.

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
      y = yS + (yI * classExNumb)
      aPoint = [x, y]
      targetValues = [1.0, 0.0, 1.0, 0.0]
      targets = [0.0]
      targets[0] = targetValues[indexToClass]
      examples << {:inputs => aPoint, :targets => targets, :exampleNumber => exampleNumber}
      exampleNumber += 1
    end
  end
  STDERR.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
  examples
end


def create3inputMultiClassTrainingSet(numberOfExamples, rightShiftUpper2Classes = 0.0)
  include ExampleDistribution

  xStart = [-1.0+rightShiftUpper2Classes, 1.0+rightShiftUpper2Classes, -1.0, 1.0]    # assumes clockwise numbering of classes, from 10:30 being class 0
  yStart = [1.0, 1.0, -1.0, -1.0]
  zStart = [0.0, 1.0, 2.0, 3.0]

  xInc = [0.0, 0.0, 0.0, 0.0]
  yInc = [1.0, 1.0, -1.0, -1.0]

  # target = [0.9, 0.9, 0.9, 0.9] # potentially will use this example set for supervised learning.

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
      y = yS + (yI * classExNumb)
      z = (yInc[0] * classExNumb)
      aPoint = [x, y, z]
      targets = [0.0, 0.0, 0.0, 0.0]
      targets[indexToClass] = 1.0
      examples << {:inputs => aPoint, :targets => targets, :exampleNumber => exampleNumber}
      exampleNumber += 1
    end
  end
  STDERR.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
  examples
end


def reportMetrics(neuronToMonitor, epochNumber, aLearningNetwork, dataArray, args)
  oneEpochsMeasures = neuronToMonitor.metricRecorder.withinEpochMeasures
  dataArray << (oneEpochsMeasures).dup if ((epochNumber+1).modulo(1) == 0)
  mse = aLearningNetwork.calcNetworksMeanSquareError
  aLearningNetwork.recordResponse(mse, epochNumber)
  if (epochNumber.modulo(5) == 0)
    puts "At Epoch # #{epochNumber} Network's MSE=\t#{aLearningNetwork.calcNetworksMeanSquareError}\n"
    theFlockLearningRate = args[:flockLearningRate]
    #oneEpochsMeasures.each_with_index do |measuresForAnExample, exampleNumber|
    #  # std("measuresForAnExample",measuresForAnExample)
    #  puts "ex #{exampleNumber}\tBP Error=\t#{measuresForAnExample[:error]}\tFlocking Error=\t#{theFlockLearningRate * measuresForAnExample[:localFlockingError]}"
    #end
  end
  mse
end

def plotMSEvsEpochNumber(aLearningNetwork)
  mseVsEpochMeasurements = aLearningNetwork.measures
  x = mseVsEpochMeasurements.collect { |aMeasure| aMeasure[:epochNumber] }
  y = mseVsEpochMeasurements.collect { |aMeasure| aMeasure[:networkMSE] }
  aPlotter = Plotter.new(title="Simplest Flocking Training Error", "Number of Epochs", "Error on Training Set", aFilename = nil)
  aPlotter.plot(x, y)
end

def plotFlockingErrorVsEpochNumber(dataArray)
  arrayOfFlockingErrors = []
  arrayOfEpochNumbers = []
  dataArray.each_with_index do |oneEpochsMeasures, epochNumber|
    oneEpochsMeasures.each do |measuresForAnExample|
      arrayOfEpochNumbers << epochNumber.to_f
      arrayOfFlockingErrors << measuresForAnExample[:localFlockingError]
    end
  end

  aPlotter2 = Plotter.new(title="Simplest Flocking FlkError", "Number of Epochs", "Flocking Error", aFilename = nil)
  aPlotter2.plot(arrayOfEpochNumbers, arrayOfFlockingErrors)
end

def epochWithRecenteringOfClusters(epochsSinceBeginningOfCycle, aLearningNetwork, dataArray, allNeuronsInOneArray, epochNumber, flockingNeurons, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer, phase1Epochs, args)
  neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
  neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
  numberOfExamples.times do |exampleNumber|
    allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
    neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate { |bpError, flockError| bpError } }
    outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.recordExampleMeasurements }
    flockingNeurons.each { |aNeuron| aNeuron.calcLocalFlockingError { |dataForReport| periodicallyDisplayContentsOfHash(dataForReport, epochNumber, interval=phase1Epochs) } } if (epochNumber > 0)
    neuronsWithInputLinks.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate { |bpError, flockError| bpError + flockError } }
  end
  neuronsWithInputLinks.each { |aNeuron| aNeuron.addAccumulationToWeight }
  flockingNeurons.each { |aNeuron| aNeuron.initializeClusterCenters } if (epochNumber == 0)
  flockingNeurons.each { |aNeuron| aNeuron.clusterAllResponses }

  mse = reportMetrics(flockingNeurons[1], epochNumber, aLearningNetwork, dataArray, args)
  puts aLearningNetwork if ((epochNumber+1).modulo(phase1Epochs) == 0)
  epochNumber += 1
  epochsSinceBeginningOfCycle += 1
  return [mse, epochNumber, epochsSinceBeginningOfCycle]
end

def epochNoClusterRecentering(epochsSinceBeginningOfCycle, aLearningNetwork, dataArray, allNeuronsInOneArray, epochNumber, flockingNeurons, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer, phase1Epochs, args)
  neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
  neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
  numberOfExamples.times do |exampleNumber|
    allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
    neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate { |bpError, flockError| bpError } }
    outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.recordExampleMeasurements }
    flockingNeurons.each { |aNeuron| aNeuron.calcLocalFlockingError { |dataForReport| periodicallyDisplayContentsOfHash(dataForReport, epochNumber, interval=phase1Epochs) } } if (epochNumber > 0)
    neuronsWithInputLinks.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate { |bpError, flockError| bpError + flockError } }
  end
  neuronsWithInputLinks.each { |aNeuron| aNeuron.addAccumulationToWeight }

  mse = reportMetrics(outputLayer[0], epochNumber, aLearningNetwork, dataArray, args)
  puts aLearningNetwork if ((epochNumber+1).modulo(phase1Epochs) == 0)
  epochNumber += 1
  epochsSinceBeginningOfCycle += 1
  return [mse, epochNumber, epochsSinceBeginningOfCycle]
end

def epochWithRecenteringOfClustersUsingMomentum(epochsSinceBeginningOfCycle, aLearningNetwork, dataArray, allNeuronsInOneArray, epochNumber, flockingNeurons, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer, phase1Epochs, args)
  alpha = 0.5
  neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
  neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
  numberOfExamples.times do |exampleNumber|
    allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
    neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate { |bpError, flockError| bpError } }
    outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.recordExampleMeasurements }
    flockingNeurons.each { |aNeuron| aNeuron.calcLocalFlockingError { |dataForReport| periodicallyDisplayContentsOfHash(dataForReport, epochNumber, interval=phase1Epochs) } } if (epochNumber > 0)
    neuronsWithInputLinks.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate { |bpError, flockError|
              bpError + aNeuron.augmentFlockingErrorUsingMomentum(flockError, alpha, exampleNumber) } }
  end
  neuronsWithInputLinks.each { |aNeuron| aNeuron.addAccumulationToWeight }
  flockingNeurons.each { |aNeuron| aNeuron.initializeClusterCenters } if (epochNumber == 0)
  flockingNeurons.each { |aNeuron| aNeuron.clusterAllResponses }

  mse = reportMetrics(outputLayer[0], epochNumber, aLearningNetwork, dataArray, args)
  puts aLearningNetwork if ((epochNumber+1).modulo(phase1Epochs) == 0)
  epochNumber += 1
  epochsSinceBeginningOfCycle += 1
  return [mse, epochNumber, epochsSinceBeginningOfCycle]
end

def epochNoClusterRecenteringUsingMomentum(epochsSinceBeginningOfCycle, aLearningNetwork, dataArray, allNeuronsInOneArray, epochNumber, flockingNeurons, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer, phase1Epochs, args)
  alpha = 0.5
  neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
  neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
  numberOfExamples.times do |exampleNumber|
    allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
    neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate { |bpError, flockError| bpError } }
    outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.recordExampleMeasurements }
    flockingNeurons.each { |aNeuron| aNeuron.calcLocalFlockingError { |dataForReport| periodicallyDisplayContentsOfHash(dataForReport, epochNumber, interval=phase1Epochs) } } if (epochNumber > 0)
    neuronsWithInputLinks.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate  { |bpError, flockError|
              bpError + aNeuron.augmentFlockingErrorUsingMomentum(flockError, alpha, exampleNumber) } }
  end
  neuronsWithInputLinks.each { |aNeuron| aNeuron.addAccumulationToWeight }

  mse = reportMetrics(outputLayer[0], epochNumber, aLearningNetwork, dataArray, args)
  puts aLearningNetwork if ((epochNumber+1).modulo(phase1Epochs) == 0)
  epochNumber += 1
  epochsSinceBeginningOfCycle += 1
  return [mse, epochNumber, epochsSinceBeginningOfCycle]
end


def simpleLearningWithFlocking(minMSE, maxEpochNumber, phase1Epochs, phase2Epochs, aLearningNetwork,
    allNeuronsInOneArray, args, dataArray, epochNumber, flockingNeurons, mse,
    neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples,
    outputLayer)

  epochsSinceBeginningOfCycle = 0
  phase1PlusPhase2Epochs = phase1Epochs + phase2Epochs
  while (mse > minMSE && epochNumber < maxEpochNumber)

    if (phase1Epochs > 0)
      flockingNeurons.each { |aNeuron| aNeuron.bpLearningRate = 0.0 } if (epochsSinceBeginningOfCycle >= phase1Epochs)
      flockingNeurons.each { |aNeuron| aNeuron.bpLearningRate = args[:bpLearningRate] } if (epochsSinceBeginningOfCycle >= phase1PlusPhase2Epochs)
      epochsSinceBeginningOfCycle = 0 if (epochsSinceBeginningOfCycle >= phase1PlusPhase2Epochs)
      puts aLearningNetwork if (epochNumber == phase1Epochs)
    end

    mse, epochNumber, epochsSinceBeginningOfCycle = epochWithRecenteringOfClusters(epochsSinceBeginningOfCycle, aLearningNetwork, dataArray, allNeuronsInOneArray, epochNumber, flockingNeurons, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer, phase1Epochs, args)
  end
  return epochNumber, mse
end

def learningWithFocusedFlocking(phase1Epochs, phase2Epochs, aLearningNetwork,
    allNeuronsInOneArray, args, dataArray, epochNumber, epochsOfFlockingOnly,
    flockingNeurons, mse, neuronsWithInputLinks,
    neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer)

  epochsSinceBeginningOfCycle = 0
  phase1PlusPhase2Epochs = phase1Epochs + phase2Epochs
  phase1 = false
  while (mse > 0.001 && epochNumber<3.0*10**3)

    if (phase1Epochs > 0)
      if (epochsSinceBeginningOfCycle >= phase1Epochs)
        puts aLearningNetwork if(phase1)
        phase1 = false
        flockingNeurons.each { |aNeuron| aNeuron.bpLearningRate = 0.0 }   #TODO WARNING: BP LEARNING Will Still Occur with regular neurons!!
      end
      if (epochsSinceBeginningOfCycle >= phase1PlusPhase2Epochs)
        puts aLearningNetwork unless(phase1)
        phase1 = true
        epochsSinceBeginningOfCycle = 0
        flockingNeurons.each { |aNeuron| aNeuron.bpLearningRate = args[:bpLearningRate] }
      end
    end

    if (phase1)
      mse, epochNumber, epochsSinceBeginningOfCycle = epochWithRecenteringOfClusters(epochsSinceBeginningOfCycle, aLearningNetwork, dataArray, allNeuronsInOneArray, epochNumber, flockingNeurons, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer, phase1Epochs, args)

      flockingNeurons.each { |aNeuron| aNeuron.bpLearningRate = 0.0 }   #TODO WARNING: BP LEARNING Will Still Occur with regular neurons!!
      epochsOfFlockingOnly.times do
        mse, epochNumber, epochsSinceBeginningOfCycle = epochNoClusterRecentering(epochsSinceBeginningOfCycle, aLearningNetwork, dataArray, allNeuronsInOneArray, epochNumber, flockingNeurons, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer, phase1Epochs, args)
      end
      flockingNeurons.each { |aNeuron| aNeuron.bpLearningRate = args[:bpLearningRate] }

    else #i.e., phase2
      mse, epochNumber, epochsSinceBeginningOfCycle = epochWithRecenteringOfClusters(epochsSinceBeginningOfCycle, aLearningNetwork, dataArray, allNeuronsInOneArray, epochNumber, flockingNeurons, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer, phase1Epochs, args)
    end
  end
  return epochNumber, mse
end

def learningWithMomentumAndFocusedFlocking(phase1Epochs, phase2Epochs, aLearningNetwork,
    allNeuronsInOneArray, args, dataArray, epochNumber, epochsOfFlockingOnly,
    flockingNeurons, mse, neuronsWithInputLinks,
    neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer)

  epochsSinceBeginningOfCycle = 0
  phase1PlusPhase2Epochs = phase1Epochs + phase2Epochs
  phase1 = false
  while (mse > 0.001 && epochNumber<3.0*10**3)

    if (phase1Epochs > 0)
      if (epochsSinceBeginningOfCycle >= phase1Epochs)
        puts aLearningNetwork if(phase1)
        phase1 = false
        flockingNeurons.each { |aNeuron| aNeuron.bpLearningRate = 0.0 }   #TODO WARNING: BP LEARNING Will Still Occur with regular neurons!!
      end
      if (epochsSinceBeginningOfCycle >= phase1PlusPhase2Epochs)
        puts aLearningNetwork unless(phase1)
        phase1 = true
        epochsSinceBeginningOfCycle = 0
        flockingNeurons.each { |aNeuron| aNeuron.bpLearningRate = args[:bpLearningRate] }
      end
    end

    if (phase1)
      mse, epochNumber, epochsSinceBeginningOfCycle = epochWithRecenteringOfClusters(epochsSinceBeginningOfCycle, aLearningNetwork, dataArray, allNeuronsInOneArray, epochNumber, flockingNeurons, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer, phase1Epochs, args)

      flockingNeurons.each { |aNeuron| aNeuron.bpLearningRate = 0.0 }   #TODO WARNING: BP LEARNING Will Still Occur with regular neurons!!
      epochsOfFlockingOnly.times do
        mse, epochNumber, epochsSinceBeginningOfCycle = epochNoClusterRecenteringUsingMomentum(epochsSinceBeginningOfCycle, aLearningNetwork, dataArray, allNeuronsInOneArray, epochNumber, flockingNeurons, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer, phase1Epochs, args)
      end
      flockingNeurons.each { |aNeuron| aNeuron.bpLearningRate = args[:bpLearningRate] }

    else #i.e., phase2
      mse, epochNumber, epochsSinceBeginningOfCycle = epochWithRecenteringOfClustersUsingMomentum(epochsSinceBeginningOfCycle, aLearningNetwork, dataArray, allNeuronsInOneArray, epochNumber, flockingNeurons, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer, phase1Epochs, args)
    end
  end
  return epochNumber, mse
end

####################################################################
srand(0)
numberOfExamples = 4
rightShiftUpper2Classes = 0.0
# examples = create3inputMultiClassTrainingSet(numberOfExamples, rightShiftUpper2Classes)
examples = createXORTrainingSet(numberOfExamples, rightShiftUpper2Classes)
puts examples

phase1Epochs = 500
phase2Epochs = 200
epochsOfFlockingOnly = 5
rightShiftUpper2Classes = 0.0
minMSE = 0.001
maxEpochNumber = 2.0*10**3

args = {:learningRate => 1.0,
        :bpLearningRate => 3.0, # 0.1,
        :weightRange => 1.0, #0.01,
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 3,
        :numberOfOutputNeurons => 1,
        :numberOfExamples => numberOfExamples,

        # Parameters associated with flocking...
        :typeOfClusterer => DynamicClusterer,
        :numberOfClusters => 2,
        :m => 1.1, # 2.0,               # 2.0
        :numExamples => numberOfExamples,
        :exampleVectorLength => 1,  # 2  #This interacts with NeuronRecorder.vectorizeEpochMeasures(anArray)
        :delta => 0.001,
        :maxNumberOfClusteringIterations => 100,
        :flockLearningRate => -0.01 # -0.001 #-0.01 seems to be good with or without momentum (using alpha=0.5)
}

# Specify network
aLearningNetwork = AllFlockingNeuronNetwork.new(args)
allNeuronLayers = aLearningNetwork.createSimpleLearningANN

allNeuronsInOneArray = allNeuronLayers.flatten
inputLayer = allNeuronLayers[0]
hiddenLayer = allNeuronLayers[1]
outputLayer = allNeuronLayers[2]
neuronsWithInputLinks = hiddenLayer + outputLayer
neuronsWithInputLinksInReverseOrder = neuronsWithInputLinks.reverse
flockingNeurons = hiddenLayer

############################### include these 2-D training examples...
distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])

std("Number of Examples", numberOfExamples)

################################

dataArray = []
mse = 99999.0
epochNumber = 0

case demoToPerform
  when "simpleLearningWithFlocking"
    epochNumber, mse = simpleLearningWithFlocking(minMSE, maxEpochNumber, phase1Epochs, phase2Epochs, aLearningNetwork, allNeuronsInOneArray, args, dataArray, epochNumber, flockingNeurons, mse, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer)
  when "learningWithFocusedFlocking"
    epochNumber, mse = learningWithFocusedFlocking(phase1Epochs, phase2Epochs, aLearningNetwork, allNeuronsInOneArray, args, dataArray, epochNumber, epochsOfFlockingOnly, flockingNeurons, mse, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer)
  when "learningWithMomentumAndFocusedFlocking"
    epochNumber, mse = learningWithMomentumAndFocusedFlocking(phase1Epochs, phase2Epochs, aLearningNetwork, allNeuronsInOneArray, args, dataArray, epochNumber, epochsOfFlockingOnly, flockingNeurons, mse, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer)
  else
    STDERR.puts "did not understand CHOICE!"
end

plotMSEvsEpochNumber(aLearningNetwork)

plotFlockingErrorVsEpochNumber(dataArray)

puts "At Epoch # #{epochNumber} Network's MSE=\t#{mse}\n\n"

puts aLearningNetwork # display neural network's final state -- after training is complete.
