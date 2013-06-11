### VERSION "nCore"
## ../nCore/bin/SinglePhase1Phase2Cycle.rb    REALLY MULTI-CYCLE VERSION!!  it just evolved into a multi-cycle system
# Purpose:  To quantitatively explore the simplest clustering w/o supervision.

demoToPerform = "learningWithMomentumAndFocusedFlocking"
# Choices are:  simpleLearningWithFlocking, learningWithFocusedFlocking, learningWithMomentumAndFocusedFlocking

require 'C:/Code/DevRuby/NN2012/nCore/lib/core/DataSet'
require 'C:/Code/DevRuby/NN2012/nCore/lib/core/NeuralParts'
require 'C:/Code/DevRuby/NN2012/nCore/lib/core/NeuralPartsExtended'
require 'C:/Code/DevRuby/NN2012/nCore/lib/core/NetworkFactories'
require 'C:/Code/DevRuby/NN2012/nCore/lib/plot/CorePlottingCode'

# THIS ROUTINE IS FOR MORE COMPLEX DATASETS --> 2 input dimensions for training set; binary target
def createTrainingSetWith2InputDimensions2Classes(numberOfExamples, rightShiftUpper2Classes = 0.0)
  include ExampleDistribution

  xStart = [-1.0, 1.0]
  yStart = [1.0, -1.0]

  xInc = [0.0+rightShiftUpper2Classes, 0.0-rightShiftUpper2Classes]
  yInc = [1.0, -1.0]

  numberOfClasses = xStart.length
  numberOfExamplesInEachClass = numberOfExamples / numberOfClasses
  exampleNumber = 0
  xIncrement = 0.0
  yIncrement = 0.0

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
      targets = [0.0, 0.0]
      targets[indexToClass] = 1.0
      examples << {:inputs => aPoint, :targets => targets, :exampleNumber => exampleNumber}
      exampleNumber += 1
    end
  end
  STDERR.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
  examples
end

def reportMetrics(outputNeuron, epochNumber, aLearningNetwork, dataArray, args)
  oneEpochsMeasures = outputNeuron.metricRecorder.withinEpochMeasures
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

  mse = reportMetrics(outputLayer[0], epochNumber, aLearningNetwork, dataArray, args)
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
    neuronsWithInputLinks.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate { |bpError, flockError|
      bpError + aNeuron.augmentFlockingErrorUsingMomentum(flockError, alpha, exampleNumber) } }
  end
  neuronsWithInputLinks.each { |aNeuron| aNeuron.addAccumulationToWeight }

  mse = reportMetrics(outputLayer[0], epochNumber, aLearningNetwork, dataArray, args)
  puts aLearningNetwork if ((epochNumber+1).modulo(phase1Epochs) == 0)
  epochNumber += 1
  epochsSinceBeginningOfCycle += 1
  return [mse, epochNumber, epochsSinceBeginningOfCycle]
end


def simpleLearningWithFlocking(phase1Epochs, phase2Epochs, aLearningNetwork,
    allNeuronsInOneArray, args, dataArray, epochNumber, flockingNeurons, mse,
    neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples,
    outputLayer)

  epochsSinceBeginningOfCycle = 0
  phase1PlusPhase2Epochs = phase1Epochs + phase2Epochs
  while (mse > 0.001 && epochNumber<3.0*10**3)

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
        puts aLearningNetwork if (phase1)
        phase1 = false
        flockingNeurons.each { |aNeuron| aNeuron.bpLearningRate = 0.0 } #TODO WARNING: BP LEARNING Will Still Occur with regular neurons!!
      end
      if (epochsSinceBeginningOfCycle >= phase1PlusPhase2Epochs)
        puts aLearningNetwork unless (phase1)
        phase1 = true
        epochsSinceBeginningOfCycle = 0
        flockingNeurons.each { |aNeuron| aNeuron.bpLearningRate = args[:bpLearningRate] }
      end
    end

    if (phase1)
      mse, epochNumber, epochsSinceBeginningOfCycle = epochWithRecenteringOfClusters(epochsSinceBeginningOfCycle, aLearningNetwork, dataArray, allNeuronsInOneArray, epochNumber, flockingNeurons, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer, phase1Epochs, args)

      flockingNeurons.each { |aNeuron| aNeuron.bpLearningRate = 0.0 } #TODO WARNING: BP LEARNING Will Still Occur with regular neurons!!
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
        puts aLearningNetwork if (phase1)
        phase1 = false
        flockingNeurons.each { |aNeuron| aNeuron.bpLearningRate = 0.0 } #TODO WARNING: BP LEARNING Will Still Occur with regular neurons!!
      end
      if (epochsSinceBeginningOfCycle >= phase1PlusPhase2Epochs)
        puts aLearningNetwork unless (phase1)
        phase1 = true
        epochsSinceBeginningOfCycle = 0
        flockingNeurons.each { |aNeuron| aNeuron.bpLearningRate = args[:bpLearningRate] }
      end
    end

    if (phase1)
      mse, epochNumber, epochsSinceBeginningOfCycle = epochWithRecenteringOfClusters(epochsSinceBeginningOfCycle, aLearningNetwork, dataArray, allNeuronsInOneArray, epochNumber, flockingNeurons, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer, phase1Epochs, args)

      flockingNeurons.each { |aNeuron| aNeuron.bpLearningRate = 0.0 } #TODO WARNING: BP LEARNING Will Still Occur with regular neurons!!
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
numberOfExamples = 8

args = {:learningRate => 1.0,
        :bpLearningRate => 0.1,
        :weightRange => 0.01,
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 0,
        :numberOfOutputNeurons => 1,
        :numberOfExamples => numberOfExamples,

        # Parameters associated with flocking...
        :typeOfClusterer => DynamicClusterer,
        :numberOfClusters => 2,
        :m => 2.0,
        :numExamples => numberOfExamples,
        :exampleVectorLength => 2,
        :delta => 0.001,
        :maxNumberOfClusteringIterations => 100,
        :flockLearningRate => -0.01 # -0.01 seems to be good with or without momentum (using alpha=0.5)
}

# Specify network
aLearningNetwork = AllFlockingNeuronNetwork.new(args)
allNeuronLayers = aLearningNetwork.createSimpleLearningANN

allNeuronsInOneArray = allNeuronLayers.flatten
inputLayer = allNeuronLayers[0]
outputLayer = allNeuronLayers[1]
neuronsWithInputLinks = outputLayer
neuronsWithInputLinksInReverseOrder = neuronsWithInputLinks.reverse
flockingNeurons = outputLayer

############################### include these 2-D training examples...
separationBetweenDataPoints = 0.0
examples = createTrainingSetWith2InputDimensions2Classes(numberOfExamples, separationBetweenDataPoints)
distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])
################################

dataArray = []
mse = 99999.0
epochNumber = 0
phase1Epochs = 200
phase2Epochs = 200
epochsOfFlockingOnly = 5

case demoToPerform
  when "simpleLearningWithFlocking"
    epochNumber, mse = simpleLearningWithFlocking(phase1Epochs, phase2Epochs, aLearningNetwork, allNeuronsInOneArray, args, dataArray, epochNumber, flockingNeurons, mse, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer)
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
