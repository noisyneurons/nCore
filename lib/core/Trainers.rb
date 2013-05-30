### VERSION "nCore"
## ../nCore/lib/core/Trainers.rb

require_relative 'Utilities'


class AbstractTrainer
  attr_accessor :trainingSequence, :minMSE, :maxNumEpochs, :examples, :dataArray, :args,
                :numberOfExamples, :dataStoreManager,
                :lastEpoch, :lastTrainingMSE, :lastTestingMSE, :dPrimes, :startTime, :elapsedTime,

                :network, :allNeuronLayers,
                :allNeuronsInOneArray, :inputLayer, :hiddenLayer, :outputLayer, :theBiasNeuron,
                :hiddenLayer1, :hiddenLayer2, :hiddenLayer3, :allHiddenLayers,
                :neuronsWithInputLinks, :neuronsWithInputLinksInReverseOrder,


                :neuronsAdaptingOnlyToBPOutputError, :neuronsToAdaptToOutputError,
                :neuronsCreatingFlockingError, :neuronsAdaptingToLocalFlockingError,
                :neuronsAdaptingToBackPropedFlockingError, :neuronsWhoseClustersNeedToBeSeeded,

                :learningRateNoFlockPhase1, :learningRateLocalFlockPhase2, :learningRateBPOutputErrorPhase2,
                :learningRateNoFlockPhase1, :learningRateForBackPropedFlockingErrorPhase2, :learningRateFlockPhase2,

                :learningRateTuner

  include ExampleDistribution
  include NeuronToNeuronConnection

  def initialize(trainingSequence, network, args)
    @trainingSequence = trainingSequence
    @network = network
    @args = args
    @dataStoreManager = SimulationDataStoreManager.instance
    @allNeuronLayers = network.createSimpleLearningANN
    @startTime = Time.now
    @elapsedTime = nil
    @minMSE = args[:minMSE]
    postInitialize()
  end

  def postInitialize
    STDERR.puts "Error: The AbstractTrainer class's postInitialize method should NOT not be called."
  end

  def simpleLearningWithFlocking(examples, arrayOfNeuronsToPlot)
    step1NameTrainingGroupsAndLearningRates
    mse, dPrimes = oneStepOfLearningAndDisplay(examples, arrayOfNeuronsToPlot)
    return trainingSequence.epochs, mse, dPrimes
  end

  def oneForwardPassEpoch(testingExamples)
    trainingExamples = examples
    distributeSetOfExamples(testingExamples)
    testMSE = measureNeuralResponsesForTesting
    distributeSetOfExamples(trainingExamples) # restore training examples
    return testMSE
  end

  def getZeroXingExampleSet(theNeuron)
    trainingExamples = examples
    minX1 = -4.0
    maxX1 = 4.0
    numberOfTrials = 100
    increment = (maxX1 - minX1) / numberOfTrials
    x0Array = []
    x1Array = []
    numberOfTrials.times do |index|
      x1 = minX1 + index * increment
      x0 = findZeroCrossingFast(x1, theNeuron)
      next if (x0.nil?)
      x1Array << x1
      x0Array << x0
    end
    self.examples = trainingExamples
    [x0Array, x1Array]
  end

  def storeEndOfTrainingMeasures(lastEpoch, lastTrainingMSE, lastTestingMSE, dPrimes)
    self.elapsedTime = Time.now - startTime
    dataStoreManager.storeEndOfTrainingMeasures(lastEpoch, lastTrainingMSE, lastTestingMSE, dPrimes, elapsedTime)
  end

  protected

  def step1NameTrainingGroupsAndLearningRates
    STDERR.puts "Error: The AbstractTrainer class's step1NameTrainingGroupsAndLearningRates method should NOT not be called."
  end

  def setUniversalNeuronGroupNames
    self.neuronsWithInputLinksInReverseOrder = neuronsWithInputLinks.reverse
    self.allNeuronsInOneArray = inputLayer + neuronsWithInputLinks
  end

  def oneStepOfLearningAndDisplay(examples, arrayOfNeuronsToPlot)
    mse, dPrimes = stepLearning(examples)
    puts network
    if arrayOfNeuronsToPlot.present?
      arrayOfNeuronsToPlot.each do |theNeuron|
        xAry, yAry = getZeroXingExampleSet(theNeuron)
        plotDotsWhereOutputGtPt5(xAry, yAry, theNeuron, trainingSequence.epochs)
      end
    end
    [mse, dPrimes]
  end

  def stepLearning(examples)
    distributeSetOfExamples(examples)
    mse = 99999.0
    dPrimes = nil
    @dPrimesOld = nil

    seedClustersInFlockingNeurons(neuronsWhoseClustersNeedToBeSeeded) # TODO How often and where do we need to do this?
    while ((mse > minMSE) && trainingSequence.stillMoreEpochs)
      case trainingSequence.status
        when :inPhase1
          neuronsWithInputLinks.each { |neuron| neuron.learningRate = learningRateNoFlockPhase1 }

          mse = noFlockingAdaptNetwork

        when :inPhase2
          @dPrimesOld = nil if (trainingSequence.atStartOfPhase2)

          (neuronsAdaptingToLocalFlockingError + neuronsAdaptingToBackPropedFlockingError).each { |neuron| neuron.learningRate = learningRateFlockPhase2 }

          # TODO NEED TO RESOLVE THIS... QUESTION BELOW
          ###------------   Adaption to FLOCKING Error  ------------------------------
          dPrimes = recenterEachNeuronsClusters(neuronsCreatingFlockingError) ## TODO WOW, why does this statement "cause" such a drop in dPrimes in 2nd hidden layer in Analogy, but not Simplest main??
          mse, dPrimes = flockingOnlyAdaptNetwork(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)

          deltaDPrimeRatios = deltaDPrimes(dPrimes)
          puts "epochs=\t#{trainingSequence.epochs}\tdeltaDPrimeRatios=\t#{deltaDPrimeRatios}\tmse, dPrimes= \t #{mse}\t#{dPrimes}"
        # trainingSequence.dPrimesHaveNotChangedMuch if (deltaDPrimeRatios.max < 0.01)
        else
          STDERR.puts "status of training sequencer is not understood"
      end
      trainingSequence.nextEpoch
    end
    return mse, dPrimes
  end


  ###------------   Adaption to OUTPUT Error  ------------------------------
  def noFlockingAdaptNetwork
    noFlockingMeasureAndStoreAllNeuralResponses
    neuronsToAdaptToOutputError.each { |aNeuron| aNeuron.addAccumulationToWeight }
    mse = logNetworksResponses(outputLayer)
  end

  def noFlockingMeasureAndStoreAllNeuralResponses
    neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate }
      outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
      neuronsToAdaptToOutputError.each { |aNeuron| aNeuron.recordResponsesForExampleToDB(aNeuron.recordResponsesForExample) }
      neuronsToAdaptToOutputError.each { |neuron| neuron.calcDeltaWsAndAccumulate }
    end
  end

  def noFlockingMeasureAndStoreAllNeuralResponsesNoDB
    neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate }
      outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
      neuronsToAdaptToOutputError.each { |aNeuron| aNeuron.recordResponsesForExample }
      neuronsToAdaptToOutputError.each { |neuron| neuron.calcDeltaWsAndAccumulate }
    end
  end


  ###------------   Adaption to FLOCKING Error  ------------------------------
  def flockingOnlyAdaptNetwork(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
    flockingOnlyMeasureAndStoreAllNeuralResponses(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
    (neuronsAdaptingToLocalFlockingError + neuronsAdaptingToBackPropedFlockingError).each { |aNeuron| aNeuron.addAccumulationToWeight }
    dPrimes = recenterEachNeuronsClusters(neuronsCreatingFlockingError)
    mse = logNetworksResponses(neuronsCreatingFlockingError)
    [mse, dPrimes]
  end

  def flockingOnlyMeasureAndStoreAllNeuralResponses(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
    neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated } # TODO some redundancy here -- particularly when placed immediately after "to prior output error adapting code
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures } # TODO some redundancy here --particularly when placed immediately after "to prior output error adapting code

    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate }
      outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
      (outputLayer - neuronsCreatingFlockingError).each { |aNeuron| aNeuron.recordResponsesForExample }

      neuronsCreatingFlockingError.each do |aNeuron|
        dataRecorded = aNeuron.recordResponsesForExample
        localFlockingError = aNeuron.calcLocalFlockingError
        dataRecorded[:localFlockingError] = localFlockingError
        aNeuron.recordResponsesForExampleToDB(dataRecorded)
      end

      unless (neuronsAdaptingToLocalFlockingError.empty?)
        neuronsAdaptingToLocalFlockingError.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate { |errorFromUpperLayers, localFlockError| localFlockError } }
      else
        neuronsCreatingFlockingError.each { |aNeuron| aNeuron.backPropagate { |errorFromUpperLayers, localFlockError| localFlockError } }
        neuronsAdaptingToBackPropedFlockingError.each { |aNeuron| aNeuron.backPropagate } # TODO So far, we are temporarily assuming that bp-flocking error only has to be bp to immediately preceding layer!!
        neuronsAdaptingToBackPropedFlockingError.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate }
      end
    end
  end

  ###------------ -----------------------------

  def measureNeuralResponsesForTesting
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
      neuronsWithInputLinks.each { |aNeuron| aNeuron.recordResponsesForExample }
    end
    mse = network.calcNetworksMeanSquareError
  end

  def findZeroCrossingFast(x1, theNeuron)
    minX0 = -2.0
    maxX0 = 2.0
    range = maxX0 - minX0

    initialOutput = ioFunctionFor2inputs(minX0, x1, theNeuron)
    initialOutputLarge = false
    initialOutputLarge = true if (initialOutput >= 0.5)

    wasThereACrossing = false
    20.times do |count|
      range = maxX0 - minX0
      bisectedX0 = minX0 + (range / 2.0)
      currentOutput = ioFunctionFor2inputs(bisectedX0, x1, theNeuron)
      currentOutputLarge = false
      currentOutputLarge = true if (currentOutput > 0.5)
      if (currentOutputLarge != initialOutputLarge)
        maxX0 = bisectedX0
        wasThereACrossing = true
      else
        minX0 = bisectedX0
      end
    end
    return nil if (wasThereACrossing == false)
    estimatedCrossingForX0 = (minX0 + maxX0) / 2.0
    return estimatedCrossingForX0
  end

  def ioFunctionFor2inputs(x0, x1, theNeuron)
    examples = [{:inputs => [x0, x1], :exampleNumber => 0}]
    distributeDataToAnArrayOfNeurons(inputLayer, examples)
    allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(0) }
    return theNeuron.output
  end

###------------  Core Support Section ------------------------------

  def distributeSetOfExamples(examples)
    @examples = examples
    @numberOfExamples = examples.length
    distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])
  end

  def recenterEachNeuronsClusters(neuronsCreatingFlockingError)
    dPrimes = neuronsCreatingFlockingError.collect { |aNeuron| aNeuron.clusterAllResponses } # TODO Perhaps we might only need to clusterAllResponses every K epochs?
  end

  def seedClustersInFlockingNeurons(neuronsCreatingFlockingError)
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate }
      neuronsWithInputLinks.each { |aNeuron| aNeuron.recordResponsesForExample }
    end
    neuronsCreatingFlockingError.each { |aNeuron| aNeuron.initializeClusterCenters } # TODO is there any case where system should be reinitialized in this manner?
    recenterEachNeuronsClusters(neuronsCreatingFlockingError)
  end

  def deltaDPrimes(dPrimes)
    deltaDPrimes = nil
    unless (@dPrimesOld.nil?)
      index = -1
      deltaDPrimeRatios = dPrimes.collect do |dPrime|
        index += 1
        deltaDPrimeRatio = (dPrime - @dPrimesOld[index]) / ((dPrime + @dPrimesOld[index])/2.0)
      end
    else
      deltaDPrimeRatios = Array.new(dPrimes.length) { 1.0 }
    end
    # puts "epochs=\t#{trainingSequence.epochs}\tdPrimes=\t#{dPrimes}\tOldDPrimes=\t#{@dPrimesOld}" if (trainingSequence.epochs > 30)
    @dPrimesOld = dPrimes
    return deltaDPrimeRatios
  end

  def logNetworksResponses(neuronsCreatingFlockingError)
    neuronsCreatingFlockingError.each { |aNeuron| aNeuron.recordResponsesForEpoch } if (neuronsCreatingFlockingError.present?)
    mse = network.calcNetworksMeanSquareError
    network.recordResponses
    mse
  end

  #def logNetworksResponsesNoDB(neuronsCreatingFlockingError)
  #   neuronsCreatingFlockingError.each { |aNeuron| aNeuron.recordResponsesForEpoch } if (neuronsCreatingFlockingError.present?)
  #   mse = network.calcNetworksMeanSquareError
  #   network.recordResponses
  #   mse
  # end
end

class SimplestNetworkTrainer < AbstractTrainer # Used for main: "SimplestFlockingDemo2.rb"

  def postInitialize
    @allNeuronsInOneArray = network.allNeuronsInOneArray
    @inputLayer = network.inputLayer
    @outputLayer = network.outputLayer
    @theBiasNeuron = network.theBiasNeuron
  end

  def hiddenLayer
    STDERR.puts "ERROR: Access to hiddenLayer has been attempted"
  end

  def hiddenLayer1
    STDERR.puts "ERROR: Access to hiddenLayer1 has been attempted"
  end

  protected

  def step1NameTrainingGroupsAndLearningRates
    # PHASE 1   -----   Adaption to OUTPUT Error  ------
    self.neuronsWithInputLinks = outputLayer
    setUniversalNeuronGroupNames
    self.neuronsToAdaptToOutputError = outputLayer
    self.learningRateNoFlockPhase1 = args[:learningRateNoFlockPhase1]

    # PHASE 2 -----   Adaption to FLOCKING Error  -------
    self.neuronsCreatingFlockingError = outputLayer
    self.neuronsAdaptingToLocalFlockingError = outputLayer
    self.neuronsAdaptingToBackPropedFlockingError = []
    self.neuronsWhoseClustersNeedToBeSeeded = neuronsCreatingFlockingError
    self.learningRateFlockPhase2 = args[:learningRateLocalFlockPhase2]
  end
end

class SimpleAdjustableLearningRateTrainer < SimplestNetworkTrainer

  def postInitialize
    @allNeuronsInOneArray = network.allNeuronsInOneArray
    @inputLayer = network.inputLayer
    @outputLayer = network.outputLayer
    @theBiasNeuron = network.theBiasNeuron
    @learningRateTuner = LearningRateTuner.new(outputLayer)
  end

  ###------------   Adaption to OUTPUT Error  ------------------------------
  def noFlockingAdaptNetwork
    noFlockingMeasureAndStoreAllNeuralResponses
    neuronsToAdaptToOutputError.each { |aNeuron| aNeuron.saveDeltaWAccumulated }
    #neuronsToAdaptToOutputError.each { |aNeuron| aNeuron.addAccumulationToWeight }
    mse = logNetworksResponses(outputLayer)
    [mse, dPrimes]
  end

  def flockingOnlyAdaptNetwork(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
    flockingOnlyMeasureAndStoreAllNeuralResponses(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
    learningRateTuner.printInfo
    learningRateTuner.tune

    (neuronsAdaptingToLocalFlockingError + neuronsAdaptingToBackPropedFlockingError).each { |aNeuron| aNeuron.addAccumulationToWeight }
    dPrimes = recenterEachNeuronsClusters(neuronsCreatingFlockingError)
    mse = logNetworksResponses(neuronsCreatingFlockingError)
    [mse, dPrimes]
  end
end

class SimpleAdjustableLearningRateTrainerMultiFlockIterations < SimplestNetworkTrainer

  def postInitialize
    @allNeuronsInOneArray = network.allNeuronsInOneArray
    @inputLayer = network.inputLayer
    @outputLayer = network.outputLayer
    @theBiasNeuron = network.theBiasNeuron
    @learningRateTuner = LearningAndFlockingTuner.new(outputLayer, args)
  end

  def noFlockingAdaptNetworkTuned
    noFlockingMeasureAndStoreAllNeuralResponsesNoDB
    neuronsToAdaptToOutputError.each { |aNeuron| aNeuron.saveDeltaWAccumulated }
    mse = logNetworksResponses(outputLayer)
  end

  def flockingOnlyAdaptNetworkTuned(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError,
      neuronsAdaptingToBackPropedFlockingError)
    flockingOnlyMeasureAndStoreAllNeuralResponses(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError,
                                                  neuronsAdaptingToBackPropedFlockingError)
    learningRateTuner.printInfo
    learningRateTuner.tune

    (neuronsAdaptingToLocalFlockingError + neuronsAdaptingToBackPropedFlockingError).each { |aNeuron| aNeuron.addAccumulationToWeight }
    dPrimes = recenterEachNeuronsClusters(neuronsCreatingFlockingError)
    mse = logNetworksResponses(neuronsCreatingFlockingError)
    [mse, dPrimes]
  end

  def flockingOnlyAdaptNetworkFixedFlockingGain(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError,
      neuronsAdaptingToBackPropedFlockingError)
    flockingOnlyMeasureAndStoreAllNeuralResponsesFixedFlockingGain(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError,
                                                                   neuronsAdaptingToBackPropedFlockingError)

    (neuronsAdaptingToLocalFlockingError + neuronsAdaptingToBackPropedFlockingError).each { |aNeuron| aNeuron.addAccumulationToWeight }
    dPrimes = recenterEachNeuronsClusters(neuronsCreatingFlockingError)
    mse = logNetworksResponses(neuronsCreatingFlockingError)
    [mse, dPrimes]
  end

  def flockingOnlyMeasureAndStoreAllNeuralResponsesFixedFlockingGain(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
    neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated } # TODO some redundancy here -- particularly when placed immediately after "to prior output error adapting code
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures } # TODO some redundancy here --particularly when placed immediately after "to prior output error adapting code

    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate }
      outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
      (outputLayer - neuronsCreatingFlockingError).each { |aNeuron| aNeuron.recordResponsesForExample }

      neuronsCreatingFlockingError.each do |aNeuron|
        dataRecorded = aNeuron.recordResponsesForExample
        localFlockingError = aNeuron.calcLocalFlockingError
        dataRecorded[:localFlockingError] = localFlockingError
        aNeuron.recordResponsesForExampleToDB(dataRecorded)
      end

      unless (neuronsAdaptingToLocalFlockingError.empty?)
        neuronsAdaptingToLocalFlockingError.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate { |errorFromUpperLayers, localFlockError| aNeuron.layerGain * aNeuron.flockingGain * localFlockError } }
      else
        neuronsCreatingFlockingError.each { |aNeuron| aNeuron.backPropagate { |errorFromUpperLayers, localFlockError| localFlockError } }
        neuronsAdaptingToBackPropedFlockingError.each { |aNeuron| aNeuron.backPropagate } # TODO So far, we are temporarily assuming that bp-flocking error only has to be bp to immediately preceding layer!!
        neuronsAdaptingToBackPropedFlockingError.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate }
      end
    end
  end


  #def flockingOnlyMeasureAndStoreAllNeuralResponses(neuronsCreatingFlockingError,
  #    neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
  #  neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }     # TODO some redundancy here -- to prior output error adapting code
  #  neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }   # TODO some redundancy here -- to prior output error adapting code
  #
  #  numberOfExamples.times do |exampleNumber|
  #    allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
  #    neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate }
  #    outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
  #    (outputLayer - neuronsCreatingFlockingError).each { |aNeuron| aNeuron.recordResponsesForExample }
  #
  #    neuronsCreatingFlockingError.each do |aNeuron|
  #      dataRecorded = aNeuron.recordResponsesForExample
  #      localFlockingError = aNeuron.calcLocalFlockingError
  #      dataRecorded[:localFlockingError] = localFlockingError
  #      aNeuron.recordResponsesForExampleToDB(dataRecorded)
  #    end
  #
  #    unless (neuronsAdaptingToLocalFlockingError.empty?)
  #      neuronsAdaptingToLocalFlockingError.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate { |errorFromUpperLayers, localFlockError| localFlockError } }
  #    else
  #      neuronsCreatingFlockingError.each { |aNeuron| aNeuron.backPropagate { |errorFromUpperLayers, localFlockError| localFlockError } }
  #      neuronsAdaptingToBackPropedFlockingError.each { |aNeuron| aNeuron.backPropagate } # TODO So far, we are temporarily assuming that bp-flocking error only has to be bp to immediately preceding layer!!
  #      neuronsAdaptingToBackPropedFlockingError.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate }
  #    end
  #  end
  #end

  def stepLearning(examples)
    distributeSetOfExamples(examples)
    mse = 99999.0
    dPrimes = nil
    @dPrimesOld = nil

    neuronsWithInputLinks.each { |neuron| neuron.learningRate = 1.0 }
    seedClustersInFlockingNeurons(neuronsWhoseClustersNeedToBeSeeded) # TODO How often and where do we need to do this?
    while ((mse > minMSE) && trainingSequence.stillMoreEpochs)
      case trainingSequence.status
        when :inPhase1
          mse = noFlockingAdaptNetworkTuned
          dPrimes = recenterEachNeuronsClusters(neuronsCreatingFlockingError)
          mse, dPrimes = flockingOnlyAdaptNetworkTuned(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
        #std("mse=\t", mse)
        #std("dPrimes=\t", dPrimes)

        when :inPhase2
          # @dPrimesOld = nil if (trainingSequence.atStartOfPhase2)

          (neuronsAdaptingToLocalFlockingError + neuronsAdaptingToBackPropedFlockingError).each { |neuron| neuron.learningRate = learningRateFlockPhase2 }

          # TODO NEED TO RESOLVE THIS... QUESTION BELOW
          ###------------   Adaption to FLOCKING Error  ------------------------------
          dPrimes = recenterEachNeuronsClusters(neuronsCreatingFlockingError) ## TODO WOW, why does this statement "cause" such a drop in dPrimes in 2nd hidden layer in Analogy, but not Simplest main??
          mse, dPrimes = flockingOnlyAdaptNetworkFixedFlockingGain(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)

          deltaDPrimeRatios = deltaDPrimes(dPrimes)
          puts "epochs=\t#{trainingSequence.epochs}\tdeltaDPrimeRatios=\t#{deltaDPrimeRatios}\tmse, dPrimes= \t #{mse}\t#{dPrimes}"
        # trainingSequence.dPrimesHaveNotChangedMuch if (deltaDPrimeRatios.max < 0.01)
        else
          STDERR.puts "status of training sequencer is not understood"
      end
      trainingSequence.nextEpoch
    end
    return mse, dPrimes
  end

end


class CircleTrainer < AbstractTrainer
  attr_accessor :learningRateNoFlockPhase1, :neuronsCreatingFlockingError, :neuronsAdaptingToLocalFlockingError,
                :learningRateLocalFlockPhase2, :neuronsAdaptingToBackPropedFlockingError, :learningRateForBackPropedFlockingErrorPhase2

  def nameTrainingGroupsAndLearningRates
    self.learningRateNoFlockPhase1 = args[:learningRateNoFlockPhase1]
    self.neuronsCreatingFlockingError = outputLayer
    self.neuronsAdaptingToLocalFlockingError = outputLayer
    self.learningRateLocalFlockPhase2 = args[:learningRateLocalFlockPhase2]
    self.neuronsAdaptingToBackPropedFlockingError = hiddenLayer
    self.learningRateForBackPropedFlockingErrorPhase2 = args[:learningRateForBackPropedFlockingErrorPhase2]
  end

  def simpleLearningWithFlocking(examples, theNeuron = nil)
    distributeSetOfExamples(examples)
    mse = 99999.0
    dPrimes = nil
    @dPrimesOld = nil

    nameTrainingGroupsAndLearningRates()

    seedClustersInFlockingNeurons(neuronsCreatingFlockingError)
    while ((mse > minMSE) && trainingSequence.stillMoreEpochs)
      case trainingSequence.status
        when :inPhase1
          neuronsWithInputLinks.each { |neuron| neuron.learningRate = learningRateNoFlockPhase1 }
          mse = adaptNetworkNoFlocking
        when :inPhase2
          @dPrimesOld = nil if (trainingSequence.atStartOfPhase2)

          neuronsAdaptingToLocalFlockingError.each { |neuron| neuron.learningRate = learningRateLocalFlockPhase2 }
          neuronsAdaptingToBackPropedFlockingError.each { |neuron| neuron.learningRate = learningRateForBackPropedFlockingErrorPhase2 }

          mse, dPrimes = adaptNetworkWithFlockingOnly(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)

          #if (trainingSequence.epochs%400 == 0)
          #  testMSE = self.oneForwardPassEpoch(testingExamples)
          #  puts "epochs=\t#{trainingSequence.epochs}\tmse=\t#{mse}\ttestMSE=\t#{testMSE}"
          #end
          printIt = selectEpochsForPrinting(epochOffsets = [0, 1, 2, 5, 10], interval=100)

          deltaDPrimeRatios = deltaDPrimes(dPrimes)

        # puts "epochs=\t#{trainingSequence.epochs}\tdeltaDPrimeRatios.max=\t#{deltaDPrimeRatios.max}\tdeltaDPrimeRatios.min=\t#{deltaDPrimeRatios.min}\tdPrimes=\t#{dPrimes}" if (printIt)
        # trainingSequence.dPrimesAreLargeEnough if ((deltaDPrimeRatios.map{|e|e.abs}).max < 0.001)
        else
          STDERR.puts "status of training sequencer is not understood"
      end
      trainingSequence.nextEpoch
    end
    return trainingSequence.epochs, mse, dPrimes
  end

  def selectEpochsForPrinting(epochOffsets, interval = 40.0)
    # TODO    This is HIGHLY UGLY CODE
    epochs = trainingSequence.epochs
    quotient = (epochs / interval).to_f
    truncatedEpochs = quotient.floor * interval
    logicalValues = epochOffsets.map { |offset| ((epochs.to_f - (truncatedEpochs + offset)).abs < 0.00000001) }
    #puts "logicalValues=\t#{logicalValues}"
    printIt = false
    printIt = true if ((logicalValues.select { |e| e }).length > 0)
    #puts "printIt=\t#{printIt}"
    return printIt
  end

  protected

  ###------------   Adaption to Only Flocking Error; NOT!! Output Error  ------------------------

  def adaptNetworkWithFlockingOnly(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
    measureAndStoreAllNeuralResponsesFlockingOnly(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
    neuronsWithInputLinks.each { |aNeuron| aNeuron.addAccumulationToWeight }
    dPrimes = recenterEachNeuronsClusters(neuronsCreatingFlockingError)
    mse = logNetworksResponses(neuronsCreatingFlockingError + neuronsAdaptingToLocalFlockingError)
    [mse, dPrimes]
  end

  def measureAndStoreAllNeuralResponsesFlockingOnly(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
    neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }

    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate } # only really need to do this for the lowest layer that will create flocking error
      outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }

      # measure flocking error created in those neurons, so designated!
      neuronsCreatingFlockingError.each do |aNeuron|
        dataRecorded = aNeuron.recordResponsesForExample
        localFlockingError = aNeuron.calcLocalFlockingError
        dataRecorded[:localFlockingError] = localFlockingError
        aNeuron.recordResponsesForExampleToDB(dataRecorded)
        aNeuron.backPropagate { |errorFromUpperLayers, localFlockError| localFlockError } # necessary for sending flocking error to hidden layer
      end

      # send flocking error to hidden layer
      neuronsAdaptingToBackPropedFlockingError.each { |aNeuron| aNeuron.backPropagate }

      # accumulate weight changes in response to that flocking error so assigned/distributed
      neuronsAdaptingToBackPropedFlockingError.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate }
      neuronsAdaptingToLocalFlockingError.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate { |errorFromUpperLayers, localFlockError| localFlockError } }
    end
  end

  ###------------ One Epoch Testing Section Section ------------------------------
  def measureNeuralResponsesForTesting
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
      neuronsWithInputLinks.each { |aNeuron| aNeuron.recordResponsesForExample }
    end
    mse = network.calcNetworksMeanSquareError
  end
end

class Trainer4Class < AbstractTrainer
  attr_accessor :learningRateNoFlockPhase1, :neuronsCreatingLocalFlockingErrorAndAdaptingToSame,
                :learningRateLocalFlockPhase2, :neuronsAdaptingOnlyToBPOutputError, :learningRateBPOutputErrorPhase2

  def simpleLearningWithFlocking(examples, theNeuron = nil)
    distributeSetOfExamples(examples)
    mse = 99999.0
    dPrimes = nil
    @dPrimesOld = nil

    nameTrainingGroupsAndLearningRates()

    seedClustersInFlockingNeurons(neuronsCreatingLocalFlockingErrorAndAdaptingToSame)
    while ((mse > minMSE) && trainingSequence.stillMoreEpochs)
      case trainingSequence.status
        when :inPhase1
          neuronsWithInputLinks.each { |neuron| neuron.learningRate = learningRateNoFlockPhase1 }

          mse = adaptNetworkNoFlocking

        when :inPhase2
          @dPrimesOld = nil if (trainingSequence.atStartOfPhase2)

          neuronsCreatingLocalFlockingErrorAndAdaptingToSame.each { |neuron| neuron.learningRate = learningRateLocalFlockPhase2 }
          neuronsAdaptingOnlyToBPOutputError.each { |neuron| neuron.learningRate = learningRateBPOutputErrorPhase2 }

          mse, dPrimes = adaptNetworkWithLocalFlockingIncluded(neuronsCreatingLocalFlockingErrorAndAdaptingToSame, neuronsAdaptingOnlyToBPOutputError)

          deltaDPrimeRatios = deltaDPrimes(dPrimes)
          puts "epochs=\t#{trainingSequence.epochs}\tdeltaDPrimeRatios=\t#{deltaDPrimeRatios}" if (trainingSequence.epochs > 30)
        # trainingSequence.dPrimesHaveNotChangedMuch if (deltaDPrimeRatios.max < 0.01)
        else
          STDERR.puts "status of training sequencer is not understood"
      end
      trainingSequence.nextEpoch
    end
    return trainingSequence.epochs, mse, dPrimes
  end

  def nameTrainingGroupsAndLearningRates
    self.learningRateNoFlockPhase1 = args[:learningRateNoFlockPhase1]
    self.neuronsCreatingLocalFlockingErrorAndAdaptingToSame = hiddenLayer
    self.learningRateLocalFlockPhase2 = args[:learningRateLocalFlockPhase2]
    self.neuronsAdaptingOnlyToBPOutputError = neuronsWithInputLinks - neuronsCreatingLocalFlockingErrorAndAdaptingToSame
    self.learningRateBPOutputErrorPhase2 = args[:learningRateBPOutputErrorPhase2]
  end
end

class TrainerAnalogy4ClassNoBPofFlockError < AbstractTrainer
  def initialize(trainingSequence, network, args)
    @trainingSequence = trainingSequence
    @network = network
    @args = args
    @dataStoreManager = SimulationDataStoreManager.instance
    @allNeuronLayers = network.createSimpleLearningANN
    @startTime = Time.now
    @elapsedTime = nil
    @minMSE = args[:minMSE]
    postInitialize
  end

  def postInitialize
    @inputLayer = network.inputLayer
    @hiddenLayer1 = network.hiddenLayer1
    @hiddenLayer2 = network.hiddenLayer2
    @hiddenLayer3 = network.hiddenLayer3
    @outputLayer = network.outputLayer
    @theBiasNeuron = network.theBiasNeuron
  end

  def simpleLearningWithFlocking(examples, arrayOfNeuronsToPlot)
    step1NameTrainingGroupsAndLearningRates
    mse, dPrimes = oneStepOfLearningAndDisplay(examples, arrayOfNeuronsToPlot)

    trainingSequence.nextStep
    step2NameTrainingGroupsAndLearningRates
    mse, dPrimes = oneStepOfLearningAndDisplay(examples, arrayOfNeuronsToPlot)

    4.times do |doubleStepNumber| # 800
      trainingSequence.nextStep
      step3NameTrainingGroupsAndLearningRates
      mse, dPrimes = oneStepOfLearningAndDisplay(examples, arrayOfNeuronsToPlot)

      trainingSequence.nextStep
      step4NameTrainingGroupsAndLearningRates
      mse, dPrimes = oneStepOfLearningAndDisplay(examples, arrayOfNeuronsToPlot)
    end

    return trainingSequence.epochs, mse, dPrimes
  end

  def step1NameTrainingGroupsAndLearningRates
    # PHASE 1   -----   Adaption to OUTPUT Error  ------
    self.neuronsWithInputLinks = hiddenLayer1 + outputLayer
    setUniversalNeuronGroupNames
    self.neuronsToAdaptToOutputError = hiddenLayer1 + outputLayer
    self.learningRateNoFlockPhase1 = args[:learningRateNoFlockPhase1]

    # PHASE 2 -----   Adaption to FLOCKING Error  -------
    self.neuronsCreatingFlockingError = hiddenLayer1
    self.neuronsAdaptingToLocalFlockingError = hiddenLayer1
    self.neuronsAdaptingToBackPropedFlockingError = []
    self.neuronsWhoseClustersNeedToBeSeeded = neuronsCreatingFlockingError
    self.learningRateFlockPhase2 = args[:learningRateLocalFlockPhase2]
  end

  def step2NameTrainingGroupsAndLearningRates
    # Expanding Network to 2 Hidden Layers
    disconnect_one_layer_from_another(hiddenLayer1, outputLayer)
    disconnect_one_layer_from_another([theBiasNeuron], outputLayer)
    connect_layer_to_another(hiddenLayer2, outputLayer, args)
    connect_layer_to_another([theBiasNeuron], outputLayer, args)

    # Because a layer has been added, succeeding layers weights should be reinitialized with random weights.
    outputLayer.each { |aNeuron| aNeuron.randomizeLinkWeights }

    # PHASE 1   -----   Adaption to OUTPUT Error  ------
    self.neuronsWithInputLinks = hiddenLayer1 + hiddenLayer2 + outputLayer
    setUniversalNeuronGroupNames
    self.neuronsToAdaptToOutputError = hiddenLayer2 + outputLayer
    self.learningRateNoFlockPhase1 = args[:learningRateNoFlockPhase1]

    # PHASE 2 -----   Adaption to FLOCKING Error  -------
    # self.neuronsCreatingLocalFlockingErrorAndAdaptingToSame = hiddenLayer2
    self.neuronsCreatingFlockingError = hiddenLayer2
    self.neuronsAdaptingToLocalFlockingError = hiddenLayer2
    self.neuronsAdaptingToBackPropedFlockingError = []
    self.neuronsWhoseClustersNeedToBeSeeded = neuronsCreatingFlockingError
    self.learningRateFlockPhase2 = args[:learningRateLocalFlockPhase2]
  end

  def step3NameTrainingGroupsAndLearningRates
    # PHASE 1   -----   Adaption to OUTPUT Error  ------
    self.neuronsWithInputLinks = hiddenLayer1 + hiddenLayer2 + outputLayer
    setUniversalNeuronGroupNames
    self.neuronsToAdaptToOutputError = hiddenLayer1 + outputLayer
    self.learningRateNoFlockPhase1 = 2.0 * args[:learningRateNoFlockPhase1]

    # PHASE 2 -----   Adaption to FLOCKING Error  -------
    self.neuronsCreatingFlockingError = hiddenLayer1
    self.neuronsAdaptingToLocalFlockingError = hiddenLayer1
    self.neuronsAdaptingToBackPropedFlockingError = []
    #self.neuronsWhoseClustersNeedToBeSeeded = neuronsCreatingFlockingError
    self.learningRateFlockPhase2 = 0.00001 * args[:learningRateLocalFlockPhase2]
  end

  def step4NameTrainingGroupsAndLearningRates
    # PHASE 1   -----   Adaption to OUTPUT Error  ------
    self.neuronsWithInputLinks = hiddenLayer1 + hiddenLayer2 + outputLayer
    setUniversalNeuronGroupNames
    self.neuronsToAdaptToOutputError = hiddenLayer2 + outputLayer
    self.learningRateNoFlockPhase1 = args[:learningRateNoFlockPhase1]

    # PHASE 2 -----   Adaption to FLOCKING Error  -------
    self.neuronsCreatingFlockingError = hiddenLayer2
    self.neuronsAdaptingToLocalFlockingError = hiddenLayer2
    self.neuronsAdaptingToBackPropedFlockingError = []
    #self.neuronsWhoseClustersNeedToBeSeeded = neuronsCreatingFlockingError
    self.learningRateFlockPhase2 = args[:learningRateLocalFlockPhase2]
  end

end # Used for main: "Analogy4Class.rb"

class TunedTrainerAnalogy4ClassNoBPofFlockError < TrainerAnalogy4ClassNoBPofFlockError
  def postInitialize
    @inputLayer = network.inputLayer
    @hiddenLayer1 = network.hiddenLayer1
    @hiddenLayer2 = network.hiddenLayer2
    @hiddenLayer3 = network.hiddenLayer3
    @outputLayer = network.outputLayer
    @theBiasNeuron = network.theBiasNeuron
    @learningRateTuner = LearningRateTuner.new(hiddenLayer1)
  end

  def noFlockingAdaptNetwork
    noFlockingMeasureAndStoreAllNeuralResponses
    neuronsAdaptingToLocalFlockingError.each { |aNeuron| aNeuron.saveDeltaWAccumulated }
    neuronsAdaptingOnlyToBPOutputError.each { |aNeuron| aNeuron.addAccumulationToWeight }
    mse = logNetworksResponses(outputLayer)
    [mse, dPrimes]
  end

  def flockingOnlyAdaptNetwork(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
    flockingOnlyMeasureAndStoreAllNeuralResponses(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
    #learningRateTuner.printInfo
    learningRateTuner.tune

    (neuronsAdaptingToLocalFlockingError + neuronsAdaptingToBackPropedFlockingError).each { |aNeuron| aNeuron.addAccumulationToWeight }
    dPrimes = recenterEachNeuronsClusters(neuronsCreatingFlockingError)
    mse = logNetworksResponses(neuronsCreatingFlockingError)
    [mse, dPrimes]
  end

  def step1NameTrainingGroupsAndLearningRates
    # PHASE 1   -----   Adaption to OUTPUT Error  ------
    self.neuronsWithInputLinks = hiddenLayer1 + outputLayer
    setUniversalNeuronGroupNames
    self.neuronsAdaptingOnlyToBPOutputError = outputLayer
    self.neuronsToAdaptToOutputError = hiddenLayer1 + outputLayer
    self.learningRateNoFlockPhase1 = args[:learningRateNoFlockPhase1]

    # PHASE 2 -----   Adaption to FLOCKING Error  -------
    self.neuronsCreatingFlockingError = hiddenLayer1
    self.neuronsAdaptingToLocalFlockingError = hiddenLayer1
    learningRateTuner.layer = hiddenLayer1
    self.neuronsAdaptingToBackPropedFlockingError = []
    self.neuronsWhoseClustersNeedToBeSeeded = neuronsCreatingFlockingError
    self.learningRateFlockPhase2 = args[:learningRateLocalFlockPhase2]
  end

  def step2NameTrainingGroupsAndLearningRates
    # Expanding Network to 2 Hidden Layers
    disconnect_one_layer_from_another(hiddenLayer1, outputLayer)
    disconnect_one_layer_from_another([theBiasNeuron], outputLayer)
    connect_layer_to_another(hiddenLayer2, outputLayer, args)
    connect_layer_to_another([theBiasNeuron], outputLayer, args)

    # Because a layer has been added, succeeding layers weights should be reinitialized with random weights.
    outputLayer.each { |aNeuron| aNeuron.randomizeLinkWeights }

    # PHASE 1   -----   Adaption to OUTPUT Error  ------
    self.neuronsWithInputLinks = hiddenLayer1 + hiddenLayer2 + outputLayer
    setUniversalNeuronGroupNames
    self.neuronsAdaptingOnlyToBPOutputError = outputLayer
    self.neuronsToAdaptToOutputError = hiddenLayer2 + outputLayer
    self.learningRateNoFlockPhase1 = args[:learningRateNoFlockPhase1]

    # PHASE 2 -----   Adaption to FLOCKING Error  -------
    # self.neuronsCreatingLocalFlockingErrorAndAdaptingToSame = hiddenLayer2
    self.neuronsCreatingFlockingError = hiddenLayer2
    self.neuronsAdaptingToLocalFlockingError = hiddenLayer2
    learningRateTuner.layer = hiddenLayer2
    self.neuronsAdaptingToBackPropedFlockingError = []
    self.neuronsWhoseClustersNeedToBeSeeded = neuronsCreatingFlockingError
    self.learningRateFlockPhase2 = args[:learningRateLocalFlockPhase2]
  end

  def step3NameTrainingGroupsAndLearningRates
    # PHASE 1   -----   Adaption to OUTPUT Error  ------
    self.neuronsWithInputLinks = hiddenLayer1 + hiddenLayer2 + outputLayer
    setUniversalNeuronGroupNames
    self.neuronsAdaptingOnlyToBPOutputError = outputLayer
    self.neuronsToAdaptToOutputError = hiddenLayer1 + outputLayer
    self.learningRateNoFlockPhase1 = 2.0 * args[:learningRateNoFlockPhase1]

    # PHASE 2 -----   Adaption to FLOCKING Error  -------
    self.neuronsCreatingFlockingError = hiddenLayer1
    self.neuronsAdaptingToLocalFlockingError = hiddenLayer1
    learningRateTuner.layer = hiddenLayer1
    self.neuronsAdaptingToBackPropedFlockingError = []
    #self.neuronsWhoseClustersNeedToBeSeeded = neuronsCreatingFlockingError
    self.learningRateFlockPhase2 = 0.00001 * args[:learningRateLocalFlockPhase2]
  end

  def step4NameTrainingGroupsAndLearningRates
    # PHASE 1   -----   Adaption to OUTPUT Error  ------
    self.neuronsWithInputLinks = hiddenLayer1 + hiddenLayer2 + outputLayer
    setUniversalNeuronGroupNames
    self.neuronsAdaptingOnlyToBPOutputError = outputLayer
    self.neuronsToAdaptToOutputError = hiddenLayer2 + outputLayer
    self.learningRateNoFlockPhase1 = args[:learningRateNoFlockPhase1]

    # PHASE 2 -----   Adaption to FLOCKING Error  -------
    self.neuronsCreatingFlockingError = hiddenLayer2
    self.neuronsAdaptingToLocalFlockingError = hiddenLayer2
    learningRateTuner.layer = hiddenLayer2
    self.neuronsAdaptingToBackPropedFlockingError = []
    #self.neuronsWhoseClustersNeedToBeSeeded = neuronsCreatingFlockingError
    self.learningRateFlockPhase2 = args[:learningRateLocalFlockPhase2]
  end
end

##

class FlockingGainTuner
  attr_accessor :neuron, :rng, :searchRange, :sqrtSearchRangeRatio, :flockingFactor, :pastFlockingFactors, :pastDPrimes, :maxHistory, :nSamples

  def initialize(neuron, args)
    @neuron = neuron
    @rng = args[:rng]
    @nSamples = 5
    @maxHistory = args[:maxHistory]
    @sqrtSearchRangeRatio = Math.sqrt(args[:searchRangeRatio])
    @searchRange = (-1.5..-0.5)
    @flockingFactor = nil
    @pastFlockingFactors = []
    @pastDPrimes = []
  end

  def collectData
    unless (flockingFactor.nil?)
      self.pastFlockingFactors << flockingFactor
      self.pastDPrimes << neuron.dPrime
    end
  end

  def setFlockingGain
    trimHistoryIfNecessary()
    self.searchRange = estimateBestSearchRange()
    self.flockingFactor = rng.rand(searchRange)
    neuron.flockingGain = flockingFactor * ratioOfOutputErrorToFlockingError
  end


  private


  def estimateBestSearchRange
    if (pastFlockingFactors.length >= nSamples)
      puts "pastFlockingFactors = #{pastFlockingFactors}"
      puts "pastDPrimes = #{pastDPrimes}"
      bestDPrime = pastDPrimes.max
      puts "bestDPrime = #{bestDPrime}"
      theIndexOfTheBest = pastDPrimes.index(bestDPrime)
      bestFlockingFactor = pastFlockingFactors[theIndexOfTheBest]

      puts "bestFlockingFactor = #{bestFlockingFactor}"

      largestAlgebraic = bestFlockingFactor / sqrtSearchRangeRatio
      proposedSmallestAlgebraic = bestFlockingFactor * sqrtSearchRangeRatio

      smallestAlgebraic = if (proposedSmallestAlgebraic < -1.5)
                                  -1.5
                                else
                                  proposedSmallestAlgebraic
                                end


      newSearchRange = (smallestAlgebraic..largestAlgebraic)
      puts "newSearchRange = #{newSearchRange}"
      return newSearchRange
    else
      return searchRange
    end
  end


  def estimateBestSearchRangeOLD
    if (pastFlockingFactors.length >= nSamples)
      puts "pastFlockingFactors = #{pastFlockingFactors}"
      puts "pastDispersions = #{pastDPrimes}"
      bestDPrime = pastDPrimes.max
      puts "bestDPrime = #{bestDPrime}"

      theIndex = pastDPrimes.index(bestDPrime)
      bestFlockingFactor = pastFlockingFactors[theIndex]

      puts "bestFlockingFactor = #{bestFlockingFactor}"

      largestAlgebraic = bestFlockingFactor / 2.0 # bestNFlockingFactors.max # best / 2.0
      smallestAlgebraic = if ((bestFlockingFactor * 2.0) < -1.5)
                            -1.5
                          else
                            bestFlockingFactor * 2.0
                          end


      newSearchRange = (smallestAlgebraic..largestAlgebraic)
      puts "newSearchRange = #{newSearchRange}"
      return newSearchRange
    else
      return searchRange
    end
  end

  def estimateBestSearchRangeORIGINAL
    if (pastFlockingFactors.length >= nSamples)
      puts "pastFlockingFactors = #{pastFlockingFactors}"
      puts "pastDispersions = #{pastDPrimes}"
      bestNSampleDispersions = pastDPrimes.sort[0...nSamples]
      puts "bestNSampleDispersions = #{bestNSampleDispersions}"
      bestNFlockingFactors = bestNSampleDispersions.collect do |aDispersion|
        theIndex = pastDPrimes.index(aDispersion)
        pastFlockingFactors[theIndex]
      end
      puts "bestNFlockingFactors = #{bestNFlockingFactors}"
      best = bestNFlockingFactors[0]

      largestAlgebraic = best / 2.0 # bestNFlockingFactors.max # best / 2.0
      smallestAlgebraic = if ((best * 2.0) < -1.5)
                            -1.5
                          else
                            best * 2.0
                          end


      newSearchRange = (smallestAlgebraic..largestAlgebraic)
      puts "newSearchRange = #{newSearchRange}"
      return newSearchRange
    else
      return searchRange
    end
  end

  def ratioOfOutputErrorToFlockingError
    outputErrorSignals = neuron.inputLinks.collect { |aLink| aLink.previousDeltaWAccumulated }
    flockingErrorSignals = neuron.inputLinks.collect { |aLink| aLink.deltaWAccumulated }
    vOutputErrorSignals = Vector.elements(outputErrorSignals)
    vFlockingErrorSignals = Vector.elements(flockingErrorSignals)
    return (vOutputErrorSignals.r / vFlockingErrorSignals.r)
  end

  def trimHistoryIfNecessary
    if (pastFlockingFactors.length > maxHistory)
      pastFlockingFactors.shift
      pastDPrimes.shift
    end
  end
end

class WeightChangeNormalizer
  attr_accessor :layer, :weightChangeSetPoint

  def initialize(layer, args, weightChangeSetPoint = 0.01)
    @layer = layer
    @weightChangeSetPoint = weightChangeSetPoint
  end

  def normalizeWeightChanges
    layerGain = weightChangeSetPoint / maxWeightChange
    layer.each { |neuron| neuron.inputLinks.each { |aLink| aLink.deltaWAccumulated = layerGain * aLink.deltaWAccumulated } }
  end

  private

  def maxWeightChange
    acrossLayerMaxValues = layer.collect do |aNeuron|
      accumulatedDeltaWsAbs = aNeuron.inputLinks.collect { |aLink| aLink.deltaWAccumulated.abs }
      accumulatedDeltaWsAbs.max
    end
    return acrossLayerMaxValues.max
  end
end

class LearningAndFlockingTuner
  attr_accessor :layer, :weightChangeNormalizer, :flockingGainTuners

  def initialize(layer, args)
    @layer = layer
    @weightChangeNormalizer = WeightChangeNormalizer.new(layer, args)
    @flockingGainTuners = layer.collect { |neuron| FlockingGainTuner.new(neuron, args) }
  end

  def tune
    flockingGainTuners.each do |aFlockingTuner|
      aFlockingTuner.collectData
      aFlockingTuner.setFlockingGain
    end
    layer.each { |neuron| neuron.inputLinks.each { |aLink| aLink.combineWeightChangesDueToOutputErrorAndFlocking } }
    weightChangeNormalizer.normalizeWeightChanges
  end

  def printInfo
  end
end


##

class TrainingSequence
  attr_accessor :network, :args, :epochs, :epochsInPhase1, :epochsInPhase2, :numberOfEpochsInCycle,
                :maxNumberOfEpochs, :epochsSinceBeginningOfCycle, :status,
                :stillMoreEpochs, :lastEpoch, :inPhase2, :inPhase1,
                :atStartOfCycle, :atStartOfPhase1, :atStartOfPhase2,
                :atStartOfTraining, :afterFirstEpoch, :dataStoreManager,
                :tagDataSet, :epochDataSet, :numberOfEpochsBetweenStoringDBRecords

  private_class_method(:new)

  @@trainingSequence = nil

  def TrainingSequence.create(network, args)
    @@trainingSequence = new(network, args) unless @@trainingSequence
    @@trainingSequence
  end

  def TrainingSequence.instance
    return @@trainingSequence
  end

  def initialize(network, args)
    @network = network
    @args = args
    @dataStoreManager = SimulationDataStoreManager.instance
    @tagDataSet = dataStoreManager.tagDataSet
    @epochDataSet = dataStoreManager.epochDataSet

    @epochsInPhase1 = args[:phase1Epochs]
    @epochsInPhase2 = args[:phase2Epochs]
    @numberOfEpochsInCycle = epochsInPhase1 + epochsInPhase2
    @maxNumberOfEpochs = args[:maxNumEpochs]
    @numberOfEpochsBetweenStoringDBRecords = @args[:numberOfEpochsBetweenStoringDBRecords]

    @epochs = -1
    @epochsSinceBeginningOfCycle = -1
    @atStartOfTraining = true
    @afterFirstEpoch = false
    @stillMoreEpochs = true
    @lastEpoch = false
    @status = :inPhase1
    nextEpoch
  end

  def startAtBeginningOfCycle
    @epochsSinceBeginningOfCycle = -1
    @status = :inPhase1
  end

  def dPrimesAreLargeEnough
    startAtBeginningOfCycle
  end

  def nextEpoch
    storeSequenceTagData

    self.epochs += 1
    self.atStartOfTraining = false if (epochs > 0)
    self.afterFirstEpoch = true unless (atStartOfTraining)

    self.epochsSinceBeginningOfCycle += 1
    self.epochsSinceBeginningOfCycle = 0 if (epochsSinceBeginningOfCycle == numberOfEpochsInCycle)

    self.atStartOfCycle = false
    self.atStartOfPhase1 = false
    if (epochsSinceBeginningOfCycle == 0)
      self.atStartOfCycle = true
      self.atStartOfPhase1 = true if (epochsInPhase1 > 0)
    end

    self.inPhase1 = false
    self.inPhase1 = true if (epochsSinceBeginningOfCycle < epochsInPhase1)

    self.atStartOfPhase2 = false
    self.atStartOfPhase2 = true if ((epochsSinceBeginningOfCycle == epochsInPhase1) && (epochsInPhase2 > 0))

    self.inPhase2 = false
    self.inPhase2 = true if (epochsSinceBeginningOfCycle >= epochsInPhase1)

    self.lastEpoch = false
    self.lastEpoch = true if (epochs == (maxNumberOfEpochs - 1))

    self.stillMoreEpochs = true
    self.stillMoreEpochs = false if (epochs >= maxNumberOfEpochs)

    self.status = :inPhase1 if (inPhase1 == true)
    self.status = :inPhase2 if (inPhase2 == true)

    dataStoreManager.epochNumber = epochs
  end

  def timeToRecordData
    record = false
    return if (epochs < 0)
    record = true if ((epochs % numberOfEpochsBetweenStoringDBRecords) == 0)
    record = true if lastEpoch
    return record
  end

  def nextStep
    self.maxNumberOfEpochs = epochs + args[:maxNumEpochs]
    self.epochsSinceBeginningOfCycle = -1
    self.atStartOfTraining = true
    self.afterFirstEpoch = false
    self.stillMoreEpochs = true
    self.lastEpoch = false
    self.status = :inPhase1
    nextEpoch
    self.atStartOfTraining = true # TODO this is a fudge...
    self.afterFirstEpoch = true unless (atStartOfTraining)
  end

  private

  def storeSequenceTagData
    if (timeToRecordData)
      epochsSinceBeginningOfPhase = nil
      if (inPhase1)
        learningPhase = 'phase1'
        epochsSinceBeginningOfPhase = epochsSinceBeginningOfCycle
      else
        learningPhase = 'phase2'
        epochsSinceBeginningOfPhase = epochsSinceBeginningOfCycle - epochsInPhase1
      end
      dataToStore = {:epochNumber => epochs, :experimentNumber => dataStoreManager.theNumberOfTheCurrentExperiment,
                     :mse => network.mse, :learningPhase => learningPhase,
                     :epochsSinceBeginningOfPhase => epochsSinceBeginningOfPhase,
                     :epochsSinceBeginningOfCycle => epochsSinceBeginningOfCycle}
      tagDataSet.insert(dataToStore)
    end
  end
end

