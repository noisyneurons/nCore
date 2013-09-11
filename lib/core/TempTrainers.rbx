
class TrainerAnalogy4Class < AbstractTrainer
  attr_accessor :learningRateNoFlockPhase1, :neuronsCreatingLocalFlockingErrorAndAdaptingToSame,
                :learningRateLocalFlockPhase2, :neuronsAdaptingOnlyToBPOutputError, :learningRateBPOutputErrorPhase2,
                :hiddenLayer1, :hiddenLayer2, :hiddenLayer3, :allHiddenLayers, :neuronsWhoseClustersNeedToBeSeeded,
                :neuronsCreatingFlockingError, :neuronsAdaptingToLocalFlockingError,
                :neuronsAdaptingToBackPropedFlockingError, :learningRateForBackPropedFlockingErrorPhase2,
                :neuronsToAdaptToOutputError, :neuronsToAdaptToOutputErrorInReverseOrder,
                :neuronsAdaptingToOutputOrLocalFlockingError, :neuronsAdaptingToOutputOrLocalFlockingErrorInReverseOrder

  def initialize(trainingSequence, network, args)
    @trainingSequence = trainingSequence
    @network = network
    @args = args
    @dataStoreManager = args[:dataStoreManager]
    @allNeuronLayers = network.createSimpleLearningANN
    @startTime = Time.now
    @elapsedTime = nil
    @minMSE = args[:minMSE]
  end

  def simpleLearningWithFlocking(examples)
    mse, dPrimes  = step1Learning(examples)
    puts network
    trainingSequence.nextStep
    dPrimes, mse = step2Learning()
    #puts network
    #trainingSequence.nextStep
    #dPrimes, mse = step3Learning()
    return trainingSequence.epochs, mse, dPrimes
  end

  def setUniversalNeuronGroupNames
    self.neuronsWithInputLinksInReverseOrder = neuronsWithInputLinks.reverse
    self.allNeuronsInOneArray = inputLayer + neuronsWithInputLinks
  end

  def nameTrainingGroupsAndLearningRatesStep1
    self.inputLayer = network.inputLayer
    self.hiddenLayer1 = network.hiddenLayer1
    self.hiddenLayer2 = network.hiddenLayer2
    self.hiddenLayer3 = network.hiddenLayer3
    self.outputLayer = network.outputLayer
    self.theBiasNeuron = network.theBiasNeuron

    # PHASE 1
    self.neuronsWithInputLinks = hiddenLayer1 + outputLayer
    setUniversalNeuronGroupNames
    self.learningRateNoFlockPhase1 = args[:learningRateNoFlockPhase1]

    # PHASE 2
    self.neuronsCreatingLocalFlockingErrorAndAdaptingToSame = hiddenLayer1
    self.neuronsWhoseClustersNeedToBeSeeded = neuronsCreatingLocalFlockingErrorAndAdaptingToSame
    self.learningRateLocalFlockPhase2 = args[:learningRateLocalFlockPhase2]
    self.neuronsAdaptingOnlyToBPOutputError = neuronsWithInputLinks - neuronsCreatingLocalFlockingErrorAndAdaptingToSame # the output neurons
    self.learningRateBPOutputErrorPhase2 = args[:learningRateBPOutputErrorPhase2]
  end

  def step1Learning(examples)
    nameTrainingGroupsAndLearningRatesStep1()
    distributeSetOfExamples(examples)
    mse = 99999.0
    dPrimes = nil
    @dPrimesOld = nil

    seedClustersInFlockingNeurons(neuronsWhoseClustersNeedToBeSeeded)
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
    return mse, dPrimes
  end


  def nameTrainingGroupsAndLearningRatesStep2
    disconnect_one_layer_from_another(hiddenLayer1, outputLayer)
    disconnect_one_layer_from_another([theBiasNeuron], outputLayer)
    connect_layer_to_another(hiddenLayer2, outputLayer, args)
    connect_layer_to_another([theBiasNeuron], outputLayer, args)

    outputLayer.each {|aNeuron| aNeuron.randomizeLinkWeights}

    # PHASE 1
    self.neuronsWithInputLinks = hiddenLayer1 + hiddenLayer2 + outputLayer
    setUniversalNeuronGroupNames
    self.neuronsToAdaptToOutputError = hiddenLayer2 + outputLayer
    self.neuronsToAdaptToOutputErrorInReverseOrder  = neuronsToAdaptToOutputError.reverse
    self.learningRateNoFlockPhase1 = args[:learningRateNoFlockPhase1]

    # PHASE 2
    previousStepsNeuronsCreatingLocalFlockingErrorAndAdaptingToSame = neuronsCreatingLocalFlockingErrorAndAdaptingToSame
    self.neuronsCreatingLocalFlockingErrorAndAdaptingToSame = hiddenLayer2
    self.neuronsWhoseClustersNeedToBeSeeded = neuronsCreatingLocalFlockingErrorAndAdaptingToSame - previousStepsNeuronsCreatingLocalFlockingErrorAndAdaptingToSame
    self.learningRateLocalFlockPhase2 = args[:learningRateLocalFlockPhase2]
    self.neuronsAdaptingOnlyToBPOutputError = outputLayer
    self.learningRateBPOutputErrorPhase2 = args[:learningRateBPOutputErrorPhase2]
    self.neuronsAdaptingToOutputOrLocalFlockingError = hiddenLayer2 + outputLayer
    self.neuronsAdaptingToOutputOrLocalFlockingErrorInReverseOrder  = neuronsAdaptingToOutputOrLocalFlockingError.reverse
  end

  def step2Learning
    nameTrainingGroupsAndLearningRatesStep2()
    mse = 99999.0
    dPrimes = nil
    @dPrimesOld = nil

    seedClustersInFlockingNeurons(neuronsWhoseClustersNeedToBeSeeded)
    while ((mse > minMSE) && trainingSequence.stillMoreEpochs)
      case trainingSequence.status
        when :inPhase1
          neuronsWithInputLinks.each { |neuron| neuron.learningRate = learningRateNoFlockPhase1 }

          mse = step2AdaptNetworkNoFlocking

        when :inPhase2
          @dPrimesOld = nil if (trainingSequence.atStartOfPhase2)

          neuronsCreatingLocalFlockingErrorAndAdaptingToSame.each { |neuron| neuron.learningRate = learningRateLocalFlockPhase2 }
          neuronsAdaptingOnlyToBPOutputError.each { |neuron| neuron.learningRate = learningRateBPOutputErrorPhase2 }

          mse, dPrimes = step2AdaptNetworkWithLocalFlockingIncluded(neuronsCreatingLocalFlockingErrorAndAdaptingToSame, neuronsAdaptingOnlyToBPOutputError)

          deltaDPrimeRatios = deltaDPrimes(dPrimes)
          puts "epochs=\t#{trainingSequence.epochs}\tdeltaDPrimeRatios=\t#{deltaDPrimeRatios}" if (trainingSequence.epochs > 30)
        # trainingSequence.dPrimesHaveNotChangedMuch if (deltaDPrimeRatios.max < 0.01)
        else
          STDERR.puts "status of training sequencer is not understood"
      end
      trainingSequence.nextEpoch
    end
    return dPrimes, mse
  end

  def step2AdaptNetworkNoFlocking
    step2MeasureAndStoreAllNeuralResponsesNoFlocking
    neuronsToAdaptToOutputError.each { |aNeuron| aNeuron.addAccumulationToWeight }
    mse =logNetworksResponses(nil)
  end

  def step2MeasureAndStoreAllNeuralResponsesNoFlocking
    neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      neuronsToAdaptToOutputErrorInReverseOrder.each { |aNeuron| aNeuron.backPropagate }
      outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
      neuronsToAdaptToOutputError.each { |aNeuron| aNeuron.recordResponsesForExampleToDB(aNeuron.recordResponsesForExample) }
      neuronsToAdaptToOutputError.each { |neuron| neuron.calcDeltaWsAndAccumulate }
    end
  end


  def step2AdaptNetworkWithLocalFlockingIncluded(neuronsCreatingLocalFlockingErrorAndAdaptingToSame, neuronsAdaptingOnlyToBPOutputError)
    step2MeasureAndStoreAllNeuralResponsesWithLocalFlockingIncluded(neuronsCreatingLocalFlockingErrorAndAdaptingToSame, neuronsAdaptingOnlyToBPOutputError)
    neuronsAdaptingToOutputOrLocalFlockingError.each { |aNeuron| aNeuron.addAccumulationToWeight }
    dPrimes = recenterEachNeuronsClusters(neuronsCreatingLocalFlockingErrorAndAdaptingToSame)
    mse = logNetworksResponses(neuronsCreatingLocalFlockingErrorAndAdaptingToSame)
    [mse, dPrimes]
  end

  def step2MeasureAndStoreAllNeuralResponsesWithLocalFlockingIncluded(neuronsCreatingAndAdaptingToThatFlockingError, neuronsAdaptingOnlyToBPOutputError)
    neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
    neuronsAdaptingToOutputOrLocalFlockingError.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      neuronsAdaptingToOutputOrLocalFlockingErrorInReverseOrder.each { |aNeuron| aNeuron.backPropagate }
      outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }

      neuronsCreatingAndAdaptingToThatFlockingError.each do |aNeuron|
        dataRecorded = aNeuron.recordResponsesForExample
        localFlockingError = aNeuron.calcLocalFlockingError
        dataRecorded[:localFlockingError] = localFlockingError
        aNeuron.recordResponsesForExampleToDB(dataRecorded)
        aNeuron.calcDeltaWsAndAccumulate { |errorFromUpperLayers, localFlockError| localFlockError }
      end

      neuronsAdaptingOnlyToBPOutputError.each do |aNeuron|
        aNeuron.recordResponsesForExampleToDB(aNeuron.recordResponsesForExample)
        aNeuron.calcDeltaWsAndAccumulate
      end
    end
  end



#  Need to adapt hiddenLayer1 to both OutputError and non-local Flocking Error coming from hiddenLayer2

  def nameTrainingGroupsAndLearningRatesStep3
  # PHASE 1
    # ? outputLayer.each {|aNeuron| aNeuron.randomizeLinkWeights} # should we start off where we left off?
    # output error adaptation naming
    self.neuronsWithInputLinks = hiddenLayer1 + hiddenLayer2 + outputLayer
    setUniversalNeuronGroupNames()
    self.neuronsToAdaptToOutputError = hiddenLayer1 + outputLayer
    self.learningRateNoFlockPhase1 = args[:learningRateNoFlockPhase1]

    # PHASE 2
    # TODO need to make seeding more robust, so we don't need to worry about this
    #  previousStepsNeuronsCreatingLocalFlockingErrorAndAdaptingToSame = neuronsCreatingLocalFlockingErrorAndAdaptingToSame
    self.neuronsCreatingFlockingError = hiddenLayer2
    self.neuronsAdaptingToBackPropedFlockingError = hiddenLayer1
    self.learningRateForBackPropedFlockingErrorPhase2 =  args[:learningRateForBackPropedFlockingErrorPhase2]
  end



  def step3Learning
    nameTrainingGroupsAndLearningRatesStep3()
    mse = 99999.0
    dPrimes = nil
    @dPrimesOld = nil

    seedClustersInFlockingNeurons(neuronsWhoseClustersNeedToBeSeeded)
    while ((mse > minMSE) && trainingSequence.stillMoreEpochs)
      case trainingSequence.status
        when :inPhase1
          neuronsWithInputLinks.each { |neuron| neuron.learningRate = learningRateNoFlockPhase1 }

          mse = step2AdaptNetworkNoFlocking

        when :inPhase2
          @dPrimesOld = nil if (trainingSequence.atStartOfPhase2)

          neuronsCreatingLocalFlockingErrorAndAdaptingToSame.each { |neuron| neuron.learningRate = learningRateLocalFlockPhase2 }
          neuronsAdaptingOnlyToBPOutputError.each { |neuron| neuron.learningRate = learningRateBPOutputErrorPhase2 }

          mse, dPrimes = step2AdaptNetworkWithLocalFlockingIncluded(neuronsCreatingLocalFlockingErrorAndAdaptingToSame, neuronsAdaptingOnlyToBPOutputError)

          deltaDPrimeRatios = deltaDPrimes(dPrimes)
          puts "epochs=\t#{trainingSequence.epochs}\tdeltaDPrimeRatios=\t#{deltaDPrimeRatios}" if (trainingSequence.epochs > 30)
        # trainingSequence.dPrimesHaveNotChangedMuch if (deltaDPrimeRatios.max < 0.01)
        else
          STDERR.puts "status of training sequencer is not understood"
      end
      trainingSequence.nextEpoch
    end
    return dPrimes, mse
  end




  def noFlockingStep3AdaptNetwork
    noFlockingStep3MeasureAndStoreAllNeuralResponses
    neuronsToAdaptToOutputError.each { |aNeuron| aNeuron.addAccumulationToWeight }
    mse =logNetworksResponses(nil)
  end

  def noFlockingStep3MeasureAndStoreAllNeuralResponses
    neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      neuronsToAdaptToOutputErrorInReverseOrder.each { |aNeuron| aNeuron.backPropagate }
      outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
      neuronsToAdaptToOutputError.each { |aNeuron| aNeuron.recordResponsesForExampleToDB(aNeuron.recordResponsesForExample) }
      neuronsToAdaptToOutputError.each { |neuron| neuron.calcDeltaWsAndAccumulate }
    end
  end






  #def step3Learning
  #  nameTrainingGroupsAndLearningRatesStep3()
  #  mse = 99999.0
  #  dPrimes = nil
  #  @dPrimesOld = nil
  #
  #  seedClustersInFlockingNeurons(neuronsWhoseClustersNeedToBeSeeded)
  #  while ((mse > minMSE) && trainingSequence.stillMoreEpochs)
  #    case trainingSequence.status
  #      when :inPhase1
  #        neuronsWithInputLinks.each { |neuron| neuron.learningRate = learningRateNoFlockPhase1 }
  #
  #        mse = adaptNetworkNoFlocking
  #
  #      when :inPhase2
  #        @dPrimesOld = nil if (trainingSequence.atStartOfPhase2)
  #
  #        neuronsAdaptingToLocalFlockingError.each { |neuron| neuron.learningRate = learningRateLocalFlockPhase2 }
  #        neuronsAdaptingToBackPropedFlockingError.each { |neuron| neuron.learningRate = learningRateForBackPropedFlockingErrorPhase2 }
  #        neuronsAdaptingOnlyToBPOutputError.each { |neuron| neuron.learningRate = learningRateBPOutputErrorPhase2 }
  #
  #        mse, dPrimes = adaptNetworkWithMainlyFlocking(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
  #
  #        deltaDPrimeRatios = deltaDPrimes(dPrimes)
  #        puts "epochs=\t#{trainingSequence.epochs}\tdeltaDPrimeRatios=\t#{deltaDPrimeRatios}" if (trainingSequence.epochs > 30)
  #      # trainingSequence.dPrimesHaveNotChangedMuch if (deltaDPrimeRatios.max < 0.01)
  #      else
  #        STDERR.puts "status of training sequencer is not understood"
  #    end
  #    trainingSequence.nextEpoch
  #  end
  #  return dPrimes, mse
  #end
  #

  ###------------   Adaption to Only Flocking Error; NOT!! Output Error  ------------------------

  def adaptNetworkWithMainlyFlocking(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
    measureAndStoreAllNeuralResponsesMainlyFlocking(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
    neuronsWithInputLinks.each { |aNeuron| aNeuron.addAccumulationToWeight }
    dPrimes = recenterEachNeuronsClusters(neuronsCreatingFlockingError)
    mse = logNetworksResponses(neuronsWithInputLinks)
    puts "Step3 mse =\t#{mse}"
    [mse, dPrimes]
  end

  def measureAndStoreAllNeuralResponsesMainlyFlocking(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
    neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }

    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate } # only really need to do this for the lowest layer that will create flocking error
      outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
      outputLayer.each { |aNeuron| aNeuron.recordResponsesForExample }
      neuronsAdaptingOnlyToBPOutputError.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate }

      # measure flocking error created in those neurons, so designated!
      neuronsCreatingFlockingError.each do |aNeuron|
        dataRecorded = aNeuron.recordResponsesForExample
        localFlockingError = aNeuron.calcLocalFlockingError
        dataRecorded[:localFlockingError] = localFlockingError
        aNeuron.recordResponsesForExampleToDB(dataRecorded)
        aNeuron.backPropagate { |errorFromUpperLayers, localFlockError| localFlockError } # necessary for sending flocking error to hidden layer
      end
      neuronsAdaptingToBackPropedFlockingError.each { |aNeuron| aNeuron.backPropagate } # This completes the process started by the previous non-trivial line of code
      neuronsAdaptingToBackPropedFlockingError.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate }

      # the following line is for hidden layer 2
      neuronsAdaptingToLocalFlockingError.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate { |errorFromUpperLayers, localFlockError| localFlockError } }
    end
  end



### Originals...
  def nameTrainingGroupsAndLearningRatesStep2Original
    # now the 2nd hidden layer is included in training
    # Step 2 training...

    disconnect_one_layer_from_another(hiddenLayer1, outputLayer)
    disconnect_one_layer_from_another([theBiasNeuron], outputLayer)
    connect_layer_to_another(hiddenLayer2, outputLayer, args)
    connect_layer_to_another([theBiasNeuron], outputLayer, args)

    self.neuronsWithInputLinks = hiddenLayer1 + hiddenLayer2 + outputLayer
    self.neuronsWithInputLinksInReverseOrder = neuronsWithInputLinks.reverse
    self.allNeuronsInOneArray = inputLayer + neuronsWithInputLinks
    self.learningRateNoFlockPhase1 = args[:learningRateNoFlockPhase1]

    previousStepsNeuronsCreatingLocalFlockingErrorAndAdaptingToSame = neuronsCreatingLocalFlockingErrorAndAdaptingToSame
    self.neuronsCreatingLocalFlockingErrorAndAdaptingToSame = hiddenLayer1 + hiddenLayer2
    self.neuronsWhoseClustersNeedToBeSeeded = neuronsCreatingLocalFlockingErrorAndAdaptingToSame - previousStepsNeuronsCreatingLocalFlockingErrorAndAdaptingToSame
    self.learningRateLocalFlockPhase2 = args[:learningRateLocalFlockPhase2]
    self.neuronsAdaptingOnlyToBPOutputError = neuronsWithInputLinks - neuronsCreatingLocalFlockingErrorAndAdaptingToSame # the output neurons
    self.learningRateBPOutputErrorPhase2 = args[:learningRateBPOutputErrorPhase2]
  end

  def step2OriginalLearning
    nameTrainingGroupsAndLearningRatesStep2()
    mse = 99999.0
    dPrimes = nil
    @dPrimesOld = nil

    seedClustersInFlockingNeurons(neuronsWhoseClustersNeedToBeSeeded)
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
    return dPrimes, mse
  end
end

