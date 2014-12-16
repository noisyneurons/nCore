### VERSION "nCore"
## ../nCore/lib/core/Sequencers.rb
## NOT CURRENTLY USED!!

require 'C:/Code/DevRuby/NN2012/nCore/lib/core/Utilities'

class Sequencer
  attr_accessor :network, :errorCriteria, :numberOfExamples, :args,
                :networkRecorder, :epochNumber

  def initialize(network, errorCriteria, numberOfExamples, args)
    @network = network
    @allNeuronLayers = @network.allNeuronLayers
    @allNeuronsUnLayered = @allNeuronLayers.flatten
    @inputLayerOfNeurons = @allNeuronLayers.first
    @neuronsForBPLearning = @allNeuronsUnLayered - @inputLayerOfNeurons
    @neuronsForBPLearningReversed = @neuronsForBPLearning.reverse
    @neuronsInOutputLayer = @allNeuronLayers.last
    @neuronsInHiddenLayer = @neuronsForBPLearning - @neuronsInOutputLayer

    @errorCriteria = errorCriteria
    @numberOfExamples = numberOfExamples
    @args = args

    @networkRecorder = NetworkRecorder.new(args)
    @epochNumber = 0
  end

  def executeCommand(*command, learnersThatShouldReceiveCommand)
    learnersThatShouldReceiveCommand.each { |learner| learner.send(*command) }
  end

  def printSampleNeuron
    logger.puts @neuronsInHiddenLayer[1]
  end

  def allNeurons(*command)
    executeCommand(*command, @allNeuronsUnLayered)
  end

  def inputLayerOfNeurons(*command)
    executeCommand(*command, @inputLayerOfNeurons)
  end

  def neuronsInHiddenLayer(*command)
    executeCommand(*command, @neuronsInHiddenLayer)
  end

  def neuronsForBPLearning(*command)
    executeCommand(*command, @neuronsForBPLearning)
  end

  def neuronsForBPLearningReversed(*command)
    executeCommand(*command, @neuronsForBPLearningReversed)
  end

  def neuronsInOutputLayer(*command)
    executeCommand(*command, @neuronsInOutputLayer)
  end

  #def processMultipleEpochs
  #   while (@network.networkMeanSquaredError > errorCriteria)
  #     yield(self)
  #   end
  #end
  #

  def processMultipleEpochs
    while (self.epochNumber < 1)
      yield(self)
    end
  end


  def processOneEpoch
    @neuronsForBPLearning.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
    yield(self)
    logger.puts "network.calcNetworksMeanSquareError, epochNumber = #{network.calcNetworksMeanSquareError},\t#{epochNumber}"
    networkRecorder.recordResponse(network.calcNetworksMeanSquareError, epochNumber)
    self.epochNumber += 1
  end

  def processAllExamples
    @numberOfExamples.times do |exampleNumber|
      yield(self, exampleNumber)
    end
  end
end

