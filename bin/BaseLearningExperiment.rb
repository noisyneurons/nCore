### VERSION "nCore"
## ../nCore/bin/BaseLearningExperiments.rb

class Experiment
  attr_accessor :network, :theTrainer, :descriptionOfExperiment, :taskID, :jobID, :jobName, :randomNumberSeed,
                :experimentLogger, :simulationDataStoreManager, :examples, :numberOfExamples, :args, :trainingSequence
  include ExampleDistribution
  include DataSetGenerators

  def initialize(descriptionOfExperiment, baseRandomNumberSeed)
    @descriptionOfExperiment = descriptionOfExperiment

    @taskID = ((ENV['SGE_TASK_ID']).to_i) || 0
    @randomNumberSeed = baseRandomNumberSeed + (taskID * 10000)
    @args = self.setParameters
    srand(@args[:randomNumberSeed])

    puts "sleeping" unless ($currentHost == "localhost")
    sleep(rand * 30) unless ($currentHost == "localhost")

    @jobID = ((ENV['JOB_ID']).to_i) || 0
    @jobName = descriptionOfExperiment[0...10]

    @experimentLogger = ExperimentLogger.new(descriptionOfExperiment, jobName)
    $globalExperimentNumber = experimentLogger.experimentNumber
    #@args = self.setParameters
    @examples = createTrainingSet
    args[:testingExamples] = createTestingSet

    @trainingSequence = args[:trainingSequence].new(args)

    #   @simulationDataStoreManager = SimulationDataStoreManager.new(args)
    @args[:trainingSequence] = trainingSequence
  end

  def setParameters

    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,
        :randomNumberSeed => randomNumberSeed,

        # training parameters re. Output Error
        :outputErrorLearningRate => 0.02,
        :minMSE => 0.0001,
        :maxNumEpochs => 4e3,
        :numLoops => 10,

        # Network Architecture
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 0,
        :numberOfOutputNeurons => 1,
        :weightRange => 1.0,
        :typeOfLink => Link,

        # Training Set parameters
        :numberOfExamples => (self.numberOfExamples = nil),
    }
  end

  def createDataSet
    STDERR.puts "Error: base class method called!!"
    STDERR.puts "Error: Incorrect Number of Examples Generated and/or Specified" unless (examples.length == args[:numberOfExamples])
    return examples
  end

  def createTrainingSet
    examples = createDataSet
    puts "length of examples = #{examples.length}"
    puts examples
    return examples
  end

  def createTestingSet
    return createDataSet
  end

  def temporarilySetSpecificWeights(network)
    selfOrgLayer = network.allNeuronLayers[1]
    selfOrgNeuron = selfOrgLayer[0]
    selfOrgNeuron.inputLinks[0].weight = 0.105
    selfOrgNeuron.inputLinks[1].weight = 0.1
  end

  def performSimulation

######################## Create Network and Trainer ....
    self.network, self.theTrainer = createNetworkAndTrainer

###################################### perform Learning/Training  ##########################################

    startingTime = Time.now
    lastEpoch, trainingMSE, testMSE = theTrainer.train
    endingTime = Time.now

############################## reporting results....

    puts "lastEpoch, trainingMSE, testMSE, startingTime, endingTime "
    puts lastEpoch, trainingMSE, testMSE, startingTime, endingTime
  end

  # routines supporting 'reporting results':

  #def reportTrainingResults(neuronToDisplay, descriptionOfExperiment, lastEpoch, lastTrainingMSE, lastTestingMSE, network, startingTime)
  #
  #  endOfTrainingReport(lastEpoch, lastTestingMSE, lastTrainingMSE, network)
  #
  #  #neuronDataSummary(neuronToDisplay)
  #
  #  #detailedNeuronDataSummary(neuronToDisplay)
  #
  #  trainingDataRecords = trainingDataSummary
  #
  #  storeSnapShotData(descriptionOfExperiment, lastEpoch, lastTestingMSE, lastTrainingMSE, network, startingTime)
  #
  #  snapShotDataSummary
  #
  #  plotMSEvsEpochNumber(trainingDataRecords)
  #
  #  # plotTrainingResults(neuronToDisplay)
  #end


  #def plotTrainingResults(arrayOfNeuronsToPlot)
  #  generatePlotForEachNeuron(arrayOfNeuronsToPlot) if arrayOfNeuronsToPlot.present?
  #end
end


