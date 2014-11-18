### VERSION "nCore"
## ../nCore/bin/BaseLearningExperiments.rb

class Experiment
  attr_accessor :network, :theTrainer, :descriptionOfExperiment, :taskID, :jobID, :jobName, :randomNumberSeed,
                :experimentLogger, :simulationDataStoreManager,
                :dataSetGenerator, :examples, :numberOfExamples, :args, :trainingSequence
  include ExampleDistribution
  include DataSetGenerators

  def initialize(baseRandomNumberSeed)

    @taskID = ((ENV['SGE_TASK_ID']).to_i) || 0
    @randomNumberSeed = baseRandomNumberSeed + (taskID * 10000)
    @numberOfExamples = nil
    @args = self.setParameters
    srand(@args[:randomNumberSeed])

    puts "sleeping" unless ($currentHost == "localhost")
    sleep(rand * 30) unless ($currentHost == "localhost")

    @descriptionOfExperiment = args[:descriptionOfExperiment]
    @jobID = ((ENV['JOB_ID']).to_i) || 0
    @jobName = descriptionOfExperiment[0...10]

    @experimentLogger = ExperimentLogger.new(descriptionOfExperiment, jobName)
    $globalExperimentNumber = experimentLogger.experimentNumber

    @examples = createTrainingSet
    args[:testingExamples] = self.createTestingSet  # TODO clumsy putting testing examples in args hash

    @trainingSequence = args[:trainingSequence].new(args)
    args[:trainingSequence] = trainingSequence     # TODO clumsy putting trainingSequence in args hash  ... create later
  end

  def setParameters
    self.numberOfExamples = nil
    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => "NA",
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
        :numberOfExamples => numberOfExamples,
    }
  end

  def createDataSet
    STDERR.puts "Error: base class method called!!"
    STDERR.puts "Error: Incorrect Number of Examples Generated and/or Specified" unless (examples.length == args[:numberOfExamples])
  end

  def createTrainingSet
    classOfDataSetGenerator = args[:classOfDataSetGenerator]
    self.dataSetGenerator = classOfDataSetGenerator.new(args)
    examples = dataSetGenerator.generate(args[:numberOfExamples], args[:standardDeviationOfAddedGaussianNoise])
    puts "length of examples = #{examples.length}"
    return examples
  end

  def createTestingSet
    testExamples = dataSetGenerator.generate(args[:numberOfTestingExamples], 0.0)
    puts "Test Examples:"
    puts testExamples
    return testExamples
  end

  def createNetworkAndTrainer
    classOfTheNetwork = args[:classOfTheNetwork]
    classOfTheTrainer = args[:classOfTheTrainer]

    network = classOfTheNetwork.new(args)
    theTrainer = classOfTheTrainer.new(examples, network, args)
    return network, theTrainer
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
    puts "#{lastEpoch}, #{trainingMSE}, #{testMSE}, #{startingTime}, #{endingTime}"
    return [lastEpoch, trainingMSE, testMSE, startingTime, endingTime]
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


