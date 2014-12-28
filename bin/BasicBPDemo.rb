### VERSION "nCore"
## ../nCore/bin/BasicBPDemo.rb
## Simple backprop demo. For XOR, and given parameters, requires 2080 epochs to converge.
##                       For OR, and given parameters, requires 166 epochs to converge.

require_relative '../lib/core/Utilities'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/core/TrainingBase'

require_relative '../lib/plot/CorePlottingCode'
require_relative '../lib/core/SimulationDataStore'

require_relative 'BaseLearningExperiment'

logger = StringIO.new

args = {
    :experimentNumber => $globalExperimentNumber,
    :descriptionOfExperiment => "Basic XOR Demo",
    :randomNumberSeed => 0,

    :classOfTheNetwork => Standard3LayerNetwork,
    :classOfTheTrainer => TrainerBase,
    :classOfDataSetGenerator => XORDataGenerator,

    # training parameters re. Output Error
    :learningRate => 3.0,
    :minMSE => 0.001,
    :trainingSequence => TrainingSequence,
    :maxNumEpochs => 2e3,

    # Network Architecture
    :numberOfInputNeurons => 2,
    :numberOfHiddenNeurons => 2,
    :numberOfOutputNeurons => 1,
    :weightRange => 1.0,

    :typeOfLink => Link,
    :typeOfNeuron => Neuron2,
    :typeOfOutputNeuron => OutputNeuron2,


    # Training Set parameters
    :numberOfExamples => 4,
    :numberOfTestingExamples => 4,

    # Results and debugging information storage/access
    :logger => logger

}

###################################### REPEATED Experiments for comparison ##########################################

numberOfRepetitions = 1


runner = ExperimentRunner.new(args)
lastExperimentRun, results  = runner.repeatSimulation(numberOfRepetitions)
logger.puts lastExperimentRun.network


loggedData = logger.string

$redis.rpush("SimulationList", loggedData)

retrievedData = $redis.rpoplpush("SimulationList", "SimulationList")

puts retrievedData

numberOfExperimentsStoredInList = $redis.llen("SimulationList")

puts "\n\nnumber Of Experiments Stored In List =\t#{numberOfExperimentsStoredInList}"