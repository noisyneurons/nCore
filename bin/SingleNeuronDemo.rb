### VERSION "nCore"
## ../nCore/bin/SingleNeuronDemo.rb
# 313 epochs to converge

require_relative '../lib/core/Utilities'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/Layers'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/core/NeuronLearningStrategies'
require_relative '../lib/core/Trainers'

require_relative '../lib/plot/CorePlottingCode'
require_relative '../lib/core/SimulationDataStore'

require_relative 'BaseLearningExperiment'

logger = StringIO.new

args = {
    :experimentNumber => $globalExperimentNumber,
    :descriptionOfExperiment => "SingleNeuronDemo OR Demo",
    :randomNumberSeed => 0,

    :classOfTheNetwork => Simplest1LayerNet,
    :classOfTheTrainer => OneNeuronSelfOrgTrainer, #TrainerBase,
    :classOfDataSetGenerator => ORDataGenerator,

    # training parameters re. Output Error
    :learningRate => 3.0,
    :minMSE => 0.001,
    :epochsForSelfOrg => 0, #150,
    :epochsForSupervisedTraining => 500, # 600,
    :trainingSequence => TrainingSequence,

    # Network Architecture
    :numberOfInputNeurons => 2,
    :numberOfOutputNeurons => 1,
    :weightRange => 1.0,

    :typeOfLink => LinkWithNormalization,
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
lastExperimentRun, results = runner.repeatSimulation(numberOfRepetitions)
#logger.puts lastExperimentRun.network

loggedData = logger.string

$redis.rpush("SimulationList", loggedData)

retrievedData = $redis.rpoplpush("SimulationList", "SimulationList")

puts retrievedData

numberOfExperimentsStoredInList = $redis.llen("SimulationList")

puts "\n\nnumber Of Experiments Stored In List =\t#{numberOfExperimentsStoredInList}"