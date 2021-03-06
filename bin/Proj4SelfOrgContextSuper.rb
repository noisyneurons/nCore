### VERSION "nCore"
## ../nCore/bin/Proj4SelfOrgContextSuper.rb

# Specific Purpose for this experiment: Get NEXT simplest versions of self-org, context, AND combined with Supervised Learning, understood and "working."
#   In this 4th project, we append a "relearning or re- self-org WITHOUT CONTEXT" just prior to the bp supervised training.
#   Presumably this will improve the accuracy of the hyperplanes in the 2nd hidden layer, compared to those in proj 3.

# General Purpose:  Start of Project 4; project to split example set to learn sub-parts, and then combine those parts/neuron-functions that
# didn't need to be separated, but instead need to be integrated to obtain better generalization.
# Ultimate goal is develop analogy processing -- where one function useful for solving one problem
# can be of use in solving another "similar-but-different" problem.  The common function(s)/neuron(s) can be thus be 'reused' -- and even potentially made
# better by improving the accuracy of the function parameters because more examples are used to learn the parameters. ala Bayes

require_relative '../lib/core/Utilities'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/Layers'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/core/NetworkFactories2'
require_relative '../lib/core/NeuronLearningStrategies'
require_relative '../lib/core/Trainers'

require_relative '../lib/plot/CorePlottingCode'

require_relative 'BaseLearningExperiment'
########################################################################

logger = StringIO.new

args = {
    :experimentNumber => $globalExperimentNumber,
    :descriptionOfExperiment => "Proj4SelfOrgContextSuper; 2 in 4 out; divide then integrate",
    :baseRandomNumberSeed => 0,

    :classOfTheNetwork => Context4LayerNetworkVer1,
    :classOfTheTrainer => Trainer4SelfOrgContextSuper,
    :classOfDataSetGenerator => Generate4ClassDataSet,

    # training parameters
    :learningRate => 0.1,
    :minMSE => 0.0,
    :epochsForSelfOrg => 150, #150,
    :epochsForSupervisedTraining => 600, # 600,
    :trainingSequence => TrainingSequence,

    # Network Architecture
    :numberOfInputNeurons => 2,
    :numberOfHiddenLayer1Neurons => 1,
    :numberOfHiddenLayer2Neurons => 2,
    :numberOfOutputNeurons => 4,

    # Neural Parts Specifications
    :typeOfLink => LinkWithNormalization,
    :typeOfNeuron => Neuron2,
    :typeOfLinkToOutput => LinkWithNormalization,
    :typeOfOutputNeuron => OutputNeuron2,

    :weightRange => 0.1,

    # Training Set parameters
    :numberOfExamples => 16,
    :numberOfTestingExamples => 160,
    :standardDeviationOfAddedGaussianNoise => 0.0,
    :verticalShift => 0.0,
    :horizontalShift => 0.0,
    :angleOfClockwiseRotationOfInputData => 0.0,

    # Results and debugging information storage/access
    :logger => logger
}


###################################### REPEATED Experiments for comparison ##########################################

numberOfRepetitions = 1

runner = ExperimentRunner.new(args)
lastExperimentRun, results = runner.repeatSimulation(numberOfRepetitions)
# logger.puts lastExperimentRun.network

loggedData = logger.string

$redis.rpush("SimulationList", loggedData)
$redis.save

retrievedData = $redis.rpoplpush("SimulationList", "SimulationList")

puts retrievedData

numberOfExperimentsStoredInList = $redis.llen("SimulationList")

puts "\n\nnumber Of Experiments Stored In List =\t#{numberOfExperimentsStoredInList}"