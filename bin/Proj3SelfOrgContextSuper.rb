### VERSION "nCore"
## ../nCore/bin/Proj3SelfOrgContextSuper.rb

# Specific Purpose for this experiment: Get SIMPLEST versions of self-org, context, AND combined with Supervised Learning,  understood and "working."
# Purpose:  Start of Project 7; project to split example set to learn sub-parts, and then combine those parts/neuron-functions that
# didn't need to be separated, but instead need to be integrated to obtain better generalization.
# Ultimate goal of project 6 is develop analogy processing -- where one function useful for solving one problem
# can be of use in solving another problem.  The common function(s)/neuron(s) can be thus be 'reused' -- and even potentially made
# better by improving the accuracy of the function parameters because more examples are used to learn the parameters.

require_relative '../lib/core/Utilities'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NeuralParts2'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/core/NetworkFactories2'
require_relative '../lib/core/Trainers'
require_relative '../lib/core/Trainers2'

require_relative '../lib/plot/CorePlottingCode'
require_relative '../lib/core/SimulationDataStore'

require_relative 'BaseLearningExperiment'
########################################################################

class Neuron2
  include NonMonotonicIOFunction
end

class OutputNeuron2
  include NonMonotonicIOFunction
end

class Experiment

  def setParameters
    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,
        :randomNumberSeed => (randomNumberSeed + 0),

        # training parameters
        :learningRate =>  0.1,
        :minMSE => 0.0, # 0.001,
        :maxEpochNumbersForEachPhase => [1, 150, 1, 150, 600],
        :trainingSequence =>  MultiPhaseTrainingSequence,

        # Network Architecture
        :numberOfInputNeurons => 2,
        :numberOfHiddenLayer1Neurons => 1,
        :numberOfHiddenLayer2Neurons => 2,
        :numberOfOutputNeurons => 4,

        :weightRange => 0.1,

        :typeOfLink => LinkWithNormalization,
        :typeOfNeuron => Neuron2,
        :typeOfLinkToOutput => LinkWithNormalization,
        :typeOfOutputNeuron => OutputNeuron2,

        # Training Set parameters
        :numberOfExamples => (self.numberOfExamples = 16),
        :numberOfTestingExamples => numberOfExamples,
        :standardDeviationOfAddedGaussianNoise => 1e-24,
        :angleOfClockwiseRotationOfInputData => 0.0
    }
  end

  def createDataSet
    gen4ClassDS
  end

  def createNetworkAndTrainer
    network = Context4LayerNetwork.new(args)

    #temporarilySetSpecificWeights(network)

    theTrainer = Trainer3SelfOrgContextSuper.new(examples, network, args)

    return network, theTrainer
  end

end

###################################### START of Main Learning  ##########################################

baseRandomNumberSeed = 0

experiment = Experiment.new("Proj3SelfOrgContextSuper; 2 in 4 out; divide then integrate", baseRandomNumberSeed)

experiment.performSimulation()

puts experiment.network