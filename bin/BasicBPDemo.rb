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

class Experiment

  def setParameters
    self.numberOfExamples = 4
    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => "Proj3SelfOrgContextSuper; 2 in 4 out; divide but do NOT INTEGRATE!",
        :randomNumberSeed => randomNumberSeed,

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
        #   :typeOfLinkToOutput => Link,


        # Training Set parameters
        :numberOfExamples => numberOfExamples,
        :numberOfTestingExamples => numberOfExamples,
    }
  end
end


###################################### START of REPEATED Experiments ##########################################

def repeatSimulation(numberOfReps = 1, randomSeedForSimulationSequence = 0)
  aryOfTrainingMSEs = []
  aryOfTestMSEs = []
  experiment = nil

  numberOfReps.times do |i|
    experimentsRandomNumberSeed = (i + randomSeedForSimulationSequence)
    experiment = Experiment.new(experimentsRandomNumberSeed)
    lastEpoch, trainingMSE, testMSE, startingTime, endingTime = experiment.performSimulation()
    aryOfTrainingMSEs << trainingMSE
    aryOfTestMSEs << testMSE
  end
  puts "\n\nmean TrainingMSE= #{aryOfTrainingMSEs.mean},\tmean TestingMSE= #{aryOfTestMSEs.mean}"
  return experiment
end

experiment = repeatSimulation
puts experiment.network

