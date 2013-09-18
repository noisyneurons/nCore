### VERSION "nCore"
## ../nCore/bin/TLearningRateExperiments.rb
# Purpose:  To quantitatively explore the simplest clustering w/o supervision.
# This is a simplified and significantly reorganized version of 'Phase1Phase2MultiCycle.rb'

require_relative '../lib/core/Utilities'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NeuralPartsExtended'
# require_relative '../lib/core/ExampleImportanceMods'    # TODO Is this useful???  So far NOT!
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/plot/CorePlottingCode'
require_relative '../lib/core/SimulationDataStore'
require_relative '../lib/core/TrainingSequencingAndGrouping'
require_relative '../lib/core/Trainers.rb'
require_relative '../lib/core/CorrectionForRateAtWhichNeuronsGainChanges'

def createTrainingSet(args)
  include ExampleDistribution
  examples = []
  examples << {:inputs => [1.0, 1.0], :targets => [1.0], :exampleNumber => 0, :class => 1}
  examples << {:inputs => [1.0, 2.0], :targets => [1.0], :exampleNumber => 1, :class => 1}
  examples << {:inputs => [1.0, 3.0], :targets => [1.0], :exampleNumber => 2, :class => 1}
  examples << {:inputs => [1.0, 4.0], :targets => [1.0], :exampleNumber => 3, :class => 1}
  examples << {:inputs => [-1.0, -1.0], :targets => [0.0], :exampleNumber => 4, :class => 0}
  examples << {:inputs => [-1.0, -2.0], :targets => [0.0], :exampleNumber => 5, :class => 0}
  examples << {:inputs => [-1.0, -3.0], :targets => [0.0], :exampleNumber => 6, :class => 0}
  examples << {:inputs => [-1.0, -4.0], :targets => [0.0], :exampleNumber => 7, :class => 0}
  STDERR.puts "****************Incorrect Number of Examples Specified!! ************************" if (args[:numberOfExamples] != examples.length)
  return examples
end

#def displayAndPlotResults(args, dPrimes, dataStoreManager, lastEpoch,
#    lastTestingMSE, lastTrainingMSE, network, theTrainer, trainingSequence)
#  puts network
#  puts "Elapsed Time=\t#{theTrainer.elapsedTime}"
#  puts "\tAt Epoch #{trainingSequence.epochs}"
#  puts "\tAt Epoch #{lastEpoch}"
#  puts "\t\tThe Network's Training MSE=\t#{lastTrainingMSE}\t and TEST MSE=\t#{lastTestingMSE}\n"
#  puts "\t\t\tThe dPrime(s) at the end of training are: #{dPrimes}"
#
##############################  plotting and visualization....
#  plotMSEvsEpochNumber(network)
#end

class Experiment
  def setParameters

    numberOfExamples = 8
    randomNumberSeed = 0

    @args = {
        :experimentNumber => Experiment.number,
        :descriptionOfExperiment => descriptionOfExperiment,
        :rng => Random.new(randomNumberSeed),

        # training parameters re. Output Error
        :outputErrorLearningRate => 0.02,
        :minMSE => 0.0001,
        :maxNumEpochs => 4e3,

        # Network Architecture
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 0,
        :numberOfOutputNeurons => 1,
        :weightRange => 1.0,
        :typeOfLink => FlockingLink,

        # Training Set parameters
        :numberOfExamples => numberOfExamples,

        # Recording and database parameters
        :intervalForSavingNeuronData => 100,
        :intervalForSavingDetailedNeuronData => 1000,
        :intervalForSavingTrainingData => 100,

        # Flocking Parameters...
        :flockingLearningRate => -0.002,
        :maxFlockingIterationsCount => 2000, # 3800, # 2000,
        :maxAbsFlockingErrorsPerExample => 0.002, # 0.00000000000001, #0.002, # 0.005,   # 0.04 / numberOfExamples = 0.005

        :typeOfClusterer => DynamicClusterer,
        :numberOfClusters => 2,
        :m => 2.0,
        :numExamples => numberOfExamples,
        :exampleVectorLength => 1,
        :delta => 1e-2,
        :maxNumberOfClusteringIterations => 10,
        :symmetricalCenters => true, # if true, speed is negatively affected

        # Inner Numeric Constraints -- used to floating point under or overflow
        :floorToPreventOverflow => 1e-30
    }
  end
end

###################################### START of Main Learning  ##########################################
srand(0)
descriptionOfExperiment = "TLearningRateExperiments using correctionFactorForRateAtWhichNeuronsGainChanges"
experiment = Experiment.new(descriptionOfExperiment)
args = experiment.setParameters
args[:trainingSequence] = trainingSequence = TrainingSequence.new(args)

############################# create training set...
examples = createTrainingSet(args)


######################## Create Network....
network = SimpleFlockingNeuronNetwork.new(args) # TODO Currently need to insure that TrainingSequence.create has been called before network creation!!!

############################### train ...
theTrainer = TrainingSupervisor.new(examples, network, args)

startingTime = Time.now
lastEpoch, lastTrainingMSE, accumulatedAbsoluteFlockingErrors = theTrainer.train

puts network
puts "lastEpoch, lastTrainingMSE, accumulatedAbsoluteFlockingErrors"
puts lastEpoch, lastTrainingMSE, accumulatedAbsoluteFlockingErrors


#arrayOfNeuronsToPlot = network.outputLayer
#theTrainer.displayTrainingResults(arrayOfNeuronsToPlot)
## theTrainer.storeEndOfTrainingMeasures(lastEpoch, lastTrainingMSE, lastTestingMSE, accumulatedAbsoluteFlockingErrors)
#displayAndPlotResults(args, accumulatedAbsoluteFlockingErrors, dataStoreManager, lastEpoch, lastTestingMSE,



lastTestingMSE = nil
dataToStoreLongTerm = {:experimentNumber => Experiment.number, :descriptionOfExperiment => descriptionOfExperiment,
                       :network => network.to_s, :time => Time.now, :elapsedTime => (Time.now - startingTime),
                       :epochs => lastEpoch, :trainMSE => lastTrainingMSE, :testMSE => lastTestingMSE,
                       :accumulatedAbsoluteFlockingErrors => accumulatedAbsoluteFlockingErrors
}
SnapShotData.new(dataToStoreLongTerm)

puts "\n\n############ NeuronData #############"
keysToRecords = []
NeuronData.lookup_values(:epochs).each do |epochNumber|
  keysToRecords << NeuronData.lookup { |q| q[:experimentNumber_epochs_neuron].eq({experimentNumber: Experiment.number, epochs: epochNumber, neuron: 2}) }
end
records = keysToRecords.collect { |recordKey| NeuronData.values(recordKey) } unless (keysToRecords.empty?)
puts records

puts "\n\n############ DetailedNeuronData #############"
keysToRecords = []
DetailedNeuronData.lookup_values(:epochs).each do |epochNumber|
  DetailedNeuronData.lookup_values(:exampleNumber).each do |anExampleNumber|
    keysToRecords << DetailedNeuronData.lookup { |q| q[:experimentNumber_epochs_neuron_exampleNumber].eq({experimentNumber: Experiment.number, epochs: epochNumber, neuron: 2, exampleNumber: anExampleNumber}) }
   end
end
records = keysToRecords.collect { |recordKey| DetailedNeuronData.values(recordKey) } unless (keysToRecords.empty?)
puts records


puts "\n\n############ TrainingData #############"
keysToRecords = TrainingData.lookup { |q| q[:experimentNumber].eq({experimentNumber: Experiment.number}) }
records = keysToRecords.collect { |recordKey| TrainingData.values(recordKey) } unless (keysToRecords.empty?)
puts records
plotMSEvsEpochNumber(records)


puts "\n\n############ SnapShotData #############"
selectedData = SnapShotData.lookup { |q| q[:experimentNumber].gte(0).order(:desc).limit(5) }
unless (selectedData.empty?)
  puts
  puts "Number\tLastEpoch\tTrainMSE\t\tTestMSE\t\tTime\t\t\t\tDescription"
  selectedData.each do |aSelectedExperiment|
    aHash = SnapShotData.values(aSelectedExperiment)
    puts "#{aHash[:experimentNumber]}\t\t#{aHash[:epochs]}\t#{aHash[:trainMSE]}\t#{aHash[:testMSE]}\t\t#{aHash[:time]}\t\t\t\t#{aHash[:descriptionOfExperiment]}"
  end
end

TrainingData.deleteData(Experiment.number)
TrainingData.deleteEntireIndex!
NeuronData.deleteData(Experiment.number)
NeuronData.deleteEntireIndex!
DetailedNeuronData.deleteData(Experiment.number)
DetailedNeuronData.deleteEntireIndex!

experiment.save

