require 'rubygems'
require 'sequel'
require 'singleton'


class SimulationDataStoreManager
  attr_accessor :db, :exampleDataSet, :epochDataSet, :tagDataSet, :exampleFeatureDataSet, :experimentDescriptionDataSet
  include Singleton

  def initialize
    @db = Sequel.sqlite('../../../data/acrossEpochsSequel.db')
    #@db = Sequel.sqlite # connect to an in-memory database

    @db.create_table! :epochs do
      Bignum :epochNumber
      Integer :neuronID
      primary_key [:epochNumber, :neuronID]
      Float :wt1
      Float :wt2
      Float :cluster0Center
      Float :cluster1Center
      Float :dPrime
    end
    @epochDataSet = @db[:epochs]

    @db.create_table! :examples do
      Bignum :epochNumber
      Integer :neuronID
      Integer :exampleNumber
      primary_key [:epochNumber, :neuronID, :exampleNumber]
      Float :netInput
      Float :error
      Float :bpError
      Float :localFlockingError
      Float :weightedErrorMetric
    end
    @exampleDataSet = @db[:examples]

    @db.create_table! :tags do
      Bignum :epochNumber, :primary_key => true
      String :learningPhase
      Integer :epochsSinceBeginningOfPhase
      Integer :epochsSinceBeginningOfCycle
      String :note
    end
    @tagDataSet = @db[:tags]

    @db.create_table! :exampleFeatures do
      Integer :exampleNumber, :primary_key => true
      Integer :class
      Integer :feature1
      Integer :feature2
    end
    @exampleFeatureDataSet = @db[:exampleFeatures]

    @db.create_table! :experimentDescriptions do
      primary_key :experimentNumber
      String :description, :text => true
      DateTime :dateTime
    end
    @experimentDescriptionDataSet = @db[:experimentDescriptions]
    postInitialize
  end

  def postInitialize
    examples = createMultiClassTrainingSet(numberOfExamples=16)
    examples.each do |anExample|
      exampleFeatureDataSet.insert(:exampleNumber => anExample[:exampleNumber], :class => anExample[:class])
    end

    descriptionOfExperiment = "This is a quick test of this database process..."
    experimentDescriptionDataSet.insert(:description => descriptionOfExperiment, :dateTime => Time.now)
  end

  def joinDataSets
    aJoinedDS = db["SELECT * FROM examples NATURAL JOIN exampleFeatures NATURAL JOIN epochs"]
  end
end

simDataStoreManager = SimulationDataStoreManager.instance

epR = simDataStoreManager.epochDataSet
epR.insert(:epochNumber => 1, :neuronID => 1, :dPrime => rand)
epR.insert(:epochNumber => 1, :neuronID => 2, :dPrime => rand)
epR.insert(:epochNumber => 1, :neuronID => 3, :dPrime => rand)
puts "epochs record count: #{epR.count}"
puts "The average dprime is: #{epR.avg(:dPrime)}"


exR = simDataStoreManager.exampleDataSet
exR.insert(:neuronID => 1, :epochNumber => 1, :exampleNumber => 1, :netInput => 0.33)
exR.insert(:neuronID => 2, :epochNumber => 1, :exampleNumber => 2, :netInput => 0.63)
exR.insert(:neuronID => 3, :epochNumber => 1, :exampleNumber => 3, :netInput => 0.93)
exR.insert(:neuronID => 1, :epochNumber => 1, :exampleNumber => 4, :netInput => 0.33)
exR.insert(:neuronID => 2, :epochNumber => 1, :exampleNumber => 5, :netInput => 0.63)
exR.insert(:neuronID => 3, :epochNumber => 1, :exampleNumber => 6, :netInput => 0.93)
exR.insert(:neuronID => 1, :epochNumber => 1, :exampleNumber => 7, :netInput => 0.33)
exR.insert(:neuronID => 2, :epochNumber => 1, :exampleNumber => 8, :netInput => 0.63)
exR.insert(:neuronID => 3, :epochNumber => 1, :exampleNumber => 9, :netInput => 0.93)
puts "examples count: #{exR.count}"
puts "The average neuron number is: #{exR.avg(:neuronID)}"


exF = simDataStoreManager.exampleFeatureDataSet

def createMultiClassTrainingSet(numberOfExamples, rightShiftUpper2Classes = 0.0)

  xStart = [-1.0+rightShiftUpper2Classes, 1.0+rightShiftUpper2Classes, 1.0, -1.0] # assumes clockwise numbering of classes, from 10:30 being class 0
  yStart = [1.0, 1.0, -1.0, -1.0]


  xInc = [0.0, 0.0, 0.0, 0.0]
  yInc = [1.0, 1.0, -1.0, -1.0]

  # target = [0.9, 0.9, 0.9, 0.9] # potentially will use this example set for supervised learning.

  numberOfClasses = xStart.length
  numberOfExamplesInEachClass = numberOfExamples / numberOfClasses
  exampleNumber = 0

  examples = []
  numberOfClasses.times do |classOfExample|
    xS = xStart[classOfExample]
    xI = xInc[classOfExample]
    yS = yStart[classOfExample]
    yI = yInc[classOfExample]
    numberOfExamplesInEachClass.times do |classExNumb|
      x = xS + (xI * classExNumb)
      y = yS + (yI * classExNumb)
      aPoint = [x, y]
      targets = [0.0, 0.0, 0.0, 0.0]
      targets[classOfExample] = 1.0
      examples << {:inputs => aPoint, :targets => targets, :exampleNumber => exampleNumber, :class => classOfExample}
      exampleNumber += 1
    end
  end
  STDERR.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
  examples
end

examples = createMultiClassTrainingSet(numberOfExamples=16)
examples.each do |anExample|
  exF.insert(:exampleNumber => anExample[:exampleNumber], :class => anExample[:class])
end

expDesc = simDataStoreManager.experimentDescriptionDataSet
descriptionOfExperiment = "This is a quick test of this database process..."
expDesc.insert(:description => descriptionOfExperiment, :dateTime => Time.now)

aJoinedDS = simDataStoreManager.joinDataSets

aJoinedDS.each_with_index do |r, aCount|
  puts "r\t#{r.to_hash}  \tCount=\t#{aCount}"
end
