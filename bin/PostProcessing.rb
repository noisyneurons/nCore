### VERSION "nCore"
## ../nCore/bin/CircleBPofFlockError.rb

require_relative 'BaseLearningExperiment'
require_relative '../lib/core/CorrectionForRateAtWhichNeuronsGainChanges'


puts "\n\n############ TrainingData #############"

#experimentNumbers = []

experimentNumbers = $redis.lrange("experimentsToPostProcess", 0, -1)

puts experimentNumbers

unless (experimentNumbers.empty?)
  experimentNumbers.each do |anExperimentNumber|
    keysToRecords = TrainingData.lookup { |q| q[:experimentNumber].eq({experimentNumber: anExperimentNumber}) }
    trainingDataRecords = nil
    unless (keysToRecords.empty?)
      keysToRecords.reject! { |recordKey| recordKey.empty? }
      trainingDataRecords = keysToRecords.collect { |recordKey| TrainingData.values(recordKey) }
    end
    #puts trainingDataRecords.class #[:epochs]
    trainingDataRecords.each {|aRecord| puts aRecord[:epochs]}
  end
end


unless (experimentNumbers.empty?)
  dataFromMultipleExperiments = []
  experimentNumbers.each do |anExperimentNumber|
    keysToRecords = TrainingData.lookup { |q| q[:experimentNumber].eq({experimentNumber: anExperimentNumber}) }
    trainingDataRecords = nil
    unless (keysToRecords.empty?)
      keysToRecords.reject! { |recordKey| recordKey.empty? }
      dataFromMultipleExperiments << keysToRecords.collect { |recordKey| TrainingData.values(recordKey) }
    end
  end

  minimums = []
  dataFromMultipleExperiments.each do |anExperiment|
    testMSEsForExperiment = anExperiment.collect {|aRecord| aRecord[:testMSE]}
    minTestMSE = testMSEsForExperiment.min
    minimums << minTestMSE
  end
  puts minimums
end


