### VERSION "nCore"
## ../nCore/bin/CircleBPofFlockError.rb

require_relative 'BaseLearningExperiment'
require_relative '../lib/core/CorrectionForRateAtWhichNeuronsGainChanges'

puts "\n\n############ TrainingData #############"
experimentNumbers = $redis.lrange("experimentsToPostProcess", 0, -1)
puts experimentNumbers

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
  lastTestMSEs = []
  dataFromMultipleExperiments.each do |anExperiment|
    testMSEsForExperiment = anExperiment.collect {|aRecord| aRecord[:testMSE]}
    minimums << testMSEsForExperiment.min
    lastTestMSEs << testMSEsForExperiment.last
  end
  puts minimums
  puts lastTestMSEs
end


