### VERSION "nCore"
## ../nCore/bin/CircleBPofFlockError.rb

require_relative 'BaseLearningExperiment'
require_relative '../lib/core/CorrectionForRateAtWhichNeuronsGainChanges'

puts "\n\n############ TrainingData #############"

keysToLastRecords = SnapShotData.lookup { |q| q[:experimentNumber].gte(0).order(:desc).limit(1) }
lastSnapShotDataRecord = SnapShotData.values(keysToLastRecords[0])
descriptionOfLastExperiment = lastSnapShotDataRecord[:descriptionOfExperiment]
jobName = descriptionOfLastExperiment[0...10]
puts "jobName=\t #{jobName}"

experimentNumbers = $redis.lrange("#{jobName}List", 0, -1)
puts experimentNumbers

unless (experimentNumbers.empty?)
  dataFromMultipleExperiments = []
  experimentNumbers.each do |anExperimentNumber|
    keysToRecords = TrainingData.lookup { |q| q[:experimentNumber].eq({experimentNumber: anExperimentNumber}) }
    unless (keysToRecords.empty?)
      keysToRecords.reject! { |recordKey| recordKey.empty? }
      dataFromMultipleExperiments << keysToRecords.collect { |recordKey| TrainingData.values(recordKey) }
    end
  end

  [:mse, :testMSE].each do |anMSEMeasure|

    minimumsOfTestMSEs = []
    lastTestMSEs = []
    dataFromMultipleExperiments.each do |anExperiment|
      testMSEsForExperiment = anExperiment.collect { |aRecord| aRecord[anMSEMeasure] }
      minimumsOfTestMSEs << testMSEsForExperiment.min
      lastTestMSEs << testMSEsForExperiment.last
    end

    puts "\n\nMEASURE = #{anMSEMeasure} is as follows:\n\n"
    puts "minimumsMSEs=\t#{minimumsOfTestMSEs}"
    puts "lastMSEs=\t#{lastTestMSEs}"
    ratios = []
    minimumsOfTestMSEs.each_with_index { |aMinimum, index| ratios << (aMinimum / lastTestMSEs[index]) }
    puts "ratio of minimum to last mse =\t#{ratios}"
  end
end


