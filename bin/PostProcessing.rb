### VERSION "nCore"
## ../nCore/bin/PostProcessing.rb

require_relative 'BaseLearningExperiment'
require_relative '../lib/core/CorrectionForRateAtWhichNeuronsGainChanges'

def median(array)
  sorted = array.sort
  len = sorted.length
  return (sorted[(len - 1) / 2] + sorted[len / 2]) / 2.0
end

keysToLastRecords = SnapShotData.lookup { |q| q[:experimentNumber].gte(0).order(:desc).limit(1) }
lastSnapShotDataRecord = SnapShotData.values(keysToLastRecords[0])
descriptionOfLastExperiment = lastSnapShotDataRecord[:descriptionOfExperiment]
jobName = descriptionOfLastExperiment[0...10]
#  jobName = "Job2ForCir"

experimentNumbers = $redis.lrange("#{jobName}List", 0, -1)

filename = "#{Dir.home}/Code/Ruby/NN2012/analysisResults/PostProcessing_#{jobName}_ExpNum#{experimentNumbers.first}"
File.open(filename, "w") do |fileOut|
  fileOut.puts "\n\n############ Analysis of Simulation Output #############"

  fileOut.puts "jobName=\t #{jobName}"
  fileOut.puts experimentNumbers

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

      fileOut.puts "\n\nMEASURE = #{anMSEMeasure} is as follows:\n"
      theMedianMinimumMSE = median(minimumsOfTestMSEs)
      fileOut.puts "\nThe Median Minimum MSE = #{theMedianMinimumMSE}\n\n"
      fileOut.puts "minimumsMSEs=\t#{minimumsOfTestMSEs}\n"
      fileOut.puts "lastMSEs=\t#{lastTestMSEs}\n"
      ratios = []
      minimumsOfTestMSEs.each_with_index { |aMinimum, index| ratios << (aMinimum / lastTestMSEs[index]) }
      fileOut.puts "ratio of minimum to last mse =\t#{ratios}"
    end
  end
end


