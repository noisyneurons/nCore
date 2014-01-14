### VERSION "nCore"
## ../nCore/bin/PostProcessing.rb

require_relative 'BaseLearningExperiment'
#require_relative '../lib/core/CorrectionForRateAtWhichNeuronsGainChanges'

def median(array)
  sorted = array.sort
  len = sorted.length
  return (sorted[(len - 1) / 2] + sorted[len / 2]) / 2.0
end

def minimumsMaximumsAndLastValuesAcrossExperiments(aMeasure, dataFromMultipleExperiments)
  minimumsOfMeasure = []
  maximumsOfMeasure = []
  lastMeasurementsAcrossExperiments = []
  dataFromMultipleExperiments.each do |anExperiment|
    measuresForExperiment = anExperiment.collect { |aRecord| aRecord[aMeasure] }
    minimumsOfMeasure << measuresForExperiment.min
    maximumsOfMeasure << measuresForExperiment.max
    lastMeasurementsAcrossExperiments << measuresForExperiment.last
  end
  return lastMeasurementsAcrossExperiments, minimumsOfMeasure, maximumsOfMeasure
end


def printStatsForMetric(aMeasure, dataFromMultipleExperiments, fileOut)

  lastMeasurementsAcrossExperiments, minimumsOfMeasure, dummy = minimumsMaximumsAndLastValuesAcrossExperiments(aMeasure, dataFromMultipleExperiments)

  fileOut.puts "\n\nMEASURE = #{aMeasure} is as follows:\n"
  theMedianMinimumOfMeasure = median(minimumsOfMeasure)
  fileOut.puts "\nThe Median of the Minimums = #{theMedianMinimumOfMeasure}\n\n"
  fileOut.puts "minimums=\t#{minimumsOfMeasure}\n"
  fileOut.puts "last measurement for each experiment=\t#{lastMeasurementsAcrossExperiments}\n"
  ratios = []
  minimumsOfMeasure.each_with_index { |aMinimum, index| ratios << (aMinimum / lastMeasurementsAcrossExperiments[index]) }
  fileOut.puts "ratio of minimum measure to last measure =\t#{ratios}"
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

    [:mse, :testMSE].each do |aMeasure|
      printStatsForMetric(aMeasure, dataFromMultipleExperiments, fileOut)
    end
  end
end


