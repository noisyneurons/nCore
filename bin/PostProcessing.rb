### VERSION "nCore"
## ../nCore/bin/PostProcessing.rb

require_relative 'BaseLearningExperiment'

def median(array)
  sorted = array.sort
  len = sorted.length
  return (sorted[(len - 1) / 2] + sorted[len / 2]) / 2.0
end

def valuesMeetingACriteriaAcrossExperiments(aMeasure, criteria, dataFromMultipleExperiments)
  observationsMeetingCriteria = []
  epochsRequiredToReachCriteriaValues = []
  indexesInExperimentalData = []
  dataFromMultipleExperiments.each do |anExperiment|
    measuresForExperiment = anExperiment.collect { |aRecord| aRecord[aMeasure] }
    observationMeetingCriteria = measuresForExperiment.send(criteria)
    observationsMeetingCriteria << observationMeetingCriteria

    index = measuresForExperiment.find_index(observationMeetingCriteria)
    epochNumbers = anExperiment.collect { |aRecord| aRecord[:epochs] }
    numberOfEpochsRequiredToReachCriteriaValue = epochNumbers[index]
    epochsRequiredToReachCriteriaValues << numberOfEpochsRequiredToReachCriteriaValue
    indexesInExperimentalData << index
  end
  return observationsMeetingCriteria, epochsRequiredToReachCriteriaValues, indexesInExperimentalData
end

def printableStatsForMetricAndCriteria(aMeasure, criteria, dataFromMultipleExperiments)
  observationsMeetingCriteria, epochsRequiredToReachCriteriaValues, dummy = valuesMeetingACriteriaAcrossExperiments(aMeasure, criteria, dataFromMultipleExperiments)

  aPrintableString = "\n\nThe Metric is '#{aMeasure}' and the Criteria is '#{criteria}'\n"

  theMedianAcrossAllExperiments = median(observationsMeetingCriteria)
  aPrintableString += "\nThe Median of the #{criteria} #{aMeasure}s across all experiments is = #{theMedianAcrossAllExperiments}\n"
  aPrintableString += "The list of all #{criteria} #{aMeasure}s:\n#{observationsMeetingCriteria}\n"

  theMedianAcrossAllExperiments = median(epochsRequiredToReachCriteriaValues)
  aPrintableString += "\nThe Median number of epochs required to reach #{criteria} #{aMeasure}s -- across all experiments is = #{theMedianAcrossAllExperiments}\n"
  aPrintableString += "The list of the number of epochs required to reach #{criteria} #{aMeasure}s:\n#{epochsRequiredToReachCriteriaValues}\n"
  aPrintableString += "__________________________________________________________________________"
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
  dataFromMultipleExperiments = []
  unless (experimentNumbers.empty?)
    experimentNumbers.each do |anExperimentNumber|
      keysToRecords = TrainingData.lookup { |q| q[:experimentNumber].eq({experimentNumber: anExperimentNumber}) }
      unless (keysToRecords.empty?)
        keysToRecords.reject! { |recordKey| recordKey.empty? }
        dataFromMultipleExperiments << keysToRecords.collect { |recordKey| TrainingData.values(recordKey) }
      end
    end

    [:mse, :testMSE].each do |aMeasure|
      [:min].each do |aCriteria|
        aString = printableStatsForMetricAndCriteria(aMeasure, aCriteria, dataFromMultipleExperiments)
        fileOut.puts aString
      end
    end
  end
end


