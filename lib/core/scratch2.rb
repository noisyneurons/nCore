def median(array)
  sorted = array.sort
  len = sorted.length
  return (sorted[(len - 1) / 2] + sorted[len / 2]) / 2.0
end



def valuesMeetingACriteriaAcrossExperiments(aMeasure, criteria, dataFromMultipleExperiments)
  observationsMeetingCriteria = []
  epochsRequiredToReachCriteriaValues = []
  dataFromMultipleExperiments.each do |anExperiment|
    measuresForExperiment = anExperiment.collect { |aRecord| aRecord[aMeasure] }
    observationMeetingCriteria = measuresForExperiment.send(criteria)
    observationsMeetingCriteria << observationMeetingCriteria

    index = measuresForExperiment.find_index(observationMeetingCriteria)
    epochNumbers = anExperiment.collect { |aRecord| aRecord[:epochs] }
    numberOfEpochsRequiredToReachCriteriaValue = epochNumbers[index]
    epochsRequiredToReachCriteriaValues << numberOfEpochsRequiredToReachCriteriaValue
  end
  return observationsMeetingCriteria, epochsRequiredToReachCriteriaValues
end


aMeasure = :mse

experiment1 = [{:mse => 1.0, :epochs => 0}, {:mse => 0.5, :epochs => 1}, {:mse => 1.5, :epochs => 2}]
experiment2 = [{:mse => 2.0, :epochs => 0}, {:mse => 2.5, :epochs => 1}, {:mse => 3.5, :epochs => 2}]

dataFromMultileExperiments = [experiment1, experiment2]

#mins, epochs =valuesMeetingACriteriaAcrossExperiments(aMeasure, :min, dataFromMultileExperiments)
#
#puts "mins, #{mins}"
#puts "epochs, #{epochs}"
#

def printableStatsForMetricAndCriteria(aMeasure, criteria, dataFromMultipleExperiments)
  observationsMeetingCriteria, epochsRequiredToReachCriteriaValues = valuesMeetingACriteriaAcrossExperiments(aMeasure, criteria, dataFromMultipleExperiments)

  aPrintableString = "\n\nThe Metric is '#{aMeasure}' and the Criteria is '#{criteria}'\n"

  theMedianAcrossAllExperiments = median(observationsMeetingCriteria)
  aPrintableString += "\nThe Median of the #{criteria} #{aMeasure}s across all experiments is = #{theMedianAcrossAllExperiments}\n"
  aPrintableString += "The list of all #{criteria} #{aMeasure}s:\n#{observationsMeetingCriteria}\n"

  theMedianAcrossAllExperiments = median(epochsRequiredToReachCriteriaValues)
  aPrintableString += "\nThe Median number of epochs required to reach #{criteria} #{aMeasure}s -- across all experiments is = #{theMedianAcrossAllExperiments}\n"
  aPrintableString += "The list of the number of epochs required to reach #{criteria} #{aMeasure}s:\n#{epochsRequiredToReachCriteriaValues}\n"
end

aString = printableStatsForMetricAndCriteria(aMeasure, :min, dataFromMultileExperiments)
puts aString