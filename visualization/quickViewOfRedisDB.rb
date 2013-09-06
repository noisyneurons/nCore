require_relative '../lib/core/SimulationDataStore'


experimentNumber =  $redis.get("experimentNumber")
puts "\nNext Experiment Number=\t #{experimentNumber}"

selectedData = SnapShotData.lookup { |q| q[:experimentNumber].gte(0).order(:desc).limit(5) }
unless (selectedData.empty?)
  puts "\n##################################################################################################################################################"

  puts "Number\tDescription\tLastEpoch\tTrainMSE\tTestMSE\tTime"
  puts
  selectedData.each do |aSelectedExperiment|
    aHash = SnapShotData.values(aSelectedExperiment)
    puts "#{aHash[:experimentNumber]}\t#{aHash[:descriptionOfExperiment]}\t#{aHash[:epochs]}\t#{aHash[:trainMSE]}\t#{aHash[:testMSE]}\t#{aHash[:time]}"
  end
  puts "################################################################################################################################################## \n\n"
end

arrayOfKeys = $redis.keys("*")
puts "Number of Keys in Redis database: #{arrayOfKeys.length}"