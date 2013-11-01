# '~/Code/Ruby/NN2012/nCore/visualization/cleanRedisDB.rb'

require_relative '../lib/core/Utilities'

$redis = Redis.new(:host => currentHost)

experimentNumber = $redis.get("experimentNumber")
puts "\nNext Experiment Number=\t #{experimentNumber}"

dataStore = SimulationDataStoreManager.new
dataStore.deleteAllDataAndIndexesExceptSnapShot!


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
puts "Remaining Keys in Redis database after selective deletion: #{arrayOfKeys}"

## --- DANGER ----###
## puts redis.flushdb