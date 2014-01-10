# '~/Code/Ruby/NN2012/nCore/visualization/cleanRedisDB.rb'

require_relative '../lib/core/Utilities'
require_relative '../lib/core/SimulationDataStore'

# $redis = Redis.new(:host => currentHost)

experimentNumber = $redis.get("experimentNumber")
puts "\nNext Experiment Number=\t #{experimentNumber}"

dataStore = SimulationDataStoreManager.new


lastExperimentForDeletion = 2039
#SnapShotData.deleteData(lastExperimentForDeletion)
#SnapShotData.deleteKey(lastExperimentForDeletion)

#ary = $redis.keys("SnapShotData*")
#ary.each { |item| $redis.del(item) }


#(1..lastExperimentForDeletion).each do |anExperimentNumber|
#  SnapShotData.deleteData(anExperimentNumber)
#  SnapShotData.deleteKey(anExperimentNumber)
#end

$redis.save

STDERR.puts "just after deleting early snapshot keys and data"

dataStore.deleteAllDataAndIndexesExceptSnapShot!

arrayOfKeys = $redis.keys("SSD*")
puts "Number of 'Snap Shot Keys' in Redis database: #{arrayOfKeys.length}"
puts "Snap Shot Keys in Redis database: #{arrayOfKeys}\n\n"


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