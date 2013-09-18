# '~/Code/Ruby/NN2012/nCore/visualization/cleanRedisDB.rb'

require_relative '../lib/core/SimulationDataStore'

theComputersName = Socket.gethostname

currentHost = "ec2-54-224-56-245.compute-1.amazonaws.com" # master external dns
currentHost = "ec2-107-20-13-47.compute-1.amazonaws.com" # node001 external dns

currentHost = "ip-10-145-223-204.ec2.internal" # master internal dns
currentHost = "ip-10-164-60-198.ec2.internal" # node001 internal dns

currentHost = "master"
currentHost = "node001"
currentHost = "localhost"

$redis = Redis.new(:host => currentHost)

experimentNumber =  $redis.get("experimentNumber")
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