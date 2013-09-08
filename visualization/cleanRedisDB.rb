require_relative '../lib/core/SimulationDataStore'

# Notes: PutLoveInYourHeart (:host => "192.168.1.128", :port => 8765)
# Notes: MakeASadSongMuchBetter (:host => "192.168.1.127", :port => 8765) Wired
# Notes: MakeASadSongMuchBetter (:host => "192.168.1.131", :port => 8765) Wireless

# redis = Redis.new(host => "localhost", :port => 8765)
redis = Redis.new

dataStore = SimulationDataStoreManager.create
dataStore.deleteTemporaryTables

experimentNumber =  redis.get("experimentNumber")
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

arrayOfKeys = redis.keys("*")
puts "Remaining Keys in Redis database after selective deletion: #{arrayOfKeys}"