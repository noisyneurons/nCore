# '~/Code/Ruby/NN2012/nCore/visualization/quickViewOfRedisDB.rb'
# This program gives you a good idea of the contents of the redis database.

require_relative '../lib/core/SimulationDataStore'

experimentNumber = $redis.get("experimentNumber")
logger.puts "\nLast Experiment Number=\t #{experimentNumber}\n\n"

arrayOfKeys = $redis.keys("SnapShotData*")
logger.puts "Number of 'Snap Shot db Keys' in Redis database: #{arrayOfKeys.length}"
logger.puts "Snap Shot db Keys in Redis database: #{arrayOfKeys}\n\n"

arrayOfKeys = $redis.keys("SSD*")
logger.puts "Number of 'SSD Data Keys' in Redis database: #{arrayOfKeys.length}"
logger.puts "SSD Keys in Redis database: #{arrayOfKeys}\n\n"

arrayOfKeys = $redis.keys("ND*")
logger.puts "Number of 'ND data Keys' in Redis database: #{arrayOfKeys.length}"
logger.puts "NeuronData Keys in Redis database: #{arrayOfKeys}\n\n"

arrayOfKeys = $redis.keys("DND*")
logger.puts "Number of 'DND data Keys' in Redis database: #{arrayOfKeys.length}"
logger.puts "DetailedNeuronData Keys in Redis database: #{arrayOfKeys}\n\n"

arrayOfKeys = $redis.keys("TrainingData*")
logger.puts "Number of 'TrainingData db Keys' in Redis database: #{arrayOfKeys.length}"
logger.puts "TrainingData db Keys in Redis database: #{arrayOfKeys}\n\n"

arrayOfKeys = $redis.keys("TD*")
logger.puts "Number of 'TD Data Keys' in Redis database: #{arrayOfKeys.length}"
logger.puts "TD Data Keys in Redis database: #{arrayOfKeys}\n\n"

arrayOfKeys = $redis.keys("*")
logger.puts "Number of Keys in Redis database: #{arrayOfKeys.length}"
logger.puts "Keys in Redis database: #{arrayOfKeys}\n\n"


#theComputersName = Socket.gethostname
#
#currentHost = "ec2-54-224-56-245.compute-1.amazonaws.com" # master external dns
#currentHost = "ec2-107-20-13-47.compute-1.amazonaws.com" # node001 external dns
#
#currentHost = "ip-10-145-223-204.ec2.internal" # master internal dns
#currentHost = "ip-10-164-60-198.ec2.internal" # node001 internal dns

#currentHost = "master"
#currentHost = "node001"
#currentHost = "localhost"


