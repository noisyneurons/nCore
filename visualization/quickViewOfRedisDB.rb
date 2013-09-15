# require_relative '../lib/core/SimulationDataStore'

require 'redis'
require 'yaml'

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
puts "Next Experiment Number=\t #{experimentNumber}\n\n"

arrayOfKeys = $redis.keys("SSD*")
puts "Number of 'Snap Shot Keys' in Redis database: #{arrayOfKeys.length}"
puts "Snap Shot Keys in Redis database: #{arrayOfKeys}\n\n"

#arrayOfKeys.each do |aKey|
#  puts "\n\nSnap Shot Key:\t#{arrayOfKeys}\tValue=\t#{$redis.get(aKey)}"
#end

arrayOfKeys = $redis.keys("ND*")
puts "Number of 'NeuronData Keys' in Redis database: #{arrayOfKeys.length}"
puts "NeuronData Keys in Redis database: #{arrayOfKeys}\n\n"


arrayOfKeys = $redis.keys("DND*")
puts "Number of 'DetailedNeuronData Keys' in Redis database: #{arrayOfKeys.length}"
puts "NeuronData Keys in Redis database: #{arrayOfKeys}\n\n"


