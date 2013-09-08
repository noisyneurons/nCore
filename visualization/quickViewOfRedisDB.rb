# require_relative '../lib/core/SimulationDataStore'

require 'redis'
require 'yaml'

theComputersName = Socket.gethostname

currentHost = "localhost"
currentHost = "ec2-54-224-56-245.compute-1.amazonaws.com" # master external dns
currentHost = "ec2-107-20-13-47.compute-1.amazonaws.com" # node001 external dns

currentHost = "ip-10-145-223-204.ec2.internal" # master internal dns
currentHost = "ip-10-164-60-198.ec2.internal" # node001 internal dns

currentHost = "master"
currentHost = "node001"

$redis = Redis.new(:host => currentHost)

experimentNumber =  $redis.get("experimentNumber")
puts "\nNext Experiment Number=\t #{experimentNumber}"

arrayOfKeys = $redis.keys("SSD*")
puts "Number of Keys in Redis database: #{arrayOfKeys.length}"

puts "\n\nSnap Shot Keys in Redis database: #{arrayOfKeys}"

#arrayOfKeys.each do |aKey|
#  puts "\n\nSnap Shot Key:\t#{arrayOfKeys}\tValue=\t#{$redis.get(aKey)}"
#end


