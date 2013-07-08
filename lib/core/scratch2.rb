require_relative 'Utilities'

require 'relix'
require 'yaml'

$redis = Redis.new
$redis.flushdb

class FlockData
  include Relix
  attr_accessor :id, :experimentNumber, :epochs, :neuron, :netInputs, :flockErrors

  relix do
    primary_key :flockDataKey
    #ordered :epochs
    multi :epochs, order: :neuron
    multi :netInputs, order: :flockErrors
  end

  relix.multi :epochs, index_values: true
  relix.multi :neuron, index_values: true
  relix.multi :netInputs, index_values: true
  relix.multi :flockErrors, index_values: true


  @@ID = 0

  def initialize(experimentNumber, epochs, neuron, netInputs=1, flockErrors=1)
    @id = @@ID
    @@ID += 1
    @experimentNumber = experimentNumber
    @epochs = epochs
    @neuron = neuron
    @netInputs = netInputs
    @flockErrors = flockErrors
    index!
  end

  def flockDataKey
     "#{experimentNumber}.#{id}"
  end

  def FlockData.values(key)
    redisKey = "FlockData:values:#{key}"
    $redis.hgetall(redisKey)
    # $redis.keys("*")
  end
end

FlockData.new(3, 1, 1, 10, 20)
FlockData.new(3, 1, 2, 10, 21)
FlockData.new(3, 2, 1, 10, 22)
FlockData.new(3, 2, 2, 11, 23)
FlockData.new(3, 3, 1, 11, 24)
FlockData.new(3, 3, 2, 11, 25)
FlockData.new(3, 3, 3, 11, 26)


p FlockData.lookup { |q| q[:epochs].eq(2) }

puts "####################################"


#p FlockData.lookup{|q| q[:epochs].lt(3)}

FlockData.lookup_values(:epochs).each do | aNumberOfEpochs |
  print "aNumberOfEpochs=\t#{aNumberOfEpochs}\t"
  someData = FlockData.lookup{|q| q[:epochs].eq(aNumberOfEpochs)}

  puts "someData=\t#{someData}"
  someData.each {|item| p FlockData.values(item)}
end

puts "####################################"

FlockData.lookup_values(:neuron).each do | aNeuron |
  print "aNeuron=\t#{aNeuron}\t"
  someData = FlockData.lookup{|q| q[:neuron].eq(aNeuron)}
  puts "someData=\t#{someData}"
end




ary = $redis.keys("*")
ary.each {|item| p item}

