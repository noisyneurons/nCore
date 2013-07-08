require_relative 'Utilities'
require 'relix'
require 'yaml'

$redis = Redis.new
# $redis.flushdb

class FlockData
  include Relix
  attr_accessor :id, :experimentNumber, :epochs, :neuron, :exampleNumber, :netInputs, :flockErrors

  relix do
    primary_key :flockDataKey
    ordered :exampleNumber
    multi :epochs, order: :neuron, order: :exampleNumber
    multi :netInputs, order: :flockErrors
    multi :epoch_neuron_exampleNumber, on: %w(epochs neuron exampleNumber)
    # multi :neuron, order: :epochs

    multi :epochs, index_values: true
    multi :neuron, index_values: true
    multi :netInputs, index_values: true
    multi :flockErrors, index_values: true
    multi :exampleNumber, index_values: true
  end


  @@ID = 0

  def initialize(experimentNumber, epochs, neuron, exampleNumber, netInputs=1, flockErrors=1)
    @id = @@ID
    @@ID += 1
    @experimentNumber = experimentNumber
    @epochs = epochs
    @neuron = neuron
    @netInputs = netInputs
    @flockErrors = flockErrors
    @exampleNumber = exampleNumber
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

FlockData.new(3, 1, 1, 1, 10, 20)
FlockData.new(3, 1, 1, 2, 10, 20)
FlockData.new(3, 1, 1, 3, 10, 20)
FlockData.new(3, 1, 2, 1, 10, 21)
FlockData.new(3, 1, 2, 2, 10, 21)
FlockData.new(3, 1, 2, 3, 10, 21)
FlockData.new(3, 2, 1, 1, 10, 22)
FlockData.new(3, 2, 1, 2, 10, 22)
FlockData.new(3, 2, 1, 3, 10, 22)
#FlockData.new(3, 2, 2, 11, 23)
#FlockData.new(3, 3, 1, 11, 24)
#FlockData.new(3, 3, 2, 11, 25)
#FlockData.new(3, 3, 3, 11, 26)


p FlockData.lookup { |q| q[:epochs].eq(2) }

puts "####################################"


#p FlockData.lookup{|q| q[:epochs].lt(3)}

FlockData.lookup_values(:epochs).each do |aNumberOfEpochs|
  print "aNumberOfEpochs=\t#{aNumberOfEpochs}\t"
  someData = FlockData.lookup { |q| q[:epochs].eq(aNumberOfEpochs) }

  puts "someData=\t#{someData}"
  someData.each { |item| p FlockData.values(item) }
end

puts "####################################"

FlockData.lookup_values(:neuron).each do |aNeuron|
  print "aNeuron=\t#{aNeuron}\t"
  someData = FlockData.lookup { |q| q[:neuron].eq(aNeuron) }
  puts "someData=\t#{someData}"
end


puts "############ Include Example Numbers #############"


aryOfExampleNumbers = FlockData.lookup_values(:exampleNumber)


FlockData.lookup_values(:epochs).each do |aNumberOfEpochs|
  p "aNumberOfEpochs=\t#{aNumberOfEpochs}\t"
  aryOfExampleNumbers.each do |exampleNumber|
    someData = FlockData.lookup { |q| q[:epoch_neuron_exampleNumber].eq({epochs: aNumberOfEpochs, neuron: 1, exampleNumber: exampleNumber}) }

    someData.each { |item| p FlockData.values(item) }
    #someData.each { |item| p FlockData.values(item)["netInputs"].to_i }
  end
end
puts "####################################"


#ary = $redis.keys("*")
#ary.each { |item| p item }

