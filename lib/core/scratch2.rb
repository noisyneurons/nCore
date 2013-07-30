require_relative 'Utilities'
require 'relix'
require 'yaml'

$redis = Redis.new
# $redis.flushdb

class SnapShotData
  include Relix
  attr_accessor :id, :experimentNumber, :experimentDescription, :network, :time, :epochs, :trainMSE, :testMSE

  relix do
    primary_key :dataKey
    ordered :experimentNumber
    multi :experimentNumber_epochs, on: %w(experimentNumber epochs)
    # multi :experimentNumber, index_values: true
  end

  @@ID = 0

  def SnapShotData.deleteTable
    ary = $redis.keys("SSD*")
    ary.each { |item| $redis.del(item) }
    ary = $redis.keys("SnapShotData*")
    ary.each { |item| $redis.del(item) }
  end


  def initialize(experimentDescription, network, time, epochs, trainMSE, testMSE = 0.0)
    @id = @@ID
    @@ID += 1
    @experimentNumber = Experiment.number

    @experimentDescription = experimentDescription
    @network = network
    @time = time
    @epochs = epochs
    @trainMSE = trainMSE
    @testMSE = testMSE

    snapShotHash = {:experimentNumber => experimentNumber, :experimentDescription => experimentDescription,
                    :network => network,
                    :time => time,
                    :epochs => epochs,
                    :trainMSE => trainMSE,
                    :testMSE => testMSE
    }

    $redis.set(dataKey, YAML.dump(snapShotHash))
    index!
  end

  def dataKey
    "SSD#{experimentNumber}.#{id}"
  end

  def SnapShotData.values(key)
    YAML.load($redis.get(key))
  end
end


selectedData = SnapShotData.lookup { |q| q[:experimentNumber].gte(0).order(:desc).limit(5) }
unless (selectedData.empty?)
  puts
  puts "Number\tDescription\tLastEpoch\tTrainMSE\tTestMSE\tTime"
  selectedData.each do |aSelectedExperiment|
    aHash = SnapShotData.values(aSelectedExperiment)
    puts "#{aHash[:experimentNumber]}\t#{aHash[:descriptionOfExperiment]}\t#{aHash[:epochs]}\t#{aHash[:trainMSE]}\t#{aHash[:testMSE]}\t#{aHash[:time]}"
  end
end
