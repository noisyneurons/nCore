### VERSION "nCore"
## ../nCore/test/TestSequencers.rb
# unit testing is done on the Sequencer code

require 'test/unit'
require 'minitest/unit'
require 'minitest/mock'

require_relative '../lib/core/Sequencers'

Tolerance = 0.00001

############################################################
##### Temp Class Modifications, Test Stubbing,  ############
#####and any other Misc Fixtures... #######################

class DummyNeuron
end

class DummyLearner
  attr_accessor :propCount, :startCount, :finishCount

  def initialize
    @propCount = 0
    @startCount = 0
    @finishCount = 0
  end

  def startBatch
    self.startCount += 1
  end

  def propagate
    self.propCount += 1
  end

  def finishBatch
    self.finishCount += 1
  end
end

class DummyNetwork
  attr_accessor :allNeuronLayers, :allLearnerLayers, :calcCount,
                :calcCountRequiredToDropMSETo0, :networkMeanSquaredError

  def initialize
    @calcCount = 0
    @calcCountRequiredToDropMSETo0 = 1
    @networkMeanSquaredError = 9999999.0
  end

  def calcNetworksMeanSquareError
    self.calcCount += 1
    self.networkMeanSquaredError = 0.0 if (calcCount >= calcCountRequiredToDropMSETo0)
  end
end

class LearningSequencer
  public :processOneExample, :executeCommand
end

############################################################
class TestSequencers1 < MiniTest::Unit::TestCase
  def setup
    @args = {}
    @arrayOfNeurons = [DummyNeuron.new, DummyNeuron.new]
    @aSecondArrayOfNeurons = [DummyNeuron.new, DummyNeuron.new, DummyNeuron.new]
    @anArrayOfArraysOfNeurons = [@arrayOfNeurons, @aSecondArrayOfNeurons]

    @arrayOfLearners = [DummyLearner.new, DummyLearner.new]
    @aSecondArrayOfLearners = [DummyLearner.new, DummyLearner.new, DummyLearner.new]
    @anArrayOfArraysOfLearners = [@arrayOfLearners, @aSecondArrayOfLearners]

    @aNetwork = DummyNetwork.new
    @aNetwork.allNeuronLayers = @anArrayOfArraysOfNeurons
    @aNetwork.allLearnerLayers = @anArrayOfArraysOfLearners
    @errorCriteria=0.001
    @numberOfExamples=4
    @commandSequencePerExample = [[:propagate, @arrayOfLearners]]
    #@commandSequencePerExample = nil
    @aSequencer = LearningSequencer.new(@aNetwork, @errorCriteria, @numberOfExamples, @commandSequencePerExample, @args)
  end

  def test_executeCommand
    command = :propagate
    learnersThatShouldReceiveCommand = @arrayOfLearners
    @aSequencer.executeCommand(command, learnersThatShouldReceiveCommand)
    @arrayOfLearners.each do |aLearner|
      assert_equal(1, aLearner.propCount, "a learner was not issued command")
    end
  end

  def test_processOneExample1
    @aSequencer.processOneExample
    @arrayOfLearners.each do |aLearner|
      assert_equal(1, aLearner.propCount, "a learner was not issued command")
    end
  end

  def test_processOneExample2
    aCommandSequencePerExample = [[:propagate, @arrayOfLearners],
                                  [:propagate, @arrayOfLearners]
    ]
    @aSequencer.commandSequencePerExample= aCommandSequencePerExample
    @aSequencer.processOneExample
    @arrayOfLearners.each do |aLearner|
      assert_equal(2, aLearner.propCount, "a learner was not issued command")
    end
  end

  def test_processOneExample3
    aCommandSequencePerExample = [[:propagate, @arrayOfLearners],
                                  [:propagate, @arrayOfLearners],
                                  [:finishBatch, @arrayOfLearners]
    ]
    @aSequencer.commandSequencePerExample= aCommandSequencePerExample
    @aSequencer.processOneExample
    @arrayOfLearners.each do |aLearner|
      assert_equal(2, aLearner.propCount, "a learner was not issued command")
      assert_equal(1, aLearner.finishCount, "a learner was not issued the finishBatch")
    end
  end

  def test_processOneEpoch
    @aSequencer.processOneEpoch
    @arrayOfLearners.each do |aLearner|
      assert_equal(1, aLearner.startCount, "a learner was not issued the finishBatch")
      assert_equal(4, aLearner.propCount, "a learner was not issued the proper number of 'propagate' command")
      assert_equal(1, aLearner.finishCount, "a learner was not issued the finishBatch")
    end
    assert_equal(1, @aNetwork.calcCount, "a learner was not issued the finishBatch")
  end

  def test_processMultipleEpochs1
    numberOfEpochs = 1
    @aNetwork.calcCountRequiredToDropMSETo0 = numberOfEpochs
    @aSequencer.processMultipleEpochs
    @arrayOfLearners.each do |aLearner|
      assert_equal(1*numberOfEpochs, aLearner.startCount, "a learner was not issued the finishBatch")
      assert_equal(4*numberOfEpochs, aLearner.propCount, "a learner was not issued the proper number of 'propagate' command")
      assert_equal(1*numberOfEpochs, aLearner.finishCount, "a learner was not issued the finishBatch")
    end
    assert_equal(1*numberOfEpochs, @aNetwork.calcCount, "a learner was not issued the finishBatch")
  end

  def test_processMultipleEpochs2
    numberOfEpochs = 100
    @aNetwork.calcCountRequiredToDropMSETo0 = numberOfEpochs
    @aSequencer.processMultipleEpochs
    @arrayOfLearners.each do |aLearner|
      assert_equal(1*numberOfEpochs, aLearner.startCount, "a learner was not issued the finishBatch")
      assert_equal(4*numberOfEpochs, aLearner.propCount, "a learner was not issued the proper number of 'propagate' command")
      assert_equal(1*numberOfEpochs, aLearner.finishCount, "a learner was not issued the finishBatch")
    end
    assert_equal(1*numberOfEpochs, @aNetwork.calcCount, "a learner was not issued the finishBatch")
  end
end
