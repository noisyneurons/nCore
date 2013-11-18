### VERSION "nCore"
## ../nCore/test/TestDataSet.rb

require "test/unit"
# require 'minitest/reporters'
# MiniTest::Reporters.use!

require_relative '../lib/core/DataSet'

Tolerance = 0.00001

include ExampleDistribution

class DummyBaseNeuron
  attr_accessor :keyToExampleData, :arrayOfSelectedData

  def initialize
    @keyToExampleData = nil
    @arrayOfSelectedData = []
  end
end

class DummyInputNeuron < DummyBaseNeuron
  def initialize
    super
    @keyToExampleData = :inputs
  end
end

class DummyOutputNeuron < DummyBaseNeuron
  def initialize
    super
    @keyToExampleData = :targets
  end
end

############################################################      N
class DataSetTest < MiniTest::Unit::TestCase


  def setup
    @examples = []
    @examples << {:inputs => [20.0, 22.0], :targets => [60.0, 62.0], :exampleNumber => 0}
    @examples << {:inputs => [30.0, 33.0], :targets => [70.0, 73.0], :exampleNumber => 1}
    @examples << {:inputs => [40.0, 44.0], :targets => [80.0, 84.0], :exampleNumber => 2}

    @arrayOfInputNeurons = Array.new(2) { |i| DummyInputNeuron.new }
    @firstInputNeuron = @arrayOfInputNeurons[0]
    @secondInputNeuron = @arrayOfInputNeurons[1]
    @arrayOfOutputNeurons = Array.new(2) { |i| DummyOutputNeuron.new }
    @firstOutputNeuron = @arrayOfOutputNeurons[0]
    @secondOutputNeuron = @arrayOfOutputNeurons[1]

  end

  def test_distributeDataToInputAndOutputNeurons1
    distributeDataToInputAndOutputNeurons(@examples, [@arrayOfInputNeurons])
    assert_equal(20.0, @firstInputNeuron.arrayOfSelectedData[0], "wrong 1st example input stored in inputNeuron 0")
  end

  def test_distributeDataToInputAndOutputNeurons2
    distributeDataToInputAndOutputNeurons(@examples, [@arrayOfInputNeurons])
    assert_equal(30.0, @firstInputNeuron.arrayOfSelectedData[1], "wrong 2nd example input stored in inputNeuron 0")
  end

  def test_distributeDataToInputAndOutputNeurons3
    distributeDataToInputAndOutputNeurons(@examples, [@arrayOfInputNeurons])
    assert_equal(22.0, @secondInputNeuron.arrayOfSelectedData[0], "wrong 1st example input stored in inputNeuron 1")
  end

  def test_distributeDataToInputAndOutputNeurons4
    distributeDataToInputAndOutputNeurons(@examples, [@arrayOfInputNeurons])
    assert_equal(44.0, @secondInputNeuron.arrayOfSelectedData[2], "wrong 3rd example input stored in inputNeuron 1")
  end

  def test_distributeDataToInputAndOutputNeurons5
    distributeDataToInputAndOutputNeurons(@examples, [@arrayOfOutputNeurons, @arrayOfInputNeurons])
    assert_equal(44.0, @secondInputNeuron.arrayOfSelectedData[2], "wrong 3rd example input stored in inputNeuron 1")
  end

  def test_distributeDataToInputAndOutputNeurons6
    distributeDataToInputAndOutputNeurons(@examples, [@arrayOfOutputNeurons, @arrayOfInputNeurons])
    assert_equal(84.0, @secondOutputNeuron.arrayOfSelectedData[2], "wrong 3rd example input stored in outputNeuron 1")
  end

  def test_distributeDataToInputAndOutputNeurons7
    distributeDataToInputAndOutputNeurons(@examples, [@arrayOfOutputNeurons, @arrayOfInputNeurons])
    assert_equal(70.0, @firstOutputNeuron.arrayOfSelectedData[1], "wrong 2nd example input stored in outputNeuron 0")
  end
end
