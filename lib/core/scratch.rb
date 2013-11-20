require 'forwardable'

class DummyLink
  attr_accessor :aWeight
  def initialize(aWeight)
    @aWeight = aWeight
  end

  def plusOne
    self.aWeight = aWeight + 1.0
  end
end

class WrappedWeight
  attr_reader :value
  def initialize(aValueForTheWeight)
    @value = aValueForTheWeight
  end

  def +(aValue)
    @value = @value + aValue
    return self
  end

end



aWrappedWeight = WrappedWeight.new(5.0)
puts aWrappedWeight

link1 = DummyLink.new(aWrappedWeight)
link2 = DummyLink.new(aWrappedWeight)

link1.plusOne
puts link1.aWeight.value
link2.plusOne
puts link2.aWeight.value
puts link1.aWeight.value
link1.plusOne
puts link2.aWeight.value
puts link1.aWeight.value







#
#class WrappedWeight  < DelegateClass(Float)
#  attr_accessor :theWeight
#
#  def initialize(theWeight)
#    @theWeight = theWeight
#  end
#
#  def self
#    return theWeight
#  end
#
#  def addOne
#    self.aFloat = theWeight + 1.0
#  end
#end
#
#wrapW = WrappedWeight.new(5.0)
#wrapW = wrapW + wrapW
#
#puts wrapW.self.to_s