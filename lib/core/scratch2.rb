#class DummyLink
#  def initialize(aWeightObject)
#    @weight = aWeightObject
#  end
#
#  def weight
#    case @weight
#      when Float
#        return @weight
#      when SharedWeight
#        return @weight.value
#      else
#        raise TypeError.new("Wrong Class for link's weight:  #{@weight.inspect} ")
#    end
#  end
#
#  def weight=(someObject)
#    case @weight
#      when Float
#        @weight = someObject
#      when SharedWeight
#        raise TypeError.new("Wrong Class used to set link's weight #{someObject.inspect} ") unless (someObject.class == Float)
#        @weight.value = someObject
#      else
#        raise TypeError.new("Wrong Class for link's weight #{@weight.inspect} ")
#    end
#    return someObject
#  end
#
#  def plusOne
#    self.weight = weight + 1.0
#  end
#
#  def plusOneReversed
#    self.weight = 1.0 + weight
#  end
#
#  def tellMe
#    puts @weight.class
#  end
#
#end
#
#
#class SharedWeight
#  attr_accessor :value
#
#  def initialize(aValueForTheWeight)
#    @value = aValueForTheWeight
#  end
#end
#
#aWrappedWeight = SharedWeight.new(5.0)
#
#
#puts DummyLink.new(8.0).weight
#puts DummyLink.new(aWrappedWeight).weight
#
#aLink = DummyLink.new(9.0)
#aLink.plusOne
#puts aLink.weight
#aLink.tellMe
#puts
#
#
#tieWeight = SharedWeight.new(20.0)
#aLink.weight = tieWeight
#aLink.plusOne
#puts aLink.weight
#aLink.plusOneReversed
#puts aLink.weight
#aLink.tellMe
#
#
#
#
#puts
#aLink2 = DummyLink.new(tieWeight)
#aLink2.plusOne
#puts aLink2.weight
#aLink2.plusOneReversed
#puts aLink2.weight
#aLink2.tellMe
#
#
#puts aLink.weight
#puts aLink2.weight