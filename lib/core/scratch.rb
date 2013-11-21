require 'forwardable'

class DummyLink
  def initialize(aWeight)
    @weight = aWeight
  end

  def weight=(aValue)

    case aValue
      when Numeric

    end
    @weight = value
  end

  def weight
    return @weight
  end



  def plusOne
    self.aWeight = aWeight + 1.0
  end

  def plusOneReversed
    self.aWeight = 1.0 + aWeight
  end
end

class WrappedWeight
  attr_reader :value

  def initialize(aValueForTheWeight)
    @value = aValueForTheWeight
  end

  def self=(anObject)
    case anObject
      when WrappedWeight
        @value = anObject.value

      when Numeric
        @value = anObject

      else
        raise TypeError.new("Cannot coerce #{otherValue.inspect} to a WrappedWeight")
    end
    return self
  end

  def +(otherValue)
    case otherValue
      when WrappedWeight
        @value = @value + otherValue.value
      when Numeric
        @value = @value + otherValue
      else
        raise TypeError.new("Cannot coerce #{otherValue.inspect} to a WrappedWeight")
    end
    return self
  end

  def -(otherValue)
    case otherValue
      when WrappedWeight
        @value = @value - otherValue.value
      when Numeric
        @value = @value - otherValue
      else
        raise TypeError.new("Cannot coerce #{otherValue.inspect} to a WrappedWeight")
    end
    return self
  end

  def *(otherValue)
    case otherValue
      when WrappedWeight
        @value = @value * otherValue.value
      when Numeric
        @value = @value * otherValue
      else
        raise TypeError.new("Cannot coerce #{otherValue.inspect} to a WrappedWeight")
    end
    return self
  end

  def /(otherValue)
    case otherValue
      when WrappedWeight
        @value = @value / otherValue.value
      when Numeric
        @value = @value / otherValue
      else
        raise TypeError.new("Cannot coerce #{otherValue.inspect} to a WrappedWeight")
    end
    return self
  end


  def coerce(other)
    case other
      when Numeric
        [self, WrappedWeight.new(other)]
      else
        raise TypeError.new("Cannot coerce #{other.inspect} to a WrappedWeight")
    end
  end
end



aWrappedWeight = WrappedWeight.new(5.0)
puts aWrappedWeight.value
self.aWrappedWeight = 6.0
puts aWrappedWeight.value


#
#link1 = DummyLink.new(aWrappedWeight)
#link2 = DummyLink.new(aWrappedWeight)
#
#link1.plusOne
#puts link1.aWeight.value
#link2.plusOne
#puts link2.aWeight.value
#link1.plusOne
#puts link2.aWeight.value
#
#puts
#link1.plusOneReversed
#puts link2.aWeight.value
#puts link1.aWeight.value
#
#puts
#link2.plusOneReversed
#puts link2.aWeight.value
#puts link1.aWeight.value
#
#
#wr1 = WrappedWeight.new(5.0)
#wr2 = WrappedWeight.new(2.0)
#
#puts
#wr3 = wr1 + wr2
#puts wr1.value
#puts wr3.value
#
#wr4 = wr1 - wr2
#puts wr1.value
#puts wr4.value
#
#wr5 = wr1 * wr2
#puts wr1.value
#puts wr5.value
#
#wr6 = wr1 / wr2
#puts wr1.value
#puts wr6.value
#
#
##
##class WrappedWeight  < DelegateClass(Float)
##  attr_accessor :theWeight
##
##  def initialize(theWeight)
##    @theWeight = theWeight
##  end
##
##  def self
##    return theWeight
##  end
##
##  def addOne
##    self.aFloat = theWeight + 1.0
##  end
##end
##
##wrapW = WrappedWeight.new(5.0)
##wrapW = wrapW + wrapW
##
##puts wrapW.self.to_s