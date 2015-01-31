#FunctionalTestOfMixtureOfModels.rb

require 'rubygems'
require 'mathn'
require 'bundler/setup'
require_relative 'Utilities'
require 'distribution'



#def suppress?(aValue, reverse)
#  returnValue = aValue
#
#  case reverse
#    when false
#      returnValue
#    when true
#      !returnValue
#  end
#end
#
#reverse = false
#aValue = true
#
#puts suppress?(aValue, reverse)



#  For testing "dup" method for "unextend"
#module ModuleA
#  def methA
#    puts @varB
#  end
#
#  def methB
#    puts @varA
#  end
#end
#
#class A
#  def initialize
#    @varA = 1.0
#    @varB = 2.0
#  end
#
#  def methA
#    puts @varA
#  end
#end
#
#anA = A.new
#anA.methA
#puts
#
#anAextended = anA.extend(ModuleA)
#anAextended.methA
#anAextended.methB
#puts
#
#dupedanAextended = anAextended.clone
#dupedanAextended.methA
##dupedanAextended.methB


# testing multiple extensions...
module ModuleA
  def methA
    puts @varB
  end

  def methB
    puts @varA
  end
end

module ModuleB
  def methA
    #super
    puts @varA
    super
  end
  def methC
    puts "HI there"
  end
end


class A
  def initialize
    @varA = 1.0
    @varB = 2.0
  end

  def methA
    puts @varA
  end
end

#anA = A.new
#anA.methA
#puts
#
#anAextended = anA.extend(ModuleA)
#anAextended.methA
## anAextended.methB
#puts
#
#
#aBextended = anAextended.extend(ModuleB)
#aBextended.methA
#aBextended.methC
#puts


arrayOfNeurons = [A.new, A.new, A.new]

# arrayOfNeurons.each { |aNeuron| aNeuron.extend(ModuleA) }


# arrayOfNeurons.collect! { |aNeuron| aNeuron.dup }

#arrayOfNeurons.replace(arrayOfDupedNeurons)

arrayOfNeurons.each { |aNeuron| aNeuron.methA }

id = 2
puts  id.odd?