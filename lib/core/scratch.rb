require_relative 'Utilities'
require_relative 'DataSet'
require_relative 'NeuralIOFunctions'


#class SimObj
#  attr_accessor :output
#  def initialize
#    @output = 20.5
#  end
#end
#
#class Simple
#  def initialize
#    @var1 = 1
#  end
#
#  def classMethod
#    puts "classMethod"
#  end
#end
#
#
#module SimpleModule
#  attr_accessor :aModuleVariable
#
#  def moduleMethod
#   puts 'test'
#  end
#
#end
#
#
##class Simple
##  include SimpleModule
##end
#
#oo = SimObj.new
#
#aSimpleClass = Simple.new
#aSimpleClass.extend(SimpleModule)
#
#
##aSimpleClass.moduleMethod
#
#aSimpleClass.aModuleVariable= oo
#
#puts "variable= #{aSimpleClass.aModuleVariable.output}"
## aSimpleClass.classMethod


class A
  def say
    puts "This is class A"
  end
end

module B
  def say
    puts "This is module B"
  end
end


anA = A.new

anA.extend(B)

anA.say