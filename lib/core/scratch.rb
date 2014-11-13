require 'rubygems'
require 'bundler/setup'


#require_relative 'Utilities'
#require_relative 'DataSet'
#require_relative 'NeuralIOFunctions'

#
#
#require 'statsample'
#
## require '/home/mark/usr/local/ruby2.1.3/ruby/lib/ruby/gems/2.1.0/gems/statsample'
## Note R like generation of random gaussian variable
## and correlation matrix
#
#ss_analysis("Statsample::Bivariate.correlation_matrix") do
#  samples=1000
#  ds=data_frame(
#      'a'=>rnorm(samples),
#      'b'=>rnorm(samples),
#      'c'=>rnorm(samples),
#      'd'=>rnorm(samples))
#  cm=cor(ds)
#  summary(cm)
#end
#
#Statsample::Analysis.run_batch # Echo output to console

module AddToArray
  def testMod
    puts "still extended: #{self.to_s}"
  end

  def -(otherArray)
    resultantArray = super
    resultantArray.extend(AddToArray)
  end

  def +(otherArray)
    resultantArray = super
    resultantArray.extend(AddToArray)
  end

  def <<(item)
    resultantArray = super
    resultantArray.extend(AddToArray)
  end
end


a = [1,2,3]
a.extend(AddToArray)
a.testMod
b = [3,4,5]
b.extend(AddToArray)
b.testMod
c = a + b
c.testMod
c << 4
c.testMod


#class ArrayS < Array
#  def testMod
#    puts "still extended: #{self.to_s}"
#    puts "class= #{self.class}"
#  end
#
#  def -(otherArray)
#    result = super
#    ArrayS.new(result)
#  end
#end
#
#
#a = ArrayS.new([1, 2, 3])
#a.testMod
#b = ArrayS.new([3, 4, 5])
#b.testMod
#
#d = a - b
#puts "class of d= #{d.class}"
#d.testMod

#c = ArrayS.new(d)
#c.testMod
