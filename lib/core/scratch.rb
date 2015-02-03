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
#module ModuleB
#  def methA
#    #super
#    puts @varA
#    super
#  end
#  def methC
#    puts "HI there"
#  end
#end
#
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


#arrayOfNeurons = [A.new, A.new, A.new]
#
## arrayOfNeurons.each { |aNeuron| aNeuron.extend(ModuleA) }
#
#
## arrayOfNeurons.collect! { |aNeuron| aNeuron.dup }
#
##arrayOfNeurons.replace(arrayOfDupedNeurons)
#
#arrayOfNeurons.each { |aNeuron| aNeuron.methA }
#
#id = 2
#puts  id.odd?

#ary = []
#
#ary[0] = 0
#ary[1] = 1
#ary[10] = 10
#
##puts ary.flatten
#ary.clear
#puts ary.length


include Distribution
include Distribution::Shorthand

def gaussPdf(x, mean, std)
  normalizedDeviationFromMean = (x - mean) / std
return norm_pdf(normalizedDeviationFromMean)  / std # NOTE: the .abs gets rid of imaginary results in some cases
end



def gaussPdf2(x, mean, std)
  std = 1e-20 if(std < 1e-20)   # 1e-20 does NOT work.  You get NaN in simulation results
  normalizedDeviationFromMean = (x - mean) / std
  return norm_pdf(normalizedDeviationFromMean)  / std # NOTE: the .abs gets rid of imaginary results in some cases
end

def gaussPdf3(x, mean, std)
  diff = x - mean
  ratio = diff / std
  std = 1e-10 if(std < 1e-10)   # 1e-20 does NOT work.  You get NaN in simulation results
  diff = ratio * std
  # puts "std=\t#{std}\tdiff=\t#{diff}\n"
  normalizedDeviationFromMean = diff / std
  return norm_pdf(normalizedDeviationFromMean)  / std
end





std = 1e-20
mean = 1e-19 # ==> 0.0  because diff is 100 larger than std
puts "#{gaussPdf(0.0, mean, std)}\t#{gaussPdf2(0.0, mean, std)}   "


std = 1e-17
mean = 1e-16 # ==> 0.0  because diff is 100 larger than std
puts "#{gaussPdf(0.0, mean, std)}\t#{gaussPdf2(0.0, mean, std)}   "


std = 1e-10
mean = 1e-9 # ==> 0.0  because diff is 100 larger than std
puts "#{gaussPdf(0.0, mean, std)}\t#{gaussPdf2(0.0, mean, std)}   "

std = 1.0
mean = 10.0 # ==> 0.0  because diff is 100 larger than std
puts "#{gaussPdf(0.0, mean, std)}\t#{gaussPdf2(0.0, mean, std)}   "

std = 1e10
mean = 1e11 # ==> 0.0  because diff is 100 larger than std
puts "#{gaussPdf(0.0, mean, std)}\t#{gaussPdf2(0.0, mean, std)}   "



# puts gaussPdf(0.0, mean, std)